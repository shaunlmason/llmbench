import atexit
import os
import socket
import subprocess
import time

import requests

from .config import HEALTH_CHECK_INTERVAL, HEALTH_CHECK_TIMEOUT, SERVICE_NAMES


def get_service_names(gpu_config: str) -> list[str]:
    """Map gpu config to systemctl service name(s)."""
    names = SERVICE_NAMES[gpu_config]
    return names if isinstance(names, list) else [names]


def _service_not_found(output: str) -> bool:
    output = output.lower()
    return "not loaded" in output or "could not be found" in output


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for_port_release(port: int, timeout: int = 10) -> bool:
    """Wait for a TCP port to become available."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _port_in_use(port):
            return True
        time.sleep(0.25)
    return not _port_in_use(port)


def stop_services(gpu_config: str) -> list[str]:
    """Stop llama-server systemctl service(s). Returns list of stopped service names."""
    stopped_services = []
    services = get_service_names(gpu_config)
    try:
        for svc in services:
            status = subprocess.run(
                ["systemctl", "is-active", "--quiet", svc],
                capture_output=True,
                text=True,
            )
            status_output = f"{status.stdout}\n{status.stderr}".strip()
            if status.returncode != 0:
                if _service_not_found(status_output):
                    print(f"Skipping missing service {svc}...")
                elif status.stderr.strip():
                    raise RuntimeError(
                        f"Failed to query service {svc}: {status.stderr.strip()}"
                    )
                else:
                    print(f"{svc} is not active, leaving it unchanged.")
                continue

            print(f"Stopping {svc}...")
            result = subprocess.run(
                ["sudo", "systemctl", "stop", svc],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                message = result.stderr.strip() or result.stdout.strip() or "unknown error"
                if _service_not_found(message):
                    print(f"Skipping missing service {svc}...")
                    continue
                raise RuntimeError(f"Failed to stop {svc}: {message}")
            stopped_services.append(svc)
    except Exception:
        if stopped_services:
            restore_services(stopped_services)
        raise
    return stopped_services


def restore_services(service_names: list[str]):
    """Restart previously stopped services."""
    for svc in service_names:
        print(f"Restoring {svc}...")
        subprocess.run(
            ["sudo", "systemctl", "start", svc],
            capture_output=True,
            text=True,
        )


def start_llama_server(
    model_path: str,
    gpu_config: str,
    context_length: int,
    port: int,
) -> subprocess.Popen:
    """Start llama-server with the given model and GPU configuration."""
    if not wait_for_port_release(port):
        raise RuntimeError(
            f"Port {port} is still in use. Refusing to start a new llama-server."
        )

    cmd = [
        "llama-server",
        "-m", str(model_path),
        "--port", str(port),
        "-c", str(context_length),
        "-ngl", "999",
    ]

    env = os.environ.copy()

    if gpu_config == "gpu0":
        env["CUDA_VISIBLE_DEVICES"] = "0"
    elif gpu_config == "gpu1":
        env["CUDA_VISIBLE_DEVICES"] = "1"
    elif gpu_config == "both":
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
        cmd.extend(["--tensor-split", "1,1"])

    print(f"Starting llama-server on {gpu_config}: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Register cleanup as safety net
    def _cleanup():
        if process.poll() is None:
            process.terminate()

    atexit.register(_cleanup)
    return process


def wait_for_health(
    port: int,
    process: subprocess.Popen,
    timeout: int = HEALTH_CHECK_TIMEOUT,
) -> bool:
    """Poll the health endpoint until the server is ready."""
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    print(f"Waiting for server at {url}...", end="", flush=True)

    while time.time() < deadline:
        returncode = process.poll()
        if returncode is not None:
            print(f" exited early (code {returncode})!")
            return False
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200 and process.poll() is None:
                print(" ready!")
                return True
        except requests.RequestException:
            pass
        print(".", end="", flush=True)
        time.sleep(HEALTH_CHECK_INTERVAL)

    returncode = process.poll()
    if returncode is not None:
        print(f" exited early (code {returncode})!")
    else:
        print(" timeout!")
    return False


def stop_llama_server(process: subprocess.Popen):
    """Gracefully stop the llama-server process."""
    if process.poll() is not None:
        return
    print("Stopping llama-server...")
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
