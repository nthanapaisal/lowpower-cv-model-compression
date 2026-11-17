#!/usr/bin/env python3
"""Report the Lab 1 board and software environment required by the handout."""

import argparse
from importlib import metadata
import os
from pathlib import Path
import platform
import socket
import subprocess
import re


COURSE_ROOT = Path("/course-data")
HELPER_ROOT = COURSE_ROOT / "lab1"
BOARD_ROLES = {
    "jetson-lpcv-01": "training",
    "jetson-lpcv-02": "training",
    "jetson-lpcv-03": "measurement",
    "jetson-lpcv-04": "measurement",
}
failures = 0


def report(status, name, value):
    global failures
    print(f"[{status}] {name}: {value}")
    if status == "FAIL":
        failures += 1


def read_text(path):
    try:
        return Path(path).read_text().replace("\x00", " ").strip()
    except OSError:
        return None


def command_output(command):
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=False
        )
        return (result.stdout + result.stderr).strip(), result.returncode
    except (OSError, subprocess.TimeoutExpired):
        return "", 1


def package_version(package):
    if package == "tensorrt":
        try:
            import tensorrt

            return tensorrt.__version__
        except (ImportError, AttributeError):
            pass
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def inside_container():
    cgroup = read_text("/proc/1/cgroup") or ""
    return (
        Path("/.dockerenv").exists()
        or Path("/run/.containerenv").exists()
        or any(marker in cgroup for marker in ("docker", "containerd", "kubepods"))
    )


def check_board(role):
    hostname = socket.gethostname().split(".")[0]
    detected_role = BOARD_ROLES.get(hostname)
    report("INFO", "Hostname", hostname)
    if detected_role:
        report("PASS", "Board role", detected_role)
        if role and role != detected_role:
            report("FAIL", "Requested role", f"expected {role}, host is {detected_role}")
    elif role and inside_container():
        report("WARN", "Board role", f"container hides board hostname; requested {role}, confirm the host before entry")
    elif role:
        report("WARN", "Board role", f"hostname is not a listed course board; requested {role}")
    else:
        report("WARN", "Board role", "unknown hostname; pass --role training or --role measurement")

    model = read_text("/proc/device-tree/model")
    if model and "orin nano" in model.lower():
        report("PASS", "Jetson model", model)
    elif model:
        report("FAIL", "Jetson model", f"expected Orin Nano, found {model}")
    elif inside_container():
        report("WARN", "Jetson model", "device-tree model is not exposed inside lpcv; check it on the board host")
    else:
        report("FAIL", "Jetson model", "device-tree model not found; run on a Jetson")

    dts = read_text("/proc/device-tree/nvidia,dtsfilename")
    report("INFO" if dts else "WARN", "Carrier-board device-tree", dts or "not available")

    l4t = read_text("/etc/nv_tegra_release")
    report("PASS" if l4t else "FAIL", "JetPack/L4T release", l4t or "not found")
    jetpack, _ = command_output(
        ["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"]
    )
    report("INFO" if jetpack else "WARN", "JetPack package", jetpack or "not installed or not queryable")


def check_container_and_course_files():
    in_container = inside_container()
    container_status = "container marker found" if in_container else "not detected; confirm you entered lpcv"
    report("PASS" if in_container else "WARN", "Container", container_status)

    required_paths = [
        HELPER_ROOT / "lab1_trt.py",
        HELPER_ROOT / "power_window.py",
        HELPER_ROOT / "splits" / "fit.txt",
        HELPER_ROOT / "splits" / "tune.txt",
        HELPER_ROOT / "splits" / "calib.txt",
        COURSE_ROOT / "imagewoof2-320" / "train",
        COURSE_ROOT / "imagewoof2-320" / "val",
    ]
    for path in required_paths:
        report("PASS" if path.exists() else "FAIL", "Course path", f"{path} {'present' if path.exists() else 'missing'}")


def check_software():
    report("INFO", "Python", platform.python_version())
    for package in ("torch", "torchvision", "timm", "tensorrt"):
        version = package_version(package)
        report("PASS" if version else "FAIL", package, version or "not installed")
        if package == "tensorrt" and version and "10.11.0.33" not in version:
            report("FAIL", "TensorRT requirement", f"PDF specifies 10.11.0.33; found {version}")

    try:
        import torch

        cuda_version = torch.version.cuda or "not available"
        report("INFO", "PyTorch CUDA version", cuda_version)
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            report("PASS", "CUDA GPU", torch.cuda.get_device_name(device))
            report("INFO", "GPU compute capability", torch.cuda.get_device_capability(device))
        else:
            report("FAIL", "CUDA GPU", "PyTorch reports CUDA unavailable")
    except Exception as error:
        report("FAIL", "PyTorch CUDA check", str(error))


def check_power_mode():
    power_mode = os.environ.get("LPCV_POWER_MODE")
    if power_mode:
        normalized = power_mode.upper().replace(" ", "")
        status = "PASS" if normalized in ("15W", "15") else "FAIL"
        report(status, "LPCV_POWER_MODE", power_mode)
    else:
        report("FAIL", "LPCV_POWER_MODE", "not set; PDF expects 15 W")

    output, code = command_output(["nvpmodel", "-q"])
    if code == 0 and output:
        report("INFO", "nvpmodel -q", output.replace("\n", " | "))
        if "15W" not in output.upper().replace(" ", ""):
            report("WARN", "nvpmodel mode", "output does not clearly show 15 W; compare with LPCV_POWER_MODE")
    else:
        report("WARN", "nvpmodel -q", "command unavailable")


def check_jetson_telemetry():
    """Read a short tegrastats sample, including the Jetson GR3D/EMC clocks."""
    try:
        result = subprocess.run(
            ["tegrastats", "--interval", "1000"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        sample = (result.stdout + result.stderr).strip().splitlines()
    except FileNotFoundError:
        report("WARN", "tegrastats", "command unavailable; GR3D/EMC telemetry not checked")
        return
    except subprocess.TimeoutExpired as error:
        sample = (error.stdout or b"").decode(errors="replace").splitlines()

    if not sample:
        report("WARN", "tegrastats", "no telemetry sample returned")
        return

    line = sample[-1]
    report("INFO", "tegrastats sample", line)
    for label, pattern in (
        ("GR3D frequency/utilization", r"GR3D_FREQ\s+[^ ]+"),
        ("EMC frequency/utilization", r"EMC_FREQ\s+[^ ]+"),
    ):
        match = re.search(pattern, line)
        report("PASS" if match else "WARN", label,
               match.group(0) if match else "not present in tegrastats sample")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("training", "measurement"), help="expected board role")
    args = parser.parse_args()

    print("Lab 1 environment check (read-only)")
    check_board(args.role)
    check_container_and_course_files()
    check_software()
    check_power_mode()
    check_jetson_telemetry()
    print("\nShared-board availability is not checked by this script.")
    print("For a training board, follow the course coordination rule; for a measurement board, confirm it is idle first.")
    print(f"\nResult: {'FAIL' if failures else 'PASS'} ({failures} failing check(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
