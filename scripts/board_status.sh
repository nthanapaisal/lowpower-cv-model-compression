#!/usr/bin/env bash

set -u

if [[ $# -ne 1 || ( "$1" != "training" && "$1" != "measurement" ) ]]; then
    echo "Usage: $0 training|measurement" >&2
    exit 2
fi

role="$1"

echo "=== Logged-in users ==="
who

echo
echo "=== Uptime and load ==="
uptime

echo
echo "=== NVIDIA GPU status ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "nvidia-smi is not installed; that is common on Jetson."
    echo "The environment check below reports the GPU through PyTorch."
fi
