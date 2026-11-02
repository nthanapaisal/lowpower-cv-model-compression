#!/usr/bin/env python3
"""
CS 6301 Low-Power Computer Vision
Assignment 2 course-supplied telemetry helper. Do not modify this file.

What it does
------------
tegrastats runs on the Jetson host as a persistent systemd service and writes
raw telemetry to /opt/lpcv/power.log. The lpcv container exposes that file
read-only as /course-power.log.

tegrastats timestamps have only one-second resolution, while samples arrive
roughly every 100 ms. This helper therefore isolates an interval by file-byte
offsets, exactly as the Assignment 1 benchmark did: it records the size of the
log when the interval begins and when it ends, and parses only the lines
written in between. The timestamps inside the log are not used.

One refinement over Assignment 1: only complete lines that lie entirely inside
the interval are counted. A line that was only partly written when the
interval began or ended straddles the boundary and is excluded.

Usage
-----
    import time
    from power_window import PowerWindow

    with PowerWindow() as idle:          # idle baseline, before any inference
        time.sleep(10)

    # ... warm-up, Part C, Part D ...

    with PowerWindow() as busy:          # sustained energy loop (>= 20 s)
        ...                              # synchronized inferences; count N

    idle.mean_mw        # mean VDD_IN over the interval, in mW
    idle.n_samples      # number of telemetry samples in the interval
    idle.duration_s     # wall-clock length of the interval (perf_counter)
    idle.samples        # raw tegrastats lines used
    idle.append_to("power.log", "DW-t6 idle")   # save raw lines for submission

    The explicit form is also available:

    w = PowerWindow().start()
    ...
    w.stop()
"""

import re
import statistics
import time
import warnings
from pathlib import Path

__all__ = ["PowerWindow", "PowerWindowError", "DEFAULT_POWER_LOG"]

DEFAULT_POWER_LOG = Path("/course-power.log")

# VDD_IN is reported as "VDD_IN <instantaneous>mW/<running average>mW".
# The instantaneous value is used, as in Assignment 1.
_VDD_IN = re.compile(r"VDD_IN\s+(\d+)mW")

# tegrastats in the course service samples roughly every 100 ms. A window whose
# sample rate falls far below this suggests a stalled service.
_EXPECTED_INTERVAL_S = 0.1
_MIN_RATE_FRACTION = 0.5


class PowerWindowError(RuntimeError):
    """Raised when telemetry for an interval cannot be trusted."""


class PowerWindow:
    """Collect host VDD_IN telemetry for one interval of the benchmark."""

    def __init__(self, log_path=DEFAULT_POWER_LOG):
        self.log_path = Path(log_path)
        self.start_offset = None
        self.end_offset = None
        self.duration_s = None
        self.samples = []      # raw tegrastats lines, in log order
        self.powers_mw = []    # instantaneous VDD_IN of each sample, in mW
        self._t0 = None

    def start(self):
        if self.start_offset is not None:
            raise PowerWindowError("This PowerWindow has already been started.")
        self._require_log()
        self.start_offset = self._log_size()
        self._t0 = time.perf_counter()
        return self

    def stop(self):
        if self.start_offset is None:
            raise PowerWindowError("stop() called before start().")
        if self.end_offset is not None:
            raise PowerWindowError("This PowerWindow has already been stopped.")
        self.duration_s = time.perf_counter() - self._t0
        self.end_offset = self._log_size()
        self._read_interval()
        self._check_sample_rate()
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            return False
        self.stop()
        return False

    @property
    def n_samples(self):
        return len(self.powers_mw)

    @property
    def mean_mw(self):
        if not self.powers_mw:
            raise PowerWindowError("No VDD_IN samples in this interval.")
        return statistics.fmean(self.powers_mw)

    def append_to(self, path, label):
        """Append this interval's raw lines to a student power log."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(f"# {label}: {self.n_samples} samples, "
                    f"{self.duration_s:.3f} s, mean VDD_IN "
                    f"{self.mean_mw:.1f} mW\n")
            for line in self.samples:
                f.write(line + "\n")

    def _require_log(self):
        if not self.log_path.exists():
            raise PowerWindowError(
                f"Course power log not found: {self.log_path}\n"
                "The Jetson host must have the persistent tegrastats service "
                "running, and the lpcv container must expose its log."
            )
        if not self.log_path.is_file():
            raise PowerWindowError(
                f"Power log is not a regular file: {self.log_path}"
            )

    def _log_size(self):
        try:
            return self.log_path.stat().st_size
        except FileNotFoundError:
            raise PowerWindowError(
                f"Power log disappeared: {self.log_path}"
            ) from None

    def _read_interval(self):
        if self.end_offset < self.start_offset:
            raise PowerWindowError(
                "The host power log became smaller during the interval. This "
                "usually means log rotation occurred. Rerun the measurement."
            )

        with self.log_path.open("rb") as f:
            if self.start_offset > 0:
                f.seek(self.start_offset - 1)
                starts_on_boundary = f.read(1) == b"\n"
            else:
                starts_on_boundary = True
            f.seek(self.start_offset)
            data = f.read(self.end_offset - self.start_offset)

        text = data.decode("utf-8", errors="replace")
        lines = text.split("\n")
        lines = lines[:-1]

        if not starts_on_boundary and lines:
            lines = lines[1:]

        for line in lines:
            match = _VDD_IN.search(line)
            if match:
                self.samples.append(line)
                self.powers_mw.append(float(match.group(1)))

        if not self.powers_mw:
            raise PowerWindowError(
                "No VDD_IN samples were found in this interval. Check that the "
                "persistent tegrastats service is running, and make the "
                "interval at least several seconds long."
            )

    def _check_sample_rate(self):
        expected = self.duration_s / _EXPECTED_INTERVAL_S
        if expected >= 10 and self.n_samples < _MIN_RATE_FRACTION * expected:
            warnings.warn(
                f"Only {self.n_samples} telemetry samples in "
                f"{self.duration_s:.1f} s (about {expected:.0f} expected). "
                "The tegrastats service may have stalled; report this in "
                "README.md and consider rerunning.",
                RuntimeWarning,
                stacklevel=3,
            )


if __name__ == "__main__":
    print(f"Reading {DEFAULT_POWER_LOG} for 5 s ...")
    with PowerWindow() as w:
        time.sleep(5)
    print(f"{w.n_samples} samples, mean VDD_IN {w.mean_mw:.1f} mW, "
          f"duration {w.duration_s:.2f} s")
