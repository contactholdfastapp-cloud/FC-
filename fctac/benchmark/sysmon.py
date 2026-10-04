"""Background system monitor: CPU / RAM / GPU utilisation / VRAM.

Uses whatever is available (psutil, pynvml, or ``nvidia-smi`` polling every
couple of seconds); missing sources are simply reported as None.  Never in
the hot path: values are sampled on a background thread.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time


class SysMonitor:
    def __init__(self, period_s: float = 1.0):
        self.period = period_s
        self.latest: dict = {}
        self._stop = threading.Event()
        self._t = None
        try:
            import psutil
            self.psutil = psutil
            self.proc = psutil.Process(os.getpid())
            psutil.cpu_percent(None)
            self.proc.cpu_percent(None)
        except Exception:
            self.psutil = None
        self.nvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.nvh = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.smi = shutil.which("nvidia-smi")

    def start(self):
        self._t = threading.Thread(target=self._run, daemon=True, name="sysmon")
        self._t.start()
        return self

    def stop(self):
        self._stop.set()

    def sample(self) -> dict:
        d: dict = {}
        if self.psutil is not None:
            d["cpu_pct"] = self.psutil.cpu_percent(None)
            d["proc_cpu_pct"] = self.proc.cpu_percent(None)
            vm = self.psutil.virtual_memory()
            d["ram_used_gb"] = round((vm.total - vm.available) / 2 ** 30, 2)
            d["proc_ram_gb"] = round(self.proc.memory_info().rss / 2 ** 30, 2)
        elif os.path.exists("/proc/loadavg"):
            d["load1"] = float(open("/proc/loadavg").read().split()[0])
        if self.nvml is not None:
            u = self.nvml.nvmlDeviceGetUtilizationRates(self.nvh)
            m = self.nvml.nvmlDeviceGetMemoryInfo(self.nvh)
            d["gpu_pct"] = u.gpu
            d["vram_used_mb"] = round(m.used / 2 ** 20)
            d["vram_total_mb"] = round(m.total / 2 ** 20)
        elif getattr(self, "smi", None):
            try:
                out = subprocess.run([self.smi, "--query-gpu=utilization.gpu,memory.used,memory.total",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout
                g, mu, mt = [x.strip() for x in out.splitlines()[0].split(",")]
                d["gpu_pct"], d["vram_used_mb"], d["vram_total_mb"] = float(g), float(mu), float(mt)
            except Exception:
                pass
        return d

    def _run(self):
        while not self._stop.is_set():
            self.latest = self.sample()
            time.sleep(self.period)

    def line(self) -> str:
        d = self.latest
        parts = []
        if "cpu_pct" in d:
            parts.append(f"CPU {d['cpu_pct']:.0f}% (proc {d['proc_cpu_pct']:.0f}%)")
        if "proc_ram_gb" in d:
            parts.append(f"RAM {d['proc_ram_gb']:.2f} GB")
        if "gpu_pct" in d:
            parts.append(f"GPU {d['gpu_pct']:.0f}%  VRAM {d['vram_used_mb']:.0f}/{d['vram_total_mb']:.0f} MB")
        if "load1" in d:
            parts.append(f"load {d['load1']:.2f}")
        return "  ".join(parts) or "system stats unavailable"
