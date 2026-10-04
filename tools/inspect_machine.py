"""Phase 0: inspect the machine the assistant will run on.

Run this ON THE GAMING PC (Windows):

    python tools/inspect_machine.py            # writes machine_report.json
    python tools/inspect_machine.py --dxdiag   # also parses dxdiag (slower)

It reports OS, CPU, RAM, GPU/VRAM/driver/compute capability, CUDA, ONNX
Runtime providers, TensorRT, DirectML, Python + key libraries, C++
toolchain, DirectX, displays and disk, then derives a recommended runtime
configuration (inference provider, precision, detector resolution).
Nothing is installed or modified.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile

TURING_NO_TC = ("1650", "1660")   # Turing GTX: fast FP16 ALUs but no Tensor Cores


def run(cmd, timeout=20) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=isinstance(cmd, str))
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:      # noqa: BLE001
        return f"ERR {e}"


def powershell(q: str) -> str:
    return run(["powershell", "-NoProfile", "-Command", q]).strip()


def os_info() -> dict:
    d = {"platform": platform.platform(), "system": platform.system(), "release": platform.release(),
         "version": platform.version(), "machine": platform.machine()}
    if sys.platform == "win32":
        d["win32_ver"] = platform.win32_ver()
        d["edition"] = platform.win32_edition() if hasattr(platform, "win32_edition") else ""
    return d


def cpu_info() -> dict:
    d = {"logical_cores": os.cpu_count(), "processor": platform.processor()}
    if sys.platform == "win32":
        out = powershell("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,"
                         "NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json")
        try:
            d["win32_processor"] = json.loads(out)
        except Exception:
            d["win32_processor_raw"] = out[:500]
    elif os.path.exists("/proc/cpuinfo"):
        txt = open("/proc/cpuinfo").read()
        m = re.search(r"model name\s*:\s*(.*)", txt)
        d["model"] = m.group(1) if m else ""
        d["avx2"] = " avx2 " in txt
        d["avx512f"] = " avx512f " in txt
    return d


def ram_info() -> dict:
    try:
        import psutil  # optional
        vm = psutil.virtual_memory()
        return {"total_gb": round(vm.total / 2 ** 30, 1), "available_gb": round(vm.available / 2 ** 30, 1)}
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = MS()
        ms.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return {"total_gb": round(ms.ullTotalPhys / 2 ** 30, 1), "available_gb": round(ms.ullAvailPhys / 2 ** 30, 1)}
    if os.path.exists("/proc/meminfo"):
        txt = open("/proc/meminfo").read()
        tot = int(re.search(r"MemTotal:\s*(\d+)", txt).group(1))
        av = int(re.search(r"MemAvailable:\s*(\d+)", txt).group(1))
        return {"total_gb": round(tot / 2 ** 20, 1), "available_gb": round(av / 2 ** 20, 1)}
    return {}


def gpu_info() -> dict:
    d: dict = {"nvidia": []}
    smi = shutil.which("nvidia-smi")
    if smi:
        out = run([smi, "--query-gpu=name,memory.total,memory.used,driver_version,compute_cap,pcie.link.gen.current,"
                        "pcie.link.width.current", "--format=csv,noheader,nounits"])
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5 and not parts[0].startswith("ERR"):
                d["nvidia"].append({"name": parts[0], "vram_mb": parts[1], "vram_used_mb": parts[2],
                                    "driver": parts[3], "compute_cap": parts[4],
                                    "pcie": "/".join(parts[5:7]) if len(parts) >= 7 else ""})
        m = re.search(r"CUDA Version:\s*([\d.]+)", run([smi]))
        d["driver_cuda_version"] = m.group(1) if m else ""
    if sys.platform == "win32":
        out = powershell("Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion,"
                         "CurrentHorizontalResolution,CurrentVerticalResolution,CurrentRefreshRate | ConvertTo-Json")
        try:
            d["video_controllers"] = json.loads(out)
        except Exception:
            d["video_controllers_raw"] = out[:500]
    return d


def module_version(name: str) -> str | None:
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "installed")
    except Exception:
        return None


def ml_info() -> dict:
    d = {"python": sys.version.split()[0], "executable": sys.executable}
    for name in ("numpy", "cv2", "scipy", "onnxruntime", "torch", "tensorrt", "onnx", "psutil", "pynvml",
                 "dxcam", "windows_capture", "mss", "PySide6"):
        d[name] = module_version(name)
    try:
        import onnxruntime as ort
        d["ort_providers"] = ort.get_available_providers()
        d["ort_device"] = ort.get_device()
    except Exception:
        d["ort_providers"] = []
    try:
        import torch
        d["torch_cuda"] = torch.cuda.is_available()
        d["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            d["torch_device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    nvcc = shutil.which("nvcc")
    d["nvcc"] = run([nvcc, "--version"]).strip().splitlines()[-1] if nvcc else None
    d["trtexec"] = shutil.which("trtexec")
    return d


def toolchain_info() -> dict:
    d = {"cmake": shutil.which("cmake"), "gcc": shutil.which("gcc"), "cl": shutil.which("cl"),
         "git": shutil.which("git"), "ffmpeg": shutil.which("ffmpeg")}
    if sys.platform == "win32":
        vswhere = os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe")
        if os.path.exists(vswhere):
            d["visual_studio"] = run([vswhere, "-latest", "-property", "displayName"]).strip()
    return d


def dx_info(enable: bool) -> dict:
    if not enable or sys.platform != "win32":
        return {}
    path = os.path.join(tempfile.gettempdir(), "fctac_dxdiag.txt")
    run(["dxdiag", "/t", path], timeout=90)
    if not os.path.exists(path):
        return {"error": "dxdiag produced no output"}
    txt = open(path, errors="ignore").read()
    out = {}
    for key in ("DirectX Version", "Feature Levels", "Driver Model", "Display Memory", "Dedicated Memory",
                "Hardware Scheduling", "Current Mode"):
        m = re.search(rf"{key}:\s*(.*)", txt)
        if m:
            out[key] = m.group(1).strip()
    return out


def display_info() -> list:
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes

    class DEVMODEW(ctypes.Structure):
        _fields_ = [("dmDeviceName", wintypes.WCHAR * 32), ("dmSpecVersion", wintypes.WORD),
                    ("dmDriverVersion", wintypes.WORD), ("dmSize", wintypes.WORD), ("dmDriverExtra", wintypes.WORD),
                    ("dmFields", wintypes.DWORD), ("dmPositionX", wintypes.LONG), ("dmPositionY", wintypes.LONG),
                    ("dmDisplayOrientation", wintypes.DWORD), ("dmDisplayFixedOutput", wintypes.DWORD),
                    ("dmColor", ctypes.c_short), ("dmDuplex", ctypes.c_short), ("dmYResolution", ctypes.c_short),
                    ("dmTTOption", ctypes.c_short), ("dmCollate", ctypes.c_short), ("dmFormName", wintypes.WCHAR * 32),
                    ("dmLogPixels", wintypes.WORD), ("dmBitsPerPel", wintypes.DWORD), ("dmPelsWidth", wintypes.DWORD),
                    ("dmPelsHeight", wintypes.DWORD), ("dmDisplayFlags", wintypes.DWORD),
                    ("dmDisplayFrequency", wintypes.DWORD)] + [("_pad", ctypes.c_byte * 64)]

    class DISPLAY_DEVICEW(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("DeviceName", wintypes.WCHAR * 32), ("DeviceString", wintypes.WCHAR * 128),
                    ("StateFlags", wintypes.DWORD), ("DeviceID", wintypes.WCHAR * 128), ("DeviceKey", wintypes.WCHAR * 128)]

    out = []
    user32 = ctypes.windll.user32
    i = 0
    while True:
        dd = DISPLAY_DEVICEW()
        dd.cb = ctypes.sizeof(dd)
        if not user32.EnumDisplayDevicesW(None, i, ctypes.byref(dd), 0):
            break
        i += 1
        if not dd.StateFlags & 0x1:     # attached to desktop
            continue
        dm = DEVMODEW()
        dm.dmSize = ctypes.sizeof(DEVMODEW)
        if user32.EnumDisplaySettingsW(dd.DeviceName, -1, ctypes.byref(dm)):
            out.append({"name": dd.DeviceName, "adapter": dd.DeviceString, "width": dm.dmPelsWidth,
                        "height": dm.dmPelsHeight, "hz": dm.dmDisplayFrequency,
                        "primary": bool(dd.StateFlags & 0x4)})
    return out


def disk_info() -> dict:
    u = shutil.disk_usage(os.path.abspath(os.sep))
    return {"total_gb": round(u.total / 2 ** 30, 1), "free_gb": round(u.free / 2 ** 30, 1)}


def recommend(rep: dict) -> dict:
    """Derive the runtime configuration from what was actually found."""
    rec = {"notes": []}
    prov = rep["ml"].get("ort_providers") or []
    gpus = rep["gpu"].get("nvidia") or []
    name = gpus[0]["name"] if gpus else ""
    cc = float(gpus[0]["compute_cap"]) if gpus and re.match(r"^\d+(\.\d+)?$", gpus[0]["compute_cap"]) else 0.0
    if "TensorrtExecutionProvider" in prov:
        rec["provider"] = "TensorrtExecutionProvider"
    elif "CUDAExecutionProvider" in prov:
        rec["provider"] = "CUDAExecutionProvider"
    elif "DmlExecutionProvider" in prov:
        rec["provider"] = "DmlExecutionProvider"
    else:
        rec["provider"] = "CPUExecutionProvider"
        if gpus:
            rec["notes"].append("NVIDIA GPU found but no GPU execution provider: "
                                "pip install onnxruntime-gpu (CUDA) or onnxruntime-directml")
    has_tc = cc >= 7.0 and not any(t in name for t in TURING_NO_TC)
    rec["tensor_cores"] = has_tc
    if cc >= 6.0:
        rec["precision"] = "fp16"
        if not has_tc:
            rec["notes"].append(f"{name or 'GPU'} has no Tensor Cores: FP16 still uses fast half-precision ALUs "
                                "on Turing GTX, but gains are smaller; benchmark FP32 vs FP16 (tools/bench_models.py)")
    else:
        rec["precision"] = "fp32"
    vram = float(gpus[0]["vram_mb"]) if gpus and gpus[0]["vram_mb"].replace(".", "").isdigit() else 0
    rec["detector_input"] = [640, 384] if (gpus and vram >= 4000) else [480, 288]
    rec["notes"].append("FC 27 shares this GPU: keep the detector small and at ~20-30 Hz; "
                        "radar-based state needs no GPU at all")
    ram = rep["ram"].get("total_gb", 0)
    if ram and ram < 12:
        rec["notes"].append("<12 GB RAM: keep replay frame cache small (--cache 24)")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="machine_report.json")
    ap.add_argument("--dxdiag", action="store_true")
    a = ap.parse_args()
    rep = {"os": os_info(), "cpu": cpu_info(), "ram": ram_info(), "gpu": gpu_info(), "ml": ml_info(),
           "toolchain": toolchain_info(), "directx": dx_info(a.dxdiag), "displays": display_info(),
           "disk": disk_info()}
    rep["recommendation"] = recommend(rep)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    g = rep["gpu"]["nvidia"][0] if rep["gpu"]["nvidia"] else {}
    print("MACHINE SUMMARY")
    print(f"  OS      : {rep['os']['platform']}")
    print(f"  CPU     : {rep['cpu'].get('model') or rep['cpu'].get('win32_processor', {}) or rep['cpu']['processor']}"
          f" ({rep['cpu']['logical_cores']} threads)")
    print(f"  RAM     : {rep['ram'].get('total_gb', '?')} GB")
    print(f"  GPU     : {g.get('name', 'none (NVIDIA)')} {g.get('vram_mb', '')} MB, driver {g.get('driver', '-')},"
          f" cc {g.get('compute_cap', '-')}")
    print(f"  Python  : {rep['ml']['python']}  ORT providers: {rep['ml'].get('ort_providers')}")
    print(f"  torch   : {rep['ml'].get('torch')} cuda={rep['ml'].get('torch_cuda')}  TensorRT: {rep['ml'].get('tensorrt')}")
    print(f"  Disk    : {rep['disk']['free_gb']} GB free")
    print(f"  Displays: {rep['displays']}")
    print("RECOMMENDED RUNTIME:", json.dumps(rep["recommendation"], indent=1))
    print(f"full report -> {a.out}")


if __name__ == "__main__":
    main()
