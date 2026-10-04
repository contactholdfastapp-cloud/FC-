# Dependencies and why each exists

| Package | Where | Why | Alternative considered |
|---|---|---|---|
| numpy | runtime | all numeric work | — |
| opencv-python (<5) | runtime | video I/O, drawing, homography, colour segmentation, replay UI (highgui) | PySide6 (much larger; not needed for a minimal UI) |
| scipy | runtime | `linear_sum_assignment` (tracking, role assignment), `least_squares` (calibration) | hand-written Hungarian (slower, more code) |
| onnxruntime[-gpu/-directml] | runtime, optional | learned vision model inference (CUDA / DirectML / CPU) | TensorRT direct (benchmarked where installed) |
| torch | training only | training detector / ranker / predictor; exported to ONNX | — |
| windows-capture | live, Windows, optional | Windows Graphics Capture of the FC 27 window (excludes our own overlay) | dxcam (desktop duplication; captures overlay too), mss (GDI, slow) |
| dxcam | live, Windows, optional | DXGI Desktop Duplication fallback | — |
| pytest | dev | tests | — |

Not used on purpose: any cloud/LLM API, any process-memory reader, any input
automation library, any driver.
