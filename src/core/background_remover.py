import ctypes
import glob
import os
import sys

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal, Qt


def _preload_nvidia_libs():
    """Preload pip-installed NVIDIA .so files so onnxruntime can find CUDA providers."""
    search_paths = list(sys.path)
    # In a frozen PyInstaller bundle sys.path points into the bundle directory,
    # not site-packages. Prepend _MEIPASS so we find the nvidia libs we bundled
    # at build time before falling through to the regular site-packages search.
    if hasattr(sys, '_MEIPASS'):
        search_paths.insert(0, sys._MEIPASS)
    for sp in search_paths:
        nvidia_base = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia_base):
            continue
        for lib_dir in sorted(glob.glob(os.path.join(nvidia_base, "*/lib"))):
            for so_file in sorted(glob.glob(os.path.join(lib_dir, "*.so.*"))):
                try:
                    ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
        break


_preload_nvidia_libs()

# Ensure CUDA device ordering matches nvidia-smi (PCI bus order).
# Without this, CUDA may assign device indices differently from nvidia-smi,
# causing the wrong GPU to be selected in the UI.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


# Available operating resolutions (max dimension passed to rembg)
REMBG_RESOLUTIONS = [
    (512, "512x512 (Fast)"),
    (1024, "1024x1024"),
    (1536, "1536x1536"),
    (2048, "2048x2048"),
    (2304, "2304x2304"),
    (0, "Full (no limit)"),
]

DEFAULT_RESOLUTION = 1024

# Available rembg models
REMBG_MODELS = [
    ("isnet-general-use", "ISNet General Use"),
    ("isnet-anime", "ISNet Anime"),
    ("u2net", "U2Net General Purpose"),
    ("u2netp", "U2Net-P Lightweight"),
    ("u2net_human_seg", "U2Net Human Seg"),
    ("silueta", "Silueta Lightweight"),
    ("birefnet-general", "BiRefNet General Use"),
    ("birefnet-general-lite", "BiRefNet General Lite"),
    ("birefnet-massive", "BiRefNet Massive"),
    ("birefnet-hrsod", "BiRefNet HRSOD"),
    ("birefnet-dis", "BiRefNet DIS"),
    ("birefnet-cod", "BiRefNet COD"),
    ("birefnet-portrait", "BiRefNet Portrait"),
    ("bria-rmbg", "BRIA RMBG 2.0"),
]

DEFAULT_MODEL = "bria-rmbg"


def get_available_devices():
    """Return a list of (device_id, display_name) tuples for the device selector.

    Always includes CPU.  Adds each detected GPU with its name and VRAM.
    """
    devices = [("cpu", "CPU")]
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    idx, name, vram = parts[0], parts[1], parts[2]
                    devices.append((f"gpu:{idx}", f"{name} ({vram} MB)"))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: if no nvidia-smi but onnxruntime reports CUDA, add a generic GPU entry
    if len(devices) == 1:
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                devices.append(("gpu:0", "GPU 0"))
            elif "ROCMExecutionProvider" in available:
                devices.append(("gpu:0", "GPU 0 (ROCm)"))
            elif "DmlExecutionProvider" in available:
                devices.append(("gpu:0", "GPU 0 (DirectML)"))
            elif "CoreMLExecutionProvider" in available:
                devices.append(("gpu:0", "GPU 0 (CoreML)"))
        except ImportError:
            pass

    return devices


def _get_ort_providers_for_device(device_id):
    """Convert a device selector value to onnxruntime provider list.

    Args:
        device_id: "cpu" or "gpu:N"

    Returns:
        List of provider tuples/strings for ort.InferenceSession.
    """
    if device_id == "cpu":
        return ["CPUExecutionProvider"]

    gpu_idx = int(device_id.split(":")[1]) if ":" in device_id else 0

    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        return ["CPUExecutionProvider"]

    # Try GPU providers in priority order
    for prov in ["CUDAExecutionProvider", "ROCMExecutionProvider",
                 "DmlExecutionProvider", "CoreMLExecutionProvider"]:
        if prov not in available:
            continue
        opts = {"device_id": str(gpu_idx)}
        return [(prov, opts), "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


def _create_session(model_name, providers):
    """Create a rembg session with explicit onnxruntime provider options."""
    import onnxruntime as ort
    from rembg.sessions import sessions_class

    session_class = None
    for sc in sessions_class:
        if sc.name() == model_name:
            session_class = sc
            break
    if session_class is None:
        raise ValueError(f"No session class found for model '{model_name}'")

    sess_opts = ort.SessionOptions()
    model_path = str(session_class.download_models())

    # Try loading with full optimization first; if the GPU arena fails during
    # optimization, retry with optimization disabled (avoids arena fragmentation
    # on GPUs with limited VRAM during graph rewriting).
    try:
        inner_session = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=providers
        )
    except Exception:
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        inner_session = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=providers
        )

    session = object.__new__(session_class)
    session.model_name = model_name
    session.inner_session = inner_session
    return session


class BackgroundRemovalWorker(QObject):
    """Worker that runs background removal in a separate thread."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, image, model_name, session_cache, max_dim=1024, device="cpu"):
        super().__init__()
        self.image = image
        self.model_name = model_name
        self.session_cache = session_cache
        self.max_dim = max_dim
        self.device = device
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            import rembg
            from PIL import Image

            if self._cancelled:
                return

            device_label = self.device.upper() if self.device == "cpu" else self.device
            self.status.emit(f"Loading model ({device_label})...")

            # Cache key includes device so switching device invalidates
            cache_key = (self.model_name, self.device)

            if cache_key in self.session_cache:
                session = self.session_cache[cache_key]
            else:
                # Delete old sessions and force GC to release GPU memory
                # before creating a new one
                self.session_cache.clear()
                import gc
                gc.collect()

                providers = _get_ort_providers_for_device(self.device)
                session = _create_session(self.model_name, providers)
                self.session_cache[cache_key] = session

            if self._cancelled:
                return

            orig_h, orig_w = self.image.shape[:2]

            # Downscale for inference if image exceeds selected operating resolution
            img_max_dim = max(orig_h, orig_w)
            if self.max_dim > 0 and img_max_dim > self.max_dim:
                scale = self.max_dim / img_max_dim
                small = cv2.resize(self.image, (int(orig_w * scale), int(orig_h * scale)),
                                   interpolation=cv2.INTER_AREA)
                self.status.emit(f"Removing background ({small.shape[1]}x{small.shape[0]})...")
            else:
                small = self.image
                self.status.emit("Removing background...")

            # Convert BGR/BGRA (OpenCV) to RGB/RGBA (PIL)
            n_channels = small.shape[2] if small.ndim == 3 else 1
            if n_channels == 4:
                alpha = small[:, :, 3]
                rgb = small[:, :, :3][:, :, ::-1]
                rgb_image = np.concatenate([rgb, alpha[:, :, np.newaxis]], axis=2)
                pil_image = Image.fromarray(rgb_image, mode='RGBA')
            elif n_channels == 3:
                rgb_image = small[:, :, ::-1]
                pil_image = Image.fromarray(rgb_image)
            else:
                raise ValueError(f"Unexpected channel count {n_channels} in image")

            # Run inference — fall back to CPU on GPU failure (e.g. OOM)
            try:
                result_rgba = rembg.remove(pil_image, session=session)
            except Exception as gpu_err:
                if self.device == "cpu":
                    raise
                self.status.emit("GPU failed, retrying on CPU...")
                print(f"GPU inference failed ({gpu_err}), falling back to CPU")
                cpu_providers = ["CPUExecutionProvider"]
                cpu_session = _create_session(self.model_name, cpu_providers)
                result_rgba = rembg.remove(pil_image, session=cpu_session)

            if self._cancelled:
                return

            self.status.emit("Compositing result...")

            # Extract alpha, upscale if we downscaled, composite onto white
            result_rgba = np.array(result_rgba)
            alpha = result_rgba[:, :, 3]

            if alpha.shape[0] != orig_h or alpha.shape[1] != orig_w:
                alpha = cv2.resize(alpha, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            alpha_f = alpha[:, :, np.newaxis].astype(np.float32) / 255.0
            rgb = self.image[:, :, :3].astype(np.float32)
            white_bg = np.full_like(rgb, 255.0)
            composited = (rgb * alpha_f + white_bg * (1.0 - alpha_f)).astype(np.uint8)

            self.finished.emit(composited)

        except Exception as e:
            self.error.emit(str(e))


class BackgroundRemover:
    """Manager class for background removal operations."""

    def __init__(self, app):
        self.app = app
        self._session_cache = {}
        self._worker = None
        self._thread = None

    def start_removal(self, image, model_name, max_dim=1024, device="cpu"):
        """Start background removal in a worker thread."""
        self.cancel()

        self._thread = QThread()
        self._worker = BackgroundRemovalWorker(
            image, model_name, self._session_cache, max_dim, device
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.status.connect(self._on_status)

        self._worker.finished.connect(lambda *_: self._thread.quit())
        self._worker.error.connect(lambda *_: self._thread.quit())
        self._thread.finished.connect(self._cleanup_thread)

        self._thread.start()

    def _cleanup_thread(self):
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

    def cancel(self):
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(5000)
        self._worker = None
        self._thread = None

    def _on_finished(self, result):
        self.app.bg_removed_image = result
        if hasattr(self.app, 'bg_removal_panel'):
            self.app.bg_removal_panel.on_removal_finished()
        if hasattr(self.app, 'bg_removal_checkbox') and self.app.bg_removal_checkbox.isChecked():
            self.app.image_processor.update_image()

    def _on_error(self, error_msg):
        print(f"Background removal error: {error_msg}")
        if hasattr(self.app, 'bg_removal_panel'):
            self.app.bg_removal_panel.on_removal_error(error_msg)

    def _on_status(self, msg):
        if hasattr(self.app, 'bg_removal_panel'):
            self.app.bg_removal_panel.update_status(msg)
