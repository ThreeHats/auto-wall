
# Startup diagnostics hook
import os
import sys
import traceback
import atexit
import datetime

# Create logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(sys.executable), "logs")
os.makedirs(log_dir, exist_ok=True)

# Create timestamped log files
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
stdout_path = os.path.join(log_dir, f"stdout_{timestamp}.log")
stderr_path = os.path.join(log_dir, f"stderr_{timestamp}.log")

# Redirect stdout and stderr to files
sys.stdout = open(stdout_path, "w", buffering=1)
sys.stderr = open(stderr_path, "w", buffering=1)

# Register cleanup function to close log files on exit
def cleanup_logs():
    if hasattr(sys.stdout, 'close') and sys.stdout is not sys.__stdout__:
        sys.stdout.close()
    if hasattr(sys.stderr, 'close') and sys.stderr is not sys.__stderr__:
        sys.stderr.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

atexit.register(cleanup_logs)

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"sys.path: {sys.path}")
print(f"Working directory: {os.getcwd()}")
print(f"_MEIPASS: {getattr(sys, '_MEIPASS', 'Not in PyInstaller bundle')}")

# Set up global exception handler to log uncaught exceptions
def global_exception_handler(exctype, value, tb):
    error_msg = ''.join(traceback.format_exception(exctype, value, tb))
    print(f"UNCAUGHT EXCEPTION: {error_msg}")
    with open(os.path.join(log_dir, f"uncaught_exception_{timestamp}.log"), "w") as f:
        f.write(error_msg)
    sys.__excepthook__(exctype, value, tb)

sys.excepthook = global_exception_handler

# Try importing key packages to verify they're available
try:
    import numpy
    print(f"NumPy version: {numpy.__version__}")
except Exception as e:
    print(f"Error importing NumPy: {e}")

try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except Exception as e:
    print(f"Error importing OpenCV: {e}")

try:
    import sklearn
    print(f"scikit-learn version: {sklearn.__version__}")
except Exception as e:
    print(f"Error importing scikit-learn: {e}")

try:
    from PyQt6.QtWidgets import QApplication
    print("PyQt6 imported successfully")
except Exception as e:
    print(f"Error importing PyQt6: {e}")
