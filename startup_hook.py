
# Startup diagnostics hook
import os
import sys
import traceback

# Redirect stdout and stderr to files in case console is hidden
log_dir = os.path.dirname(sys.executable)
sys.stdout = open(os.path.join(log_dir, "stdout.log"), "w")
sys.stderr = open(os.path.join(log_dir, "stderr.log"), "w")

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"sys.path: {sys.path}")
print(f"Working directory: {os.getcwd()}")
print(f"_MEIPASS: {getattr(sys, '_MEIPASS', 'Not in PyInstaller bundle')}")

# Set up global exception handler to log uncaught exceptions
def global_exception_handler(exctype, value, tb):
    error_msg = ''.join(traceback.format_exception(exctype, value, tb))
    print(f"UNCAUGHT EXCEPTION: {error_msg}")
    with open(os.path.join(log_dir, "uncaught_exception.log"), "w") as f:
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
