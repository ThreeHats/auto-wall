import os
import sys
import traceback
import time
import datetime
import atexit

from src.utils.update_checker import check_for_updates, fetch_version

# Version information - will be updated by GitHub workflow
APP_VERSION = "1.2.0"
GITHUB_REPO = "ThreeHats/auto-wall"

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Create log directory if it doesn't exist
def setup_logging(debug_mode=False):
    """Set up logging to files with proper cleanup."""
    if debug_mode:
        # In debug mode, keep console output visible and set up debug logger
        print("=" * 50)
        print("AUTO-WALL DEBUG MODE")
        print("=" * 50)
        from src.utils.debug_logger import log_info
        log_info("Starting Auto-Wall in debug mode - console output enabled")
        return None, None
    
    # Normal mode - redirect to log files
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Use fixed filenames instead of timestamps
    stdout_path = os.path.join(log_dir, "stdout.log")
    stderr_path = os.path.join(log_dir, "stderr.log")
    
    # Open log files (overwriting any existing files)
    sys.stdout = open(stdout_path, "w", buffering=1)
    sys.stderr = open(stderr_path, "w", buffering=1)
    
    # Register cleanup function
    atexit.register(cleanup_logs)
    
    return stdout_path, stderr_path

def cleanup_logs():
    """Close log files."""
    # Close the redirected stdout/stderr
    if hasattr(sys.stdout, 'close') and sys.stdout is not sys.__stdout__:
        sys.stdout.close()
    if hasattr(sys.stderr, 'close') and sys.stderr is not sys.__stderr__:
        sys.stderr.close()
    
    # Restore original stdout/stderr
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

def excepthook(exc_type, exc_value, exc_tb):
    """Custom exception handler to log exceptions."""
    tb_text = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    error_msg = f"Uncaught exception: {exc_value}\n{tb_text}"
    print(error_msg)
    
    # Try to show a message box with the error
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox
        # Create application if it doesn't exist
        app = QApplication.instance() 
        if app is None:
            app = QApplication(sys.argv)
        
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Icon.Critical)
        error_box.setWindowTitle("Auto-Wall Error")
        error_box.setText(f"An error occurred: {str(exc_value)}")
        error_box.setDetailedText(tb_text)
        error_box.exec()
    except Exception as e:
        print(f"Failed to display error dialog: {e}")
    
    # Also write to a log file
    try:
        log_path = "auto-wall-error.log"
        with open(log_path, "w") as f:
            f.write(error_msg)
        print(f"Error log written to: {log_path}")
    except Exception as e:
        print(f"Failed to write error log: {e}")

# Install the custom exception hook
sys.excepthook = excepthook

def main():
    """Launch the Auto-Wall application with splash screen."""
    try:
        # Check for debug mode argument
        debug_mode = "--debug" in sys.argv or "-d" in sys.argv
        
        # Setup logging early
        stdout_path, stderr_path = setup_logging(debug_mode)
        
        if debug_mode:
            print(f"Auto-Wall version {APP_VERSION} (Debug Mode)")
            print("Debug output will appear in this console.")
            print("-" * 50)
        else:
            print(f"Auto-Wall version {APP_VERSION}")
            print(f"Logging to {stdout_path} and {stderr_path}")
        
        print("Starting Auto-Wall application...")
        
        # Import PyQt6 for splash screen
        from PyQt6.QtWidgets import QApplication, QSplashScreen, QLabel
        from PyQt6.QtCore import Qt, QTimer, QSize, QThread
        from PyQt6.QtGui import QPixmap, QFont, QColor, QPainter
        
        # Create application instance
        app = QApplication(sys.argv)
        
        # Create the splash screen
        splash_path = os.path.join(os.path.dirname(__file__), "resources", "splash.png")
        
        # If splash image doesn't exist, create a simple one
        if not os.path.exists(splash_path):
            splash_pixmap = QPixmap(600, 400)
            splash_pixmap.fill(QColor(40, 40, 45))
            painter = QPainter(splash_pixmap)
            painter.setPen(QColor(255, 255, 255))
            font = QFont("Arial", 24)
            painter.setFont(font)
            painter.drawText(splash_pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "Loading Auto-Wall...")
            painter.end()
        else:
            splash_pixmap = QPixmap(splash_path)
        
        splash = QSplashScreen(splash_pixmap, Qt.WindowType.WindowStaysOnTopHint)
        splash.show()
        
        # Process events to make sure splash is displayed
        app.processEvents()
        
        # Function to update splash message
        def update_splash(message):
            splash.showMessage(f"  {message}", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft, Qt.GlobalColor.white)
            app.processEvents()
        
        # Load modules with splash screen updates
        update_splash("Loading NumPy...")
        import numpy
        print(f"NumPy version: {numpy.__version__}")
        
        update_splash("Loading OpenCV...")
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        
        update_splash("Loading scikit-learn...")
        import sklearn
        print(f"scikit-learn version: {sklearn.__version__}")
        
        update_splash("Initializing UI...")
        
        # Import the application
        print("Importing WallDetectionApp...")
        from src.gui.app import WallDetectionApp
        
        update_splash("Creating application window...")
        
        # Function to show the main window and close splash
        def show_window():
            try:
                version_string = f"{APP_VERSION}-debug" if debug_mode else APP_VERSION
                window = WallDetectionApp(version=version_string, github_repo=GITHUB_REPO)
                window.show()
                splash.finish(window)
                print("Window displayed successfully")
                
                if debug_mode:
                    from src.utils.debug_logger import log_info
                    log_info("Auto-Wall window displayed successfully in debug mode")
                
                # Start update check after window is shown (skip in debug mode)
                if not debug_mode:
                    QTimer.singleShot(2000, lambda: check_for_updates(window))
            except Exception as e:
                print(f"Error creating or showing window: {e}")
                traceback.print_exc()
                splash.close()
                raise
        
        # Start the window after a short delay
        QTimer.singleShot(800, show_window)  # Add a slight delay for better UX
        
        print("Starting application event loop...")
        return app.exec()
    
    except Exception as e:
        print(f"ERROR: Application failed to start: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    # Check for help argument
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Auto-Wall - Battle Map Wall Detection Tool")
        print(f"Version: {APP_VERSION}")
        print("")
        print("Usage:")
        print("  python auto_wall.py          # Normal mode (output to log files)")
        print("  python auto_wall.py --debug  # Debug mode (console output visible)")
        print("  python auto_wall.py -d       # Debug mode (short form)")
        print("  python auto_wall.py --help   # Show this help message")
        sys.exit(0)
    
    sys.exit(main())
