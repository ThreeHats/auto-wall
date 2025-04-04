import os
import sys
import traceback

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

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
    """Launch the Auto-Wall application."""
    try:
        print("Starting Auto-Wall application...")
        
        # Try importing key modules to verify they're loaded correctly
        print("Importing NumPy...")
        import numpy
        print(f"NumPy version: {numpy.__version__}")
        
        print("Importing OpenCV...")
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        
        print("Importing scikit-learn...")
        import sklearn
        print(f"scikit-learn version: {sklearn.__version__}")
        
        print("Importing PyQt6...")
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QTimer, QSize
        
        # Import the application
        print("Importing WallDetectionApp...")
        from src.gui.app import WallDetectionApp
        
        # Initialize the application
        print("Creating QApplication...")
        app = QApplication(sys.argv)
        
        print("Creating main window...")
        def show_window():
            try:
                window = WallDetectionApp()
                window.show()
                print("Window displayed successfully")
            except Exception as e:
                print(f"Error creating or showing window: {e}")
                traceback.print_exc()
                raise
                
        # Start the window after a short delay
        QTimer.singleShot(100, show_window)
        
        print("Starting application event loop...")
        return app.exec()
    
    except Exception as e:
        print(f"ERROR: Application failed to start: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
