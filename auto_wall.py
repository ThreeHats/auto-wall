import os
import sys
import traceback

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the main application
from PyQt6.QtWidgets import QApplication

def main():
    """Launch the Auto-Wall application."""
    try:
        # Only import the WallDetectionApp here inside the try block
        # This will help us catch import errors more clearly
        from src.gui.app import WallDetectionApp
        
        app = QApplication(sys.argv)
        window = WallDetectionApp()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"ERROR: Application failed to start: {e}")
        traceback.print_exc()
        print("\nPress Enter to exit...")
        input()  # This will keep the console open until user presses Enter

if __name__ == "__main__":
    main()
