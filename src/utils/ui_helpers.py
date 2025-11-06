import os

def apply_stylesheet(self):
    """Apply the application stylesheet from the CSS file."""
    try:
        # Get the path to the stylesheet
        import sys
        if getattr(sys, 'frozen', False):
            # Running as PyInstaller bundle
            if sys.platform == "darwin":
                # macOS app bundle
                style_path = os.path.join(os.path.dirname(sys.executable), '..', 'Resources', 'src', 'styles', 'style.qss')
            else:
                # Other platforms
                style_path = os.path.join(os.path.dirname(sys.executable), 'src', 'styles', 'style.qss')
        else:
            # Running as script
            style_path = os.path.join(os.path.dirname(__file__), '..', 'styles', 'style.qss')
        
        # Check if the file exists
        if not os.path.exists(style_path):
            print(f"Warning: Stylesheet not found at {style_path}")
            return
            
        # Read and apply the stylesheet
        with open(style_path, 'r') as f:
            stylesheet = f.read()
            self.setStyleSheet(stylesheet)
            print(f"Applied stylesheet from {style_path}")
    except Exception as e:
        print(f"Error applying stylesheet: {e}")

def resizeEvent(self, event):
    """Handle window resize events to update the image display."""
    super().resizeEvent(event)
    
    # If we have a current image displayed, update it to fit the new window size
    if hasattr(self, 'processed_image') and self.processed_image is not None:
        self.refresh_display()
        
    # If we're in UVTT preview mode, redraw the preview
    if hasattr(self, 'uvtt_preview_active') and self.uvtt_preview_active and self.uvtt_walls_preview:
        self.export_panel.display_uvtt_preview()
    
    # Update the position of the update notification
    if hasattr(self, 'update_notification'):
        self.update_notification.setGeometry(
            self.width() - 250, 10, 240, 40
        )

def keyPressEvent(self, event):
    """Handle key press events."""
    # Add debugging for Ctrl+Z
    if event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
        print("Ctrl+Z detected via keyPressEvent")
        # Use unified undo
        self.unified_undo()
    else:
        super().keyPressEvent(event)
