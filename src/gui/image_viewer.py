from PyQt6.QtWidgets import (QLabel)
from PyQt6.QtCore import Qt, QPoint

class InteractiveImageLabel(QLabel):
    """Custom QLabel that handles mouse events for contour/line deletion and mask editing."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        # Set mouse tracking to enable hover effects
        self.setMouseTracking(True)
        # Enable mouse interaction
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # For drag selection
        self.selection_start = None
        self.selection_current = None
        # For drawing
        self.last_point = None
        
    def mousePressEvent(self, event):
        """Handle mouse click events."""
        if self.parent_app:
            pos = event.position()
            x, y = int(pos.x()), int(pos.y())
            self.selection_start = QPoint(x, y)
            self.selection_current = self.selection_start
            
            # Handle left button clicks
            if event.button() == Qt.MouseButton.LeftButton:
                if self.parent_app.deletion_mode_enabled or self.parent_app.color_selection_mode_enabled:
                    self.parent_app.selection_manager.start_selection(x, y)
                elif self.parent_app.edit_mask_mode_enabled:
                    # Start drawing on mask
                    self.last_point = QPoint(x, y)
                    self.parent_app.drawing_tools.start_drawing(x, y)
                elif self.parent_app.thin_mode_enabled:
                    self.parent_app.selection_manager.start_selection(x, y)
                
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for hover highlighting, drag selection, and drawing."""
        if self.parent_app:
            pos = event.position()
            x, y = int(pos.x()), int(pos.y())
            
            # If dragging with left button
            if self.selection_start and event.buttons() & Qt.MouseButton.LeftButton:
                self.selection_current = QPoint(x, y)
                if self.parent_app.deletion_mode_enabled or self.parent_app.color_selection_mode_enabled:
                    self.parent_app.selection_manager.update_selection(x, y)
                elif self.parent_app.edit_mask_mode_enabled and self.last_point:
                    # Continue drawing on mask
                    current_point = QPoint(x, y)
                    self.parent_app.drawing_tools.continue_drawing(self.last_point.x(), self.last_point.y(), x, y)
                    self.last_point = current_point
                elif self.parent_app.thin_mode_enabled:
                    self.parent_app.selection_manager.update_selection(x, y)
            # Just hovering - this always runs for any mouse movement
            else:
                if self.parent_app.deletion_mode_enabled or self.parent_app.thin_mode_enabled:
                    self.parent_app.handle_hover(pos.x(), pos.y())
                elif self.parent_app.edit_mask_mode_enabled:
                    # Always update brush preview when hovering in edit mask mode
                    self.parent_app.drawing_tools.update_brush_preview(x, y)
                
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events for completing drag selection or drawing."""
        if self.parent_app:
            if self.selection_start and event.button() == Qt.MouseButton.LeftButton:
                pos = event.position()
                x, y = int(pos.x()), int(pos.y())
                self.selection_current = QPoint(x, y)
                
                if self.parent_app.deletion_mode_enabled or self.parent_app.color_selection_mode_enabled:
                    self.parent_app.selection_manager.end_selection(x, y)
                elif self.parent_app.edit_mask_mode_enabled:
                    # End drawing on mask
                    self.parent_app.drawing_tools.end_drawing()
                    self.last_point = None
                elif self.parent_app.thin_mode_enabled:
                    self.parent_app.selection_manager.end_selection(x, y)
                
                # Clear selection points
                self.selection_start = None
                self.selection_current = None
                
        super().mouseReleaseEvent(event)
    
    def leaveEvent(self, event):
        """Handle mouse leaving the widget."""
        if self.parent_app:
            if self.parent_app.deletion_mode_enabled or self.parent_app.thin_mode_enabled:
                self.parent_app.clear_hover()
            elif self.parent_app.edit_mask_mode_enabled:
                # Clear brush preview when mouse leaves the widget
                self.parent_app.drawing_tools.clear_brush_preview()
        super().leaveEvent(event)

