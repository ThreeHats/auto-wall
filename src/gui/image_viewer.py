from PyQt6.QtWidgets import (QLabel)
from PyQt6.QtCore import Qt, QPoint, QPointF
from PyQt6.QtGui import QWheelEvent, QTransform, QPainter, QPixmap
import cv2

from src.utils.geometry import point_to_line_distance, convert_to_image_coordinates

class InteractiveImageLabel(QLabel):
    """Custom QLabel that handles mouse events for contour/line deletion and mask editing, with zoom and pan support."""
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
        
        # Zoom and pan state
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.pan_offset = QPointF(0, 0)
        self.base_pixmap = None
          # Pan state
        self.panning = False
        self.pan_start_pos = None
        self.pan_start_offset = None
    def set_base_pixmap(self, pixmap, preserve_view=False):
        """Set the base pixmap for zoom and pan operations."""
        self.base_pixmap = pixmap
        if not preserve_view:
            # Fit the image to the window and center it when setting a new pixmap
            self.fit_to_window()
        else:
            self.update_display()
        
    def update_display(self):
        """Update the display with current zoom and pan."""
        if self.base_pixmap is None:
            return
        
        # Get the widget's current size
        widget_size = self.size()
        if widget_size.width() <= 0 or widget_size.height() <= 0:
            return
              # Create a pixmap that fits the widget size
        display_pixmap = QPixmap(widget_size)
        # Fill with the application's background color instead of black
        from PyQt6.QtGui import QColor
        app_bg_color = QColor(30, 30, 30)  # #1e1e1e from the stylesheet
        display_pixmap.fill(app_bg_color)
        
        # Calculate the scaled image size
        scaled_size = self.base_pixmap.size() * self.zoom_factor
        
        # Create the scaled image
        scaled_pixmap = self.base_pixmap.scaled(
            scaled_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Calculate the position to draw the scaled image (considering pan offset)
        draw_x = int(self.pan_offset.x())
        draw_y = int(self.pan_offset.y())
        
        # Draw the scaled image onto the display pixmap
        painter = QPainter(display_pixmap)
        painter.drawPixmap(draw_x, draw_y, scaled_pixmap)
        painter.end()
        
        # Set the clipped pixmap (this won't change the widget size)
        self.setPixmap(display_pixmap)
        
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        if self.parent_app and self.base_pixmap:
            # Get the mouse position relative to the widget
            mouse_pos = event.position()
            
            # Calculate zoom factor change
            zoom_in = event.angleDelta().y() > 0
            zoom_factor_change = 1.15 if zoom_in else 1.0 / 1.15  # Slightly more gradual
            
            # Calculate new zoom factor
            old_zoom = self.zoom_factor
            new_zoom = old_zoom * zoom_factor_change
            new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
            
            if new_zoom != old_zoom:
                # Calculate the zoom ratio
                zoom_ratio = new_zoom / old_zoom
                
                # Adjust pan offset to keep the point under the mouse cursor stationary
                # Formula: new_pan = mouse_pos - (mouse_pos - old_pan) * zoom_ratio
                old_pan = self.pan_offset
                new_pan_x = mouse_pos.x() - (mouse_pos.x() - old_pan.x()) * zoom_ratio
                new_pan_y = mouse_pos.y() - (mouse_pos.y() - old_pan.y()) * zoom_ratio
                  # Update zoom factor and pan offset
                self.zoom_factor = new_zoom
                self.pan_offset = QPointF(new_pan_x, new_pan_y)
                
                self.update_display()
        super().wheelEvent(event)
        
    def display_to_image_coords(self, display_point):
        """Convert display coordinates to image coordinates accounting for zoom and pan."""
        if self.base_pixmap is None or self.parent_app.current_image is None:
            return None
            
        # Get the current pixmap dimensions and original image dimensions
        pixmap_size = self.base_pixmap.size()
        img_height, img_width = self.parent_app.current_image.shape[:2]
        
        # Account for the actual scaling between pixmap and original image
        pixmap_to_image_scale_x = img_width / pixmap_size.width()
        pixmap_to_image_scale_y = img_height / pixmap_size.height()
        
        # Convert from display coordinates to pixmap coordinates
        # Account for pan offset and zoom factor
        pixmap_x = (display_point.x() - self.pan_offset.x()) / self.zoom_factor
        pixmap_y = (display_point.y() - self.pan_offset.y()) / self.zoom_factor
        
        # Convert from pixmap coordinates to original image coordinates
        image_x = int(pixmap_x * pixmap_to_image_scale_x)
        image_y = int(pixmap_y * pixmap_to_image_scale_y)
          # Check bounds
        if (image_x < 0 or image_x >= img_width or 
            image_y < 0 or image_y >= img_height):
            return None
            
        return (image_x, image_y)
        
    def reset_view(self):
        """Reset zoom and pan to default values."""
        self.zoom_factor = 1.0
        self.center_image()
        self.update_display()
        
    def fit_to_window(self):
        """Fit the image to the window size."""
        if self.base_pixmap is None:
            return
            
        widget_size = self.size()
        pixmap_size = self.base_pixmap.size()
        
        # Calculate scale to fit
        scale_x = widget_size.width() / pixmap_size.width()
        scale_y = widget_size.height() / pixmap_size.height()
        scale = min(scale_x, scale_y)
        
        self.zoom_factor = scale
        
        # Center the image after scaling
        self.center_image()
        self.update_display()
        
    def mousePressEvent(self, event):
        """Handle mouse click events."""
        if self.parent_app:
            pos = event.position()
            x, y = int(pos.x()), int(pos.y())
            
            # Handle right button clicks for panning
            if event.button() == Qt.MouseButton.RightButton:
                self.panning = True
                self.pan_start_pos = pos
                self.pan_start_offset = QPointF(self.pan_offset)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                return
            
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
        """Handle mouse move events for hover highlighting, drag selection, drawing, and panning."""
        if self.parent_app:
            pos = event.position()
            x, y = int(pos.x()), int(pos.y())
            
            # Handle panning with right mouse button
            if self.panning and event.buttons() & Qt.MouseButton.RightButton:
                delta = pos - self.pan_start_pos
                self.pan_offset = self.pan_start_offset + delta
                self.update_display()
                return
            
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
                    self.handle_hover(pos.x(), pos.y())
                elif self.parent_app.edit_mask_mode_enabled:
                    # Always update brush preview when hovering in edit mask mode
                    self.parent_app.drawing_tools.update_brush_preview(x, y)                
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events for completing drag selection, drawing, or panning."""
        if self.parent_app:
            # Handle panning end
            if event.button() == Qt.MouseButton.RightButton and self.panning:
                self.panning = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
                return
                
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
                self.clear_hover()
            elif self.parent_app.edit_mask_mode_enabled:
                # Clear brush preview when mouse leaves the widget
                self.parent_app.drawing_tools.clear_brush_preview()
        super().leaveEvent(event)

    def handle_hover(self, x, y):
        """Handle mouse hover events for highlighting contours."""
        if not self.parent_app.current_contours or self.parent_app.current_image is None:
            return
            
        # Convert display coordinates to image coordinates
        img_x, img_y = convert_to_image_coordinates(self.parent_app, x, y)
        
        # Check if coordinates are valid
        if img_x is None or img_y is None:
            self.clear_hover()
            return
            
        # Find the contour under the cursor - only check edges
        found_index = -1
        min_distance = float('inf')
        
        # Check if cursor is on a contour edge
        for i, contour in enumerate(self.parent_app.current_contours):
            contour_points = contour.reshape(-1, 2)
            
            for j in range(len(contour_points)):
                p1 = contour_points[j]
                p2 = contour_points[(j + 1) % len(contour_points)]
                distance = point_to_line_distance(self.parent_app, img_x, img_y, p1[0], p1[1], p2[0], p2[1])
                
                # If point is close enough to a line segment and closer than any previous match
                if distance < 5 and distance < min_distance:  # Threshold for line detection (pixels)
                    min_distance = distance
                    found_index = i
        
        # Update highlight if needed
        if found_index != self.parent_app.highlighted_contour_index:
            self.parent_app.highlighted_contour_index = found_index
            self.update_highlight()

    def clear_hover(self):
        """Clear any contour highlighting."""
        if self.parent_app.highlighted_contour_index != -1:
            self.parent_app.highlighted_contour_index = -1
            self.update_highlight()

    def update_highlight(self):
        """Update the display with highlighted contour."""
        if self.parent_app.original_processed_image is None:
            return
            
        # Start with the original image (without highlights)
        self.parent_app.processed_image = self.parent_app.original_processed_image.copy()
        
        # If a contour is highlighted, draw it with a different color/thickness
        if self.parent_app.highlighted_contour_index != -1 and self.parent_app.highlighted_contour_index < len(self.parent_app.current_contours):
            # Use different colors based on the current mode
            if self.parent_app.deletion_mode_enabled:
                highlight_color = (0, 0, 255)  # Red for delete
            elif self.parent_app.thin_mode_enabled:
                highlight_color = (255, 0, 255)  # Magenta for thin
            else:
                highlight_color = (0, 0, 255)  # Default: red
                
            highlight_thickness = 3
            cv2.drawContours(
                self.parent_app.processed_image, 
                [self.parent_app.current_contours[self.parent_app.highlighted_contour_index]], 
                0, highlight_color, highlight_thickness
            )
            
        # Update the display while preserving the current zoom and pan
        self.parent_app.image_processor.display_image(self.parent_app.processed_image, preserve_view=True)

    def center_image(self):
        """Center the image in the widget."""
        if self.base_pixmap is None:
            return
            
        widget_size = self.size()
        pixmap_size = self.base_pixmap.size()
        
        # Calculate the position to center the image
        center_x = (widget_size.width() - pixmap_size.width() * self.zoom_factor) / 2
        center_y = (widget_size.height() - pixmap_size.height() * self.zoom_factor) / 2
        
        self.pan_offset = QPointF(center_x, center_y)