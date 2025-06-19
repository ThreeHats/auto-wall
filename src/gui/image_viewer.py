from PyQt6.QtWidgets import (QLabel)
from PyQt6.QtCore import Qt, QPoint, QPointF
from PyQt6.QtGui import QWheelEvent, QTransform, QPainter, QPixmap, QImage, QCursor
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
        
        # For region-based updates
        self.last_updated_region = None
    
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
        
        # Store this clean pixmap as our original for overlays
        # This ensures we have a clean base to draw overlays on top of
        self.original_display_pixmap = display_pixmap.copy()
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
        if self.base_pixmap is None:
            return None
            
        # Get the current pixmap dimensions - this is the display image (full resolution)
        pixmap_size = self.base_pixmap.size()
        
        # Convert from display coordinates to pixmap coordinates
        # Account for pan offset and zoom factor
        pixmap_x = (display_point.x() - self.pan_offset.x()) / self.zoom_factor
        pixmap_y = (display_point.y() - self.pan_offset.y()) / self.zoom_factor
        
        # The pixmap coordinates are already in the display image coordinate space
        image_x = int(pixmap_x)
        image_y = int(pixmap_y)
        
        # Check bounds against the pixmap (display image) dimensions
        if (image_x < 0 or image_x >= pixmap_size.width() or 
            image_y < 0 or image_y >= pixmap_size.height()):
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
    
    def enterEvent(self, event):
        """Handle mouse entering the widget."""
        if self.parent_app and self.parent_app.edit_mask_mode_enabled:
            # Show brush preview when mouse enters the widget
            cursor_pos = self.mapFromGlobal(QCursor.pos())
            if self.rect().contains(cursor_pos):
                self.parent_app.drawing_tools.update_brush_preview(cursor_pos.x(), cursor_pos.y(), force=True)
                # Start the idle timer to ensure the preview stays visible
                if hasattr(self.parent_app.drawing_tools, 'mouse_idle_timer'):
                    self.parent_app.drawing_tools.mouse_idle_timer.start(
                        self.parent_app.drawing_tools.idle_detection_interval
                    )
        super().enterEvent(event)

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
            return        # Convert to working coordinates for contour matching if needed
        # img_x, img_y are in display image coordinates (full resolution)
        # but contours are in working resolution, so scale down if necessary
        working_x, working_y = img_x, img_y
        if self.parent_app.scale_factor != 1.0 and self.parent_app.original_image is not None:
            working_x = int(img_x * self.parent_app.scale_factor)
            working_y = int(img_y * self.parent_app.scale_factor)
        
        # Find the contour under the cursor - only check edges
        found_index = -1
        min_distance = float('inf')
        
        # Check if cursor is on a contour edge
        for i, contour in enumerate(self.parent_app.current_contours):
            contour_points = contour.reshape(-1, 2)
            
            for j in range(len(contour_points)):
                p1 = contour_points[j]
                p2 = contour_points[(j + 1) % len(contour_points)]
                
                # Calculate distance
                distance = self.calculate_point_to_line_distance(working_x, working_y, p1[0], p1[1], p2[0], p2[1])
                
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
            
            # Get the highlighted contour
            highlighted_contour = self.parent_app.current_contours[self.parent_app.highlighted_contour_index]
            
            # Scale the contour to match the display image if needed
            if self.parent_app.scale_factor != 1.0 and self.parent_app.original_image is not None:
                # Scale contour to original resolution for display
                scaled_contour = self.parent_app.contour_processor.scale_contours_to_original([highlighted_contour], self.parent_app.scale_factor)[0]
            else:
                # No scaling needed
                scaled_contour = highlighted_contour
                
            cv2.drawContours(
                self.parent_app.processed_image, 
                [scaled_contour], 
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
        
    def calculate_point_to_line_distance(self, x, y, x1, y1, x2, y2):
        """Calculate the distance from point (x,y) to line segment (x1,y1)-(x2,y2)."""
        import math
        
        # Line segment length squared
        l2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
        
        if l2 == 0:  # Line segment is a point
            return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
        
        # Calculate projection of point onto line
        t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / l2
        
        # If projection is outside segment, calculate distance to endpoints
        if t < 0:
            return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
        elif t > 1:
            return math.sqrt((x - x2) ** 2 + (y - y2) ** 2)
        
        # Calculate distance to line
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        return math.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2)
    
    def update_region(self, region_image, x, y, width, height):
        """Update only a specific region of the display for improved performance.
        
        Args:
            region_image: The numpy image data for the region to update
            x, y: The top-left coordinates of the region in image space
            width, height: The dimensions of the region
        """
        if self.base_pixmap is None:
            return
            
        # Convert region image to QImage
        if len(region_image.shape) == 3:
            if region_image.shape[2] == 3:  # RGB
                bytes_per_line = 3 * width
                qimg_format = QImage.Format.Format_RGB888
            elif region_image.shape[2] == 4:  # RGBA
                bytes_per_line = 4 * width
                qimg_format = QImage.Format.Format_RGBA8888
        else:  # Grayscale
            bytes_per_line = width
            qimg_format = QImage.Format.Format_Grayscale8
            
        region_qimg = QImage(region_image.data.tobytes(), width, height, bytes_per_line, qimg_format)
        region_pixmap = QPixmap.fromImage(region_qimg)
        
        # Get the display position accounting for zoom and pan
        display_x = int(x * self.zoom_factor + self.pan_offset.x())
        display_y = int(y * self.zoom_factor + self.pan_offset.y())
        display_width = int(width * self.zoom_factor)
        display_height = int(height * self.zoom_factor)
        
        # Create a scaled version of the region pixmap if zoomed
        if self.zoom_factor != 1.0:
            region_pixmap = region_pixmap.scaled(
                display_width, 
                display_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        # Create a painter to update just this region of the displayed pixmap
        current_pixmap = self.pixmap()
        if current_pixmap is None:
            self.update_display()  # Fall back to full update if no pixmap exists
            return
            
        painter = QPainter(current_pixmap)
        painter.drawPixmap(display_x, display_y, region_pixmap)
        painter.end()
          # Set the updated pixmap
        self.setPixmap(current_pixmap)
        
        # Store the last updated region for potential future optimizations
        self.last_updated_region = (display_x, display_y, display_width, display_height)
        
        # Request a repaint of just this region for efficiency
        self.update(display_x, display_y, display_width, display_height)
    def draw_brush_overlay_on_region(self, img_x, img_y, brush_size, is_erase_mode=False):
        """Draw brush overlay directly on the display pixmap for ultra-fast preview.
        
        Args:
            img_x, img_y: Brush position in image coordinates
            brush_size: Size of brush in pixels
            is_erase_mode: Whether in erase mode (affects brush outline color)
        """
        if self.base_pixmap is None:
            return
            
        # Store last brush position for optimization
        if not hasattr(self, 'last_brush_position'):
            self.last_brush_position = None
            self.last_brush_size = None
            self.last_brush_is_erase = None
        
        # Create a backup of the current pixmap if it doesn't already exist
        if not hasattr(self, 'original_display_pixmap') or self.original_display_pixmap is None:
            # Store a clean copy of the current display pixmap 
            # that we can use as our base for drawing overlays
            # This ensures we're always starting with an unmodified view
            self.reset_brush_overlay()
            self.original_display_pixmap = self.pixmap().copy()
        
        # Calculate region that will be affected by the brush
        # Add padding for the brush outline
        region_x = max(0, img_x - brush_size - 2)
        region_y = max(0, img_y - brush_size - 2)
        region_width = min(brush_size * 2 + 4, self.base_pixmap.width() - region_x)
        region_height = min(brush_size * 2 + 4, self.base_pixmap.height() - region_y)
        
        # Convert to display coordinates
        display_x = int(region_x * self.zoom_factor + self.pan_offset.x())
        display_y = int(region_y * self.zoom_factor + self.pan_offset.y())
        display_brush_x = int(img_x * self.zoom_factor + self.pan_offset.x())
        display_brush_y = int(img_y * self.zoom_factor + self.pan_offset.y())
        scaled_brush_size = int(brush_size * self.zoom_factor)
        
        # We'll always start with our clean original pixmap
        # This ensures we don't build up tinting or artifacts from multiple overlays
        temp_display_pixmap = self.original_display_pixmap.copy()
        
        # Set up for pure overlay drawing using a high-contrast outline
        from PyQt6.QtGui import QPen
        if is_erase_mode:
            # Use a red pen for erase mode
            pen = QPen(Qt.GlobalColor.red)
        else:
            # Use a green pen for draw mode 
            pen = QPen(Qt.GlobalColor.green)
            
        pen.setWidth(2)  # Make it slightly thicker for better visibility
        painter = QPainter(temp_display_pixmap)
        painter.setPen(pen)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        
        # Draw only the outline (not filled)
        painter.drawEllipse(display_brush_x - scaled_brush_size, 
                           display_brush_y - scaled_brush_size,
                           scaled_brush_size * 2, scaled_brush_size * 2)
        painter.end()
        
        # Set the updated pixmap
        self.setPixmap(temp_display_pixmap)
        
        # Store position for optimization
        self.last_brush_position = (display_brush_x, display_brush_y)
        self.last_brush_size = scaled_brush_size
        self.last_brush_is_erase = is_erase_mode
        
        # Update last position
        self.last_brush_position = (display_brush_x, display_brush_y)
        self.last_brush_size = brush_size
        self.last_brush_is_erase = is_erase_mode
          # Request a repaint of just this region for efficiency
        update_width = int(region_width * self.zoom_factor)
        update_height = int(region_height * self.zoom_factor)
        self.update(display_x, display_y, update_width, update_height)
    def reset_brush_overlay(self):
        """Reset the brush overlay, restoring the original display without the brush preview."""
        # Reset tracking variables
        self.last_brush_position = None
        self.last_brush_size = None
        self.last_brush_is_erase = None
        
        # Reset the stored original display pixmap to force a clean recreation
        self.original_display_pixmap = None
            
        # Force a clean update of the display to remove any overlays
        if self.base_pixmap is not None:
            self.update_display()
    
    def draw_shape_overlay_circle(self, center_x, center_y, radius, thickness, is_erase_mode=False):
        """Draw circle overlay directly on the display pixmap without any color tinting.
        
        Args:
            center_x, center_y: Center position of circle in image coordinates
            radius: Radius of circle in pixels
            thickness: Outline thickness in pixels
            is_erase_mode: Whether in erase mode (affects outline color)
        """
        if self.base_pixmap is None:
            return
            
        # Create a backup of the current pixmap if it doesn't already exist
        if not hasattr(self, 'original_display_pixmap') or self.original_display_pixmap is None:
            self.original_display_pixmap = self.pixmap().copy()
            
        # Calculate the region that will be affected
        region_x = max(0, center_x - radius - thickness - 2)
        region_y = max(0, center_y - radius - thickness - 2)
        region_width = min((radius + thickness + 2) * 2, self.base_pixmap.width() - region_x)
        region_height = min((radius + thickness + 2) * 2, self.base_pixmap.height() - region_y)
        
        # Convert to display coordinates
        display_center_x = int(center_x * self.zoom_factor + self.pan_offset.x())
        display_center_y = int(center_y * self.zoom_factor + self.pan_offset.y())
        scaled_radius = int(radius * self.zoom_factor)
        scaled_thickness = max(2, int(thickness * self.zoom_factor))
            
        # Always start with our clean original pixmap
        temp_display_pixmap = self.original_display_pixmap.copy()
        
        # Set up for pure overlay drawing using a high-contrast outline
        from PyQt6.QtGui import QPen
        if is_erase_mode:
            # Use a red pen for erase mode
            pen = QPen(Qt.GlobalColor.red)
        else:
            # Use a green pen for draw mode 
            pen = QPen(Qt.GlobalColor.green)
            
        pen.setWidth(scaled_thickness)
        
        painter = QPainter(temp_display_pixmap)
        painter.setPen(pen)
        # Draw only the outline (not filled)
        painter.drawEllipse(
            display_center_x - scaled_radius, 
            display_center_y - scaled_radius,
            scaled_radius * 2, 
            scaled_radius * 2
        )
        painter.end()
        
        # Set the updated pixmap
        self.setPixmap(temp_display_pixmap)
            
    def draw_shape_overlay_ellipse(self, center_x, center_y, width_radius, height_radius, thickness, is_erase_mode=False):
        """Draw ellipse overlay directly on the display pixmap without any color tinting.
        
        Args:
            center_x, center_y: Center position of ellipse in image coordinates
            width_radius, height_radius: Half-width and half-height of ellipse in pixels
            thickness: Outline thickness in pixels
            is_erase_mode: Whether in erase mode (affects outline color)
        """
        if self.base_pixmap is None:
            return
            
        # Create a backup of the current pixmap if it doesn't already exist
        if not hasattr(self, 'original_display_pixmap') or self.original_display_pixmap is None:
            self.original_display_pixmap = self.pixmap().copy()
            
        # Convert to display coordinates
        display_center_x = int(center_x * self.zoom_factor + self.pan_offset.x())
        display_center_y = int(center_y * self.zoom_factor + self.pan_offset.y())
        scaled_width_radius = int(width_radius * self.zoom_factor)
        scaled_height_radius = int(height_radius * self.zoom_factor)
        scaled_thickness = max(2, int(thickness * self.zoom_factor))
            
        # Always start with our clean original pixmap
        temp_display_pixmap = self.original_display_pixmap.copy()
        
        # Set up for pure overlay drawing using a high-contrast outline
        from PyQt6.QtGui import QPen
        if is_erase_mode:
            # Use a red pen for erase mode
            pen = QPen(Qt.GlobalColor.red)
        else:
            # Use a green pen for draw mode 
            pen = QPen(Qt.GlobalColor.green)
            
        pen.setWidth(scaled_thickness)
        
        painter = QPainter(temp_display_pixmap)
        painter.setPen(pen)
        # Draw only the outline (not filled)
        painter.drawEllipse(
            display_center_x - scaled_width_radius, 
            display_center_y - scaled_height_radius,
            scaled_width_radius * 2, 
            scaled_height_radius * 2
        )
        painter.end()
        
        # Set the updated pixmap
        self.setPixmap(temp_display_pixmap)
            
    def draw_shape_overlay_rectangle(self, x1, y1, x2, y2, thickness, is_erase_mode=False):
        """Draw rectangle overlay directly on the display pixmap without any color tinting.
        
        Args:
            x1, y1: Top-left corner in image coordinates
            x2, y2: Bottom-right corner in image coordinates
            thickness: Outline thickness in pixels
            is_erase_mode: Whether in erase mode (affects outline color)
        """
        if self.base_pixmap is None:
            return
            
        # Create a backup of the current pixmap if it doesn't already exist
        if not hasattr(self, 'original_display_pixmap') or self.original_display_pixmap is None:
            self.original_display_pixmap = self.pixmap().copy()
            
        # Convert to display coordinates
        display_x1 = int(x1 * self.zoom_factor + self.pan_offset.x())
        display_y1 = int(y1 * self.zoom_factor + self.pan_offset.y())
        display_x2 = int(x2 * self.zoom_factor + self.pan_offset.x())
        display_y2 = int(y2 * self.zoom_factor + self.pan_offset.y())
        scaled_thickness = max(2, int(thickness * self.zoom_factor))
            
        # Always start with our clean original pixmap
        temp_display_pixmap = self.original_display_pixmap.copy()
        
        # Set up for pure overlay drawing using a high-contrast outline
        from PyQt6.QtGui import QPen
        if is_erase_mode:
            # Use a red pen for erase mode
            pen = QPen(Qt.GlobalColor.red)
        else:
            # Use a green pen for draw mode 
            pen = QPen(Qt.GlobalColor.green)
            
        pen.setWidth(scaled_thickness)
        
        painter = QPainter(temp_display_pixmap)
        painter.setPen(pen)
        # Draw only the outline (not filled)
        painter.drawRect(
            min(display_x1, display_x2), 
            min(display_y1, display_y2),
            abs(display_x2 - display_x1), 
            abs(display_y2 - display_y1)
        )
        painter.end()
        
        # Set the updated pixmap
        self.setPixmap(temp_display_pixmap)
            
    def draw_shape_overlay_line(self, x1, y1, x2, y2, thickness, is_erase_mode=False):
        """Draw line overlay directly on the display pixmap without any color tinting.
        
        Args:
            x1, y1: Start point in image coordinates
            x2, y2: End point in image coordinates
            thickness: Line thickness in pixels
            is_erase_mode: Whether in erase mode (affects outline color)
        """
        if self.base_pixmap is None:
            return
            
        # Create a backup of the current pixmap if it doesn't already exist
        if not hasattr(self, 'original_display_pixmap') or self.original_display_pixmap is None:
            self.original_display_pixmap = self.pixmap().copy()
            
        # Convert to display coordinates
        display_x1 = int(x1 * self.zoom_factor + self.pan_offset.x())
        display_y1 = int(y1 * self.zoom_factor + self.pan_offset.y())
        display_x2 = int(x2 * self.zoom_factor + self.pan_offset.x())
        display_y2 = int(y2 * self.zoom_factor + self.pan_offset.y())
        scaled_thickness = max(2, int(thickness * self.zoom_factor))
            
        # Always start with our clean original pixmap
        temp_display_pixmap = self.original_display_pixmap.copy()
        
        # Set up for pure overlay drawing using a high-contrast outline
        from PyQt6.QtGui import QPen
        if is_erase_mode:
            # Use a red pen for erase mode
            pen = QPen(Qt.GlobalColor.red)
        else:
            # Use a green pen for draw mode 
            pen = QPen(Qt.GlobalColor.green)
            
        pen.setWidth(scaled_thickness)
        
        painter = QPainter(temp_display_pixmap)
        painter.setPen(pen)
        # Draw the line
        painter.drawLine(
            display_x1, 
            display_y1,
            display_x2, 
            display_y2
        )
        painter.end()
        
        # Set the updated pixmap
        self.setPixmap(temp_display_pixmap)