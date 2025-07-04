from PyQt6.QtWidgets import (QLabel)
from PyQt6.QtCore import Qt, QPoint, QPointF
from PyQt6.QtGui import QWheelEvent, QTransform, QPainter, QPixmap, QImage, QCursor
import cv2

from src.utils.geometry import convert_to_image_coordinates

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
            
            # Convert display coordinates to image coordinates
            img_pos = self.display_to_image_coords(QPoint(x, y))
            if img_pos is None:
                super().mousePressEvent(event)
                return
                
            img_x, img_y = img_pos
            
            # Special handling for UVTT preview mode
            if self.parent_app.uvtt_preview_active:
                if event.button() == Qt.MouseButton.LeftButton:
                    if self.parent_app.uvtt_draw_mode:
                        # Save the current wall state for undo before drawing a new wall
                        self.parent_app.export_panel.save_wall_state_for_undo()
                        
                        # Start drawing a new wall
                        self.parent_app.drawing_new_wall = True
                        
                        # Check if Ctrl is held and if we have a previous wall to continue from
                        if (event.modifiers() & Qt.KeyboardModifier.ControlModifier and 
                            '_preview_pixels' in self.parent_app.uvtt_walls_preview and 
                            len(self.parent_app.uvtt_walls_preview['_preview_pixels']) > 0):
                            
                            # Use the endpoint of the last wall as the starting point for the new wall
                            last_wall = self.parent_app.uvtt_walls_preview['_preview_pixels'][-1]
                            # Get the last point of the last wall
                            if len(last_wall) > 1:
                                last_x = float(last_wall[1]["x"])
                                last_y = float(last_wall[1]["y"])
                                self.parent_app.new_wall_start = (last_x, last_y)
                                self.parent_app.new_wall_end = (img_x, img_y)
                                self.parent_app.setStatusTip("Continuing wall from previous endpoint")
                            else:
                                # Fallback if the last wall is malformed
                                self.parent_app.new_wall_start = (img_x, img_y)
                                self.parent_app.new_wall_end = (img_x, img_y)
                        else:
                            # Standard wall starting at cursor
                            self.parent_app.new_wall_start = (img_x, img_y)
                            self.parent_app.new_wall_end = (img_x, img_y)  # Initialize with same point
                        
                    elif self.parent_app.uvtt_edit_mode:
                        # First check if we're clicking on a wall point
                        wall_idx, point_idx = self.find_closest_wall_point(img_x, img_y)
                        if wall_idx != -1:
                            # Found a wall point
                            if not hasattr(self.parent_app, 'selected_points'):
                                self.parent_app.selected_points = []
                            
                            point_is_selected = (wall_idx, point_idx) in self.parent_app.selected_points
                            
                            if point_is_selected:
                                # This point is already selected - start moving all selected points
                                self.parent_app.export_panel.save_wall_state_for_undo()
                                self.store_initial_positions_for_points()
                                self.parent_app.multi_wall_drag = True
                                self.parent_app.multi_wall_drag_start = (img_x, img_y)
                                self.parent_app.dragging_from_line = False  # We're dragging points
                                self.parent_app.setStatusTip(f"Moving {len(self.parent_app.selected_points)} selected points")
                                print(f"Starting drag of {len(self.parent_app.selected_points)} selected points")
                            else:
                                # This point is not selected - clear selection and move just this point
                                self.parent_app.selected_wall_indices = []
                                self.parent_app.selected_points = [(wall_idx, point_idx)]
                                self.parent_app.selected_wall_index = wall_idx
                                self.parent_app.selected_point_index = point_idx
                                
                                self.parent_app.export_panel.save_wall_state_for_undo()
                                self.store_initial_positions_for_points()
                                self.parent_app.multi_wall_drag = True
                                self.parent_app.multi_wall_drag_start = (img_x, img_y)
                                self.parent_app.dragging_from_line = False  # We're dragging a point
                                self.parent_app.setStatusTip(f"Moving single wall point")
                                print(f"Starting drag of single point {point_idx} on wall {wall_idx}")
                            return
                        
                        # Check if we're clicking on a wall line
                        wall_idx = self.find_wall_under_cursor(img_x, img_y)
                        if wall_idx != -1:
                            # Found a wall line
                            wall_is_selected = wall_idx in self.parent_app.selected_wall_indices
                            
                            if wall_is_selected:
                                # This wall is already selected - start moving all selected walls
                                self.parent_app.export_panel.save_wall_state_for_undo()
                                self.store_initial_positions_for_walls()
                                self.parent_app.multi_wall_drag = True
                                self.parent_app.multi_wall_drag_start = (img_x, img_y)
                                self.parent_app.dragging_from_line = True  # We're dragging wall lines
                                self.parent_app.setStatusTip(f"Moving {len(self.parent_app.selected_wall_indices)} selected walls")
                                print(f"Starting drag of {len(self.parent_app.selected_wall_indices)} selected walls")
                            else:
                                # This wall is not selected - clear selection and move just this wall
                                self.parent_app.selected_wall_indices = [wall_idx]
                                self.parent_app.selected_points = []
                                self.parent_app.selected_wall_index = wall_idx
                                self.parent_app.selected_point_index = -1
                                
                                self.parent_app.export_panel.save_wall_state_for_undo()
                                self.store_initial_positions_for_walls()
                                self.parent_app.multi_wall_drag = True
                                self.parent_app.multi_wall_drag_start = (img_x, img_y)
                                self.parent_app.dragging_from_line = True  # We're dragging a wall line
                                self.parent_app.setStatusTip(f"Moving single wall")
                                print(f"Starting drag of single wall {wall_idx}")
                            
                            # Update the display to show selection
                            self.parent_app.export_panel.display_uvtt_preview()
                            return
                        
                        # No wall point or line was clicked - clear all selections and start selection box
                        # This ensures that clicking on empty space always clears existing selections
                        self.parent_app.selected_wall_indices = []
                        self.parent_app.selected_points = []
                        self.parent_app.selected_wall_index = -1
                        self.parent_app.selected_point_index = -1
                        
                        # Clear any previous drag operations
                        self.parent_app.multi_wall_drag = False
                        self.parent_app.multi_wall_drag_start = None
                        self.parent_app.dragging_from_line = False
                        
                        self.parent_app.selecting_walls = True
                        self.parent_app.wall_selection_start = (img_x, img_y)
                        self.parent_app.wall_selection_current = (img_x, img_y)
                        print(f"Starting selection box at ({img_x}, {img_y})")
                            
                    elif self.parent_app.uvtt_delete_mode:
                        # First check if we're directly clicking on a wall for immediate deletion
                        wall_idx = self.find_wall_under_cursor(img_x, img_y)
                        if wall_idx != -1:
                            # Save current state for undo - use force=True to ensure state is saved
                            # This ensures each delete operation can be individually undone
                            self.parent_app.export_panel.save_wall_state_for_undo(force=True)
                            
                            # Delete the wall
                            if self.parent_app.uvtt_walls_preview and '_preview_pixels' in self.parent_app.uvtt_walls_preview:
                                # Delete from both collections
                                if wall_idx < len(self.parent_app.uvtt_walls_preview['line_of_sight']):
                                    del self.parent_app.uvtt_walls_preview['line_of_sight'][wall_idx]
                                
                                if wall_idx < len(self.parent_app.uvtt_walls_preview['_preview_pixels']):
                                    del self.parent_app.uvtt_walls_preview['_preview_pixels'][wall_idx]
                                
                                # Reset selection
                                self.parent_app.selected_wall_index = -1
                                self.parent_app.selected_point_index = -1
                                
                                # Ensure we stay in delete mode
                                self.parent_app.uvtt_delete_mode = True
                                
                                # Update the display
                                self.parent_app.export_panel.display_uvtt_preview()
                                self.parent_app.setStatusTip(f"Deleted wall {wall_idx}")
                                
                                # Save the final state for undo
                                self.parent_app.export_panel.save_wall_state_for_undo(force=True)
                        else:
                            # No wall was clicked - start a selection box
                            self.parent_app.selecting_walls = True
                            self.parent_app.wall_selection_start = (img_x, img_y)
                            self.parent_app.wall_selection_current = (img_x, img_y)
                            # Clear existing selection when starting a new one
                            self.parent_app.selected_wall_indices = []
                    
                    # Update the preview
                    self.parent_app.export_panel.display_uvtt_preview()
                    return
            
            # Handle regular mode clicks
            elif event.button() == Qt.MouseButton.LeftButton:
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
                
            # Convert display coordinates to image coordinates
            img_pos = self.display_to_image_coords(QPoint(x, y))
            
            # Handle UVTT preview mode
            if self.parent_app.uvtt_preview_active and img_pos is not None:
                img_x, img_y = img_pos
                
                # Show wall preview when Ctrl is held in drawing mode (but not actively drawing)
                if (self.parent_app.uvtt_draw_mode and 
                    not self.parent_app.drawing_new_wall and 
                    event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                    
                    # Store mouse position for preview line display
                    self.parent_app.preview_mouse_pos = (img_x, img_y)
                    self.parent_app.ctrl_held_for_preview = True
                    
                    # Update display to show preview
                    self.parent_app.export_panel.display_uvtt_preview()
                elif (self.parent_app.uvtt_draw_mode and 
                      not self.parent_app.drawing_new_wall and
                      hasattr(self.parent_app, 'ctrl_held_for_preview') and
                      self.parent_app.ctrl_held_for_preview):
                    # Ctrl was released, clear preview
                    self.parent_app.ctrl_held_for_preview = False
                    self.parent_app.export_panel.display_uvtt_preview()
                
                # Left button dragging for various UVTT editing operations
                if event.buttons() & Qt.MouseButton.LeftButton:
                    if self.parent_app.uvtt_draw_mode and self.parent_app.drawing_new_wall:
                        # Update the end point of the wall being drawn
                        self.parent_app.new_wall_end = (img_x, img_y)
                        self.parent_app.export_panel.display_uvtt_preview()
                        return
                        
                    elif self.parent_app.uvtt_edit_mode:
                        # Check if we're doing a multi-wall/point drag operation
                        if hasattr(self.parent_app, 'multi_wall_drag') and self.parent_app.multi_wall_drag and hasattr(self.parent_app, 'multi_wall_drag_start'):
                            print(f"MouseMove: Multi-drag active, dragging_from_line={getattr(self.parent_app, 'dragging_from_line', 'NOT_SET')}")
                            
                            # Calculate movement delta from the initial drag start position
                            start_x, start_y = self.parent_app.multi_wall_drag_start
                            dx = img_x - start_x
                            dy = img_y - start_y
                            
                            # Don't update the drag start point - keep it as the original starting position
                            # This ensures we calculate the total movement from the initial click position
                                
                            # Handle different cases based on what was initially clicked:
                            # 1. If dragging from a wall line: move the entire wall(s)
                            # 2. If dragging from a point: move only that point and any other selected points
                            
                            # Check if we're dragging from a line or a point
                            if self.parent_app.dragging_from_line:
                                print(f"MouseMove: Dragging from line, selected_wall_indices={getattr(self.parent_app, 'selected_wall_indices', 'NOT_SET')}")
                                # We're dragging from a wall line, so move entire walls
                                if self.move_selected_walls_absolute(img_x, img_y):
                                    self.parent_app.export_panel.display_uvtt_preview()
                                    return
                            elif hasattr(self.parent_app, 'selected_points') and self.parent_app.selected_points:
                                print(f"MouseMove: Dragging from points, selected_points={getattr(self.parent_app, 'selected_points', 'NOT_SET')}")
                                # We're dragging from selected points, move only those points
                                if self.move_selected_wall_points_absolute(img_x, img_y):
                                    self.parent_app.export_panel.display_uvtt_preview()
                                    return
                                
                            # If we get here, we didn't find any special drag case, so just update the display
                            print("MouseMove: No specific drag case found, updating display")
                            self.parent_app.export_panel.display_uvtt_preview()
                        
                        # Single point movement (endpoint drag) - only if we have a specific selected wall/point
                        elif (self.parent_app.selected_wall_index != -1 and 
                              '_preview_pixels' in self.parent_app.uvtt_walls_preview and 
                              self.parent_app.selected_wall_index < len(self.parent_app.uvtt_walls_preview['_preview_pixels'])):
                            
                            wall_idx = self.parent_app.selected_wall_index
                            point_idx = self.parent_app.selected_point_index
                            
                            # Update the pixel coordinates
                            wall_points = self.parent_app.uvtt_walls_preview['_preview_pixels'][wall_idx]
                            if point_idx < len(wall_points):
                                # Track the movement but don't save state during drag operation
                                old_x = float(wall_points[point_idx]["x"])
                                old_y = float(wall_points[point_idx]["y"])
                                
                                # Update coordinates
                                wall_points[point_idx]["x"] = float(img_x)
                                wall_points[point_idx]["y"] = float(img_y)
                                
                                # Don't save state during endpoint dragging - only at start and end
                                # State is already saved in mousePressEvent and will be saved on mouseReleaseEvent
                                
                                # Update the grid coordinates in the UVTT data
                                if 'line_of_sight' in self.parent_app.uvtt_walls_preview and wall_idx < len(self.parent_app.uvtt_walls_preview['line_of_sight']):
                                    # Get the grid size
                                    grid_size = self.parent_app.uvtt_walls_preview['resolution']['pixels_per_grid']
                                    if grid_size <= 0:
                                        grid_size = 70  # Default
                                    
                                    # Update grid coordinates
                                    self.parent_app.uvtt_walls_preview['line_of_sight'][wall_idx][point_idx]["x"] = float(img_x / grid_size)
                                    self.parent_app.uvtt_walls_preview['line_of_sight'][wall_idx][point_idx]["y"] = float(img_y / grid_size)
                                
                                self.parent_app.export_panel.display_uvtt_preview()
                                return
                        
                        # Handle wall selection box updates (for edit mode)
                        elif self.parent_app.selecting_walls:
                            # Update the current point of the selection box
                            self.parent_app.wall_selection_current = (img_x, img_y)
                            # Update walls in the selection box
                            self.update_walls_in_selection()
                            # Update the display
                            self.parent_app.export_panel.display_uvtt_preview()
                            return
                    
                    # Handle wall selection box updates (for delete mode)
                    elif self.parent_app.uvtt_delete_mode and self.parent_app.selecting_walls:
                        # Update the current point of the selection box
                        self.parent_app.wall_selection_current = (img_x, img_y)
                        # Update walls in the selection box
                        self.update_walls_in_selection()
                        # Update the display
                        self.parent_app.export_panel.display_uvtt_preview()
                        return
            
            # If dragging with left button in regular mode
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
                
                # Handle UVTT preview mode
                if self.parent_app.uvtt_preview_active:
                    # Convert display coordinates to image coordinates
                    img_pos = self.display_to_image_coords(QPoint(x, y))
                    if img_pos is not None:
                        img_x, img_y = img_pos
                        
                        if self.parent_app.uvtt_draw_mode and self.parent_app.drawing_new_wall:
                            # Finish drawing a new wall if the start and end points are different
                            if (self.parent_app.new_wall_start is not None and 
                                self.parent_app.new_wall_end is not None and
                                self.parent_app.new_wall_start != self.parent_app.new_wall_end):
                                
                                # Extract start and end points
                                start_x, start_y = self.parent_app.new_wall_start
                                end_x, end_y = self.parent_app.new_wall_end
                                
                                # Create a new wall
                                if self.parent_app.uvtt_walls_preview:
                                    # State has already been saved in mousePressEvent before starting to draw the wall
                                    
                                    # Get the grid size
                                    grid_size = self.parent_app.uvtt_walls_preview['resolution']['pixels_per_grid']
                                    if grid_size <= 0:
                                        grid_size = 70  # Default
                                    
                                    # Create wall points in pixel coordinates
                                    wall_pixels = [
                                        {"x": float(start_x), "y": float(start_y)},
                                        {"x": float(end_x), "y": float(end_y)}
                                    ]
                                    
                                    # Create wall points in grid coordinates
                                    wall_grid = [
                                        {"x": float(start_x / grid_size), "y": float(start_y / grid_size)},
                                        {"x": float(end_x / grid_size), "y": float(end_y / grid_size)}
                                    ]
                                    
                                    # Add to the walls list
                                    self.parent_app.uvtt_walls_preview['_preview_pixels'].append(wall_pixels)
                                    self.parent_app.uvtt_walls_preview['line_of_sight'].append(wall_grid)
                                    
                                    self.parent_app.setStatusTip(f"Added new wall from ({start_x:.1f}, {start_y:.1f}) to ({end_x:.1f}, {end_y:.1f})")
                            
                            # Reset drawing state
                            self.parent_app.drawing_new_wall = False
                            self.parent_app.new_wall_start = None
                            self.parent_app.new_wall_end = None
                            
                            # Save the final state for undo
                            self.parent_app.export_panel.save_wall_state_for_undo(force=True)
                            
                        elif self.parent_app.uvtt_edit_mode and self.parent_app.selected_wall_index != -1:
                            # Finish editing a wall point or multi-wall drag
                            if hasattr(self.parent_app, 'multi_wall_drag') and self.parent_app.multi_wall_drag:
                                # Finish multi-wall drag
                                self.parent_app.multi_wall_drag = False
                                self.parent_app.multi_wall_drag_start = None
                                
                                # Handle based on what was dragged
                                if self.parent_app.dragging_from_line:
                                    # Status update with count of moved walls
                                    wall_count = len(self.parent_app.selected_wall_indices)
                                    if wall_count > 1:
                                        self.parent_app.setStatusTip(f"Moved {wall_count} walls")
                                    else:
                                        self.parent_app.setStatusTip(f"Moved wall")
                                    
                                    print(f"Dragging completed: Moved {wall_count} walls by dragging line")
                                else:
                                    # Clean up special point selection data
                                    if hasattr(self.parent_app, 'selected_points'):
                                        point_count = len(self.parent_app.selected_points)
                                        if point_count > 1:
                                            self.parent_app.setStatusTip(f"Moved {point_count} points")
                                        else:
                                            self.parent_app.setStatusTip(f"Moved wall point")
                                        
                                        print(f"Dragging completed: Moved {point_count} point(s) by dragging point")
                                        # Don't clear selected_points here - keep them selected for future operations
                                
                                # Reset the dragging_from_line flag
                                self.parent_app.dragging_from_line = False
                                
                                # Clean up initial position data
                                if hasattr(self.parent_app, 'initial_wall_positions'):
                                    delattr(self.parent_app, 'initial_wall_positions')
                                if hasattr(self.parent_app, 'initial_point_positions'):
                                    delattr(self.parent_app, 'initial_point_positions')
                                
                                # Save the final state for undo
                                self.parent_app.export_panel.save_wall_state_for_undo()
                            else:
                                # Finish single point move
                                self.parent_app.setStatusTip(f"Moved wall point")
                                print(f"Dragging completed: Moved single point")
                                
                                # Clean up initial position data
                                if hasattr(self.parent_app, 'initial_wall_positions'):
                                    delattr(self.parent_app, 'initial_wall_positions')
                                if hasattr(self.parent_app, 'initial_point_positions'):
                                    delattr(self.parent_app, 'initial_point_positions')
                                
                                # Save the final state for undo
                                self.parent_app.export_panel.save_wall_state_for_undo()
                            
                            # Reset drag-specific state but keep selections
                            self.parent_app.selected_wall_index = -1
                            self.parent_app.selected_point_index = -1
                            
                            # Clean up any endpoint move tracking
                            if hasattr(self.parent_app, 'last_endpoint_move_position'):
                                delattr(self.parent_app, 'last_endpoint_move_position')
                            if hasattr(self.parent_app, 'last_endpoint_move_time'):
                                delattr(self.parent_app, 'last_endpoint_move_time')
                            
                            # Ensure we stay in edit mode
                            self.parent_app.uvtt_edit_mode = True
                            
                        # Handle completing a wall selection box
                        elif (self.parent_app.uvtt_edit_mode or self.parent_app.uvtt_delete_mode) and self.parent_app.selecting_walls:
                            # Finalize the selection
                            self.parent_app.selecting_walls = False
                            
                            # If in delete mode with points selected, delete the walls that have selected points
                            if self.parent_app.uvtt_delete_mode and self.parent_app.selected_wall_indices:
                                self.handle_selected_walls_deletion()
                            
                            # If in edit mode with points selected, provide feedback
                            if self.parent_app.uvtt_edit_mode and hasattr(self.parent_app, 'selected_points'):
                                # selected_points is already populated by update_walls_in_selection
                                wall_count = len(self.parent_app.selected_wall_indices)
                                point_count = len(self.parent_app.selected_points)
                                if point_count > 0:
                                    self.parent_app.setStatusTip(f"Selected {point_count} points from {wall_count} walls. Drag points to move them.")
                                else:
                                    self.parent_app.setStatusTip("No points selected in the selection box.")
                        
                        
                        # Update the preview
                        self.parent_app.export_panel.display_uvtt_preview()
                    
                    # Reset selection points
                    self.selection_start = None
                    return
                
                # Regular mode handling
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
            
        # Use refresh_display to preserve grid overlay and other overlays
        self.parent_app.refresh_display()

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
        
    def find_closest_wall_point(self, x, y, max_distance=10):
        """Find the closest wall endpoint to the given coordinates.
        
        Args:
            x, y: Image coordinates to check
            max_distance: Maximum distance to consider a point close enough
            
        Returns:
            Tuple of (wall_index, point_index) or (-1, -1) if none found within distance
        """
        if not self.parent_app.uvtt_preview_active or not self.parent_app.uvtt_walls_preview:
            return -1, -1
            
        if '_preview_pixels' not in self.parent_app.uvtt_walls_preview:
            return -1, -1
            
        closest_wall = -1
        closest_point = -1
        min_distance = max_distance  # Initialize with max threshold
        
        # Check all wall endpoints
        wall_points_list = self.parent_app.uvtt_walls_preview['_preview_pixels']
        for wall_idx, wall_points in enumerate(wall_points_list):
            # Each wall is a list of points
            if len(wall_points) < 2:
                continue
                
            # Check start point
            start_x = wall_points[0]["x"]
            start_y = wall_points[0]["y"]
            start_distance = ((start_x - x)**2 + (start_y - y)**2)**0.5
            
            if start_distance < min_distance:
                min_distance = start_distance
                closest_wall = wall_idx
                closest_point = 0
                
            # Check end point
            end_x = wall_points[1]["x"]
            end_y = wall_points[1]["y"]
            end_distance = ((end_x - x)**2 + (end_y - y)**2)**0.5
            
            if end_distance < min_distance:
                min_distance = end_distance
                closest_wall = wall_idx
                closest_point = 1
        
        return closest_wall, closest_point
        
    def find_all_wall_points_in_radius(self, x, y, radius=10):
        """Find all wall endpoints within the specified radius of the given coordinates.
        
        Args:
            x, y: Image coordinates to check
            radius: Maximum distance to consider a point close enough
            
        Returns:
            List of (wall_index, point_index) tuples for points within the radius
        """
        if not self.parent_app.uvtt_preview_active or not self.parent_app.uvtt_walls_preview:
            return []
            
        if '_preview_pixels' not in self.parent_app.uvtt_walls_preview:
            return []
            
        points_in_radius = []
        
        # Check all wall endpoints
        wall_points_list = self.parent_app.uvtt_walls_preview['_preview_pixels']
        for wall_idx, wall_points in enumerate(wall_points_list):
            # Each wall is a list of points
            if len(wall_points) < 2:
                continue
            
            # Check every point in the wall
            for point_idx, point in enumerate(wall_points):
                point_x = point["x"]
                point_y = point["y"]
                distance = ((point_x - x)**2 + (point_y - y)**2)**0.5
                
                if distance <= radius:
                    points_in_radius.append((wall_idx, point_idx))
        
        return points_in_radius

    def find_wall_under_cursor(self, x, y, max_distance=5):
        """Find a wall segment close to the cursor position.
        
        Args:
            x, y: Image coordinates to check
            max_distance: Maximum distance from line to consider it selected
            
        Returns:
            Wall index or -1 if none found
        """
        if not self.parent_app.uvtt_preview_active or not self.parent_app.uvtt_walls_preview:
            return -1
            
        if '_preview_pixels' not in self.parent_app.uvtt_walls_preview:
            return -1
            
        closest_wall = -1
        min_distance = max_distance  # Initialize with max threshold
        
        # Check all wall segments
        wall_points_list = self.parent_app.uvtt_walls_preview['_preview_pixels']
        for wall_idx, wall_points in enumerate(wall_points_list):
            # Each wall is a list of points
            if len(wall_points) < 2:
                continue
                
            # Get wall coordinates
            start_x = wall_points[0]["x"]
            start_y = wall_points[0]["y"]
            end_x = wall_points[1]["x"]
            end_y = wall_points[1]["y"]
            
            # Calculate distance from point to line segment using our local method
            distance = self.calculate_point_to_line_distance(x, y, start_x, start_y, end_x, end_y)
            
            if distance < min_distance:
                min_distance = distance
                closest_wall = wall_idx
        
        return closest_wall
    
    def update_walls_in_selection(self):
        """Update the list of wall points within the current selection box."""
        if not self.parent_app.selecting_walls or not self.parent_app.uvtt_walls_preview:
            return
            
        if not self.parent_app.wall_selection_start or not self.parent_app.wall_selection_current:
            return
            
        # Calculate selection rectangle in image coordinates
        start_x, start_y = self.parent_app.wall_selection_start
        current_x, current_y = self.parent_app.wall_selection_current
        
        x1 = min(start_x, current_x)
        y1 = min(start_y, current_y)
        x2 = max(start_x, current_x)
        y2 = max(start_y, current_y)
        
        # Reset the selections
        self.parent_app.selected_wall_indices = []
        if not hasattr(self.parent_app, 'selected_points'):
            self.parent_app.selected_points = []
        else:
            self.parent_app.selected_points = []
        
        # Check each wall point to see if it's within the selection rectangle
        if '_preview_pixels' in self.parent_app.uvtt_walls_preview:
            wall_points_list = self.parent_app.uvtt_walls_preview['_preview_pixels']
            selected_walls_set = set()  # Track which walls have points selected
            
            for wall_idx, wall_points in enumerate(wall_points_list):
                for point_idx, point in enumerate(wall_points):
                    # Get point coordinates
                    point_x = point["x"]
                    point_y = point["y"]
                    
                    # Check if this point is within the selection rectangle
                    if x1 <= point_x <= x2 and y1 <= point_y <= y2:
                        # Add this point to the selected points list
                        self.parent_app.selected_points.append((wall_idx, point_idx))
                        # Track that this wall has selected points
                        selected_walls_set.add(wall_idx)
            
            # Update selected_wall_indices to include all walls that have selected points
            # This is used for display purposes (highlighting walls that have selected points)
            self.parent_app.selected_wall_indices = list(selected_walls_set)
            
            print(f"Selection box: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
            print(f"Selected {len(self.parent_app.selected_points)} points from {len(selected_walls_set)} walls")
    
    def handle_selected_walls_deletion(self):
        """Delete all walls that are currently selected."""
        if not self.parent_app.selected_wall_indices or not self.parent_app.uvtt_walls_preview:
            return
            
        # If there are walls selected
        if self.parent_app.selected_wall_indices:
            # Count the selected walls
            count = len(self.parent_app.selected_wall_indices)
            
            # Save current state for undo - use force=True to ensure state is saved
            # We only save once before the entire batch deletion
            self.parent_app.export_panel.save_wall_state_for_undo(force=True)
            
            # Sort indices in descending order to avoid index shifting problems when deleting
            sorted_indices = sorted(self.parent_app.selected_wall_indices, reverse=True)
            
            # Delete walls in both collections
            for idx in sorted_indices:
                if 'line_of_sight' in self.parent_app.uvtt_walls_preview and idx < len(self.parent_app.uvtt_walls_preview['line_of_sight']):
                    del self.parent_app.uvtt_walls_preview['line_of_sight'][idx]
                
                if '_preview_pixels' in self.parent_app.uvtt_walls_preview and idx < len(self.parent_app.uvtt_walls_preview['_preview_pixels']):
                    del self.parent_app.uvtt_walls_preview['_preview_pixels'][idx]
            
            # Clear the selection
            self.parent_app.selected_wall_indices = []
            self.parent_app.selected_wall_index = -1
            self.parent_app.selected_point_index = -1
            
            # Ensure we stay in delete mode
            self.parent_app.uvtt_delete_mode = True
            
            # Update status and display
            self.parent_app.setStatusTip(f"Deleted {count} walls")
            self.parent_app.export_panel.display_uvtt_preview()
    
    def move_selected_walls(self, dx, dy):
        """Move all selected walls by the given delta amounts.
        
        When dragging from a wall line:
        - Move all points in the selected walls
        - This maintains the shape of all walls
        - All points in selected walls will move regardless of shared points
        - This is a simplified approach that avoids complex shared point handling
        """
        if not self.parent_app.selected_wall_indices or not self.parent_app.uvtt_walls_preview:
            return False
            
        # Track if any walls were actually moved
        walls_moved = False
            
        # Get the grid size for updating grid coordinates
        grid_size = self.parent_app.uvtt_walls_preview['resolution']['pixels_per_grid']
        if grid_size <= 0:
            grid_size = 70  # Default
            
        # Debug log
        print(f"Moving {len(self.parent_app.selected_wall_indices)} walls by dx={dx}, dy={dy}")
        
        # Move each selected wall - ALL points in the selected walls move
        for wall_idx in self.parent_app.selected_wall_indices:
            # Update pixel coordinates
            if '_preview_pixels' in self.parent_app.uvtt_walls_preview and wall_idx < len(self.parent_app.uvtt_walls_preview['_preview_pixels']):
                wall_points = self.parent_app.uvtt_walls_preview['_preview_pixels'][wall_idx]
                for point_idx, point in enumerate(wall_points):
                    # Move all points in the wall
                    point["x"] = float(point["x"] + dx)
                    point["y"] = float(point["y"] + dy)
                    walls_moved = True
                
            # Update grid coordinates
            if 'line_of_sight' in self.parent_app.uvtt_walls_preview and wall_idx < len(self.parent_app.uvtt_walls_preview['line_of_sight']):
                grid_points = self.parent_app.uvtt_walls_preview['line_of_sight'][wall_idx]
                for point_idx, point in enumerate(grid_points):
                    point["x"] = float(point["x"] + (dx / grid_size))
                    point["y"] = float(point["y"] + (dy / grid_size))
                    
        return walls_moved
        
    def move_selected_wall_points(self, dx, dy):
        """Move only specific wall points by the given delta amounts, keeping other points in place.
        
        This implements the constrained movement feature for Ctrl+click and drag operations.
        In this mode, only the points that were within the radius of the initial click will move.
        """
        if not hasattr(self.parent_app, 'selected_points') or not self.parent_app.selected_points or not self.parent_app.uvtt_walls_preview:
            return False
            
        # Track if any points were actually moved
        points_moved = False
            
        # Get the grid size for updating grid coordinates
        grid_size = self.parent_app.uvtt_walls_preview['resolution']['pixels_per_grid']
        if grid_size <= 0:
            grid_size = 70  # Default
        
        # Create a set of coordinates that were explicitly selected 
        # (these are the only points we want to move in Ctrl+drag mode)
        selected_coords = set()
        for wall_idx, point_idx in self.parent_app.selected_points:
            if (wall_idx < len(self.parent_app.uvtt_walls_preview['_preview_pixels']) and
                point_idx < len(self.parent_app.uvtt_walls_preview['_preview_pixels'][wall_idx])):
                
                point = self.parent_app.uvtt_walls_preview['_preview_pixels'][wall_idx][point_idx]
                coord_key = (round(point["x"], 4), round(point["y"], 4))
                selected_coords.add(coord_key)
        
        # Move only the selected points
        for wall_idx, point_idx in self.parent_app.selected_points:
            # Get the point's coordinates
            if ('_preview_pixels' in self.parent_app.uvtt_walls_preview and 
                wall_idx < len(self.parent_app.uvtt_walls_preview['_preview_pixels']) and
                point_idx < len(self.parent_app.uvtt_walls_preview['_preview_pixels'][wall_idx])):
                
                # Update pixel coordinates
                point = self.parent_app.uvtt_walls_preview['_preview_pixels'][wall_idx][point_idx]
                point["x"] = float(point["x"] + dx)
                point["y"] = float(point["y"] + dy)
                points_moved = True
                
                # Update grid coordinates
                if ('line_of_sight' in self.parent_app.uvtt_walls_preview and 
                    wall_idx < len(self.parent_app.uvtt_walls_preview['line_of_sight']) and
                    point_idx < len(self.parent_app.uvtt_walls_preview['line_of_sight'][wall_idx])):
                    
                    grid_point = self.parent_app.uvtt_walls_preview['line_of_sight'][wall_idx][point_idx]
                    grid_point["x"] = float(grid_point["x"] + (dx / grid_size))
                    grid_point["y"] = float(grid_point["y"] + (dy / grid_size))
                    
        return points_moved

    def store_initial_positions_for_walls(self):
        """Store the initial positions of all selected walls before starting a drag operation."""
        if not hasattr(self.parent_app, 'selected_wall_indices') or not self.parent_app.uvtt_walls_preview:
            return
            
        self.parent_app.initial_wall_positions = {}
        
        # Store initial positions for all selected walls
        for wall_idx in self.parent_app.selected_wall_indices:
            if '_preview_pixels' in self.parent_app.uvtt_walls_preview and wall_idx < len(self.parent_app.uvtt_walls_preview['_preview_pixels']):
                wall_points = self.parent_app.uvtt_walls_preview['_preview_pixels'][wall_idx]
                # Store a copy of all points in this wall
                self.parent_app.initial_wall_positions[wall_idx] = []
                for point in wall_points:
                    self.parent_app.initial_wall_positions[wall_idx].append({
                        "x": float(point["x"]),
                        "y": float(point["y"])
                    })
        
        print(f"Stored initial positions for {len(self.parent_app.initial_wall_positions)} walls")
    
    def store_initial_positions_for_points(self):
        """Store the initial positions of all selected points before starting a drag operation."""
        if not hasattr(self.parent_app, 'selected_points') or not self.parent_app.uvtt_walls_preview:
            return
            
        self.parent_app.initial_point_positions = {}
        
        # Store initial positions for all selected points
        for wall_idx, point_idx in self.parent_app.selected_points:
            if ('_preview_pixels' in self.parent_app.uvtt_walls_preview and 
                wall_idx < len(self.parent_app.uvtt_walls_preview['_preview_pixels']) and
                point_idx < len(self.parent_app.uvtt_walls_preview['_preview_pixels'][wall_idx])):
                
                point = self.parent_app.uvtt_walls_preview['_preview_pixels'][wall_idx][point_idx]
                key = (wall_idx, point_idx)
                self.parent_app.initial_point_positions[key] = {
                    "x": float(point["x"]),
                    "y": float(point["y"])
                }
        
        print(f"Stored initial positions for {len(self.parent_app.initial_point_positions)} points")

    def move_selected_walls_absolute(self, mouse_x, mouse_y):
        """Move all selected walls to absolute position based on mouse movement from initial drag start."""
        if (not hasattr(self.parent_app, 'selected_wall_indices') or 
            not hasattr(self.parent_app, 'initial_wall_positions') or
            not hasattr(self.parent_app, 'multi_wall_drag_start') or
            not self.parent_app.uvtt_walls_preview):
            print("move_selected_walls_absolute: Missing required attributes")
            return False
            
        # Calculate total movement from initial drag start
        start_x, start_y = self.parent_app.multi_wall_drag_start
        dx = mouse_x - start_x
        dy = mouse_y - start_y
        
        print(f"move_selected_walls_absolute: Moving {len(self.parent_app.selected_wall_indices)} walls")
        print(f"  Start pos: ({start_x}, {start_y}), Current pos: ({mouse_x}, {mouse_y})")
        print(f"  Delta: ({dx}, {dy})")
        
        # Track if any walls were actually moved
        walls_moved = False
            
        # Get the grid size for updating grid coordinates
        grid_size = self.parent_app.uvtt_walls_preview['resolution']['pixels_per_grid']
        if grid_size <= 0:
            grid_size = 70  # Default
            
        # Move each selected wall using initial positions + delta
        for wall_idx in self.parent_app.selected_wall_indices:
            if wall_idx in self.parent_app.initial_wall_positions:
                initial_points = self.parent_app.initial_wall_positions[wall_idx]
                print(f"  Moving wall {wall_idx} with {len(initial_points)} points")
                
                # Update pixel coordinates
                if '_preview_pixels' in self.parent_app.uvtt_walls_preview and wall_idx < len(self.parent_app.uvtt_walls_preview['_preview_pixels']):
                    wall_points = self.parent_app.uvtt_walls_preview['_preview_pixels'][wall_idx]
                    for point_idx, point in enumerate(wall_points):
                        if point_idx < len(initial_points):
                            # Set position based on initial position + total delta
                            new_x = float(initial_points[point_idx]["x"] + dx)
                            new_y = float(initial_points[point_idx]["y"] + dy)
                            point["x"] = new_x
                            point["y"] = new_y
                            walls_moved = True
                            print(f"    Point {point_idx}: ({initial_points[point_idx]['x']}, {initial_points[point_idx]['y']}) -> ({new_x}, {new_y})")
                
                # Update grid coordinates
                if 'line_of_sight' in self.parent_app.uvtt_walls_preview and wall_idx < len(self.parent_app.uvtt_walls_preview['line_of_sight']):
                    grid_points = self.parent_app.uvtt_walls_preview['line_of_sight'][wall_idx]
                    for point_idx, point in enumerate(grid_points):
                        if point_idx < len(initial_points):
                            # Set position based on initial position + total delta converted to grid units
                            point["x"] = float((initial_points[point_idx]["x"] + dx) / grid_size)
                            point["y"] = float((initial_points[point_idx]["y"] + dy) / grid_size)
                    
        print(f"move_selected_walls_absolute: Moved {len(self.parent_app.selected_wall_indices)} walls, result={walls_moved}")
        return walls_moved

    def move_selected_wall_points_absolute(self, mouse_x, mouse_y):
        """Move selected wall points to absolute position based on mouse movement from initial drag start."""
        if (not hasattr(self.parent_app, 'selected_points') or 
            not hasattr(self.parent_app, 'initial_point_positions') or
            not hasattr(self.parent_app, 'multi_wall_drag_start') or
            not self.parent_app.uvtt_walls_preview):
            print("move_selected_wall_points_absolute: Missing required attributes")
            return False
            
        # Calculate total movement from initial drag start
        start_x, start_y = self.parent_app.multi_wall_drag_start
        dx = mouse_x - start_x
        dy = mouse_y - start_y
        
        print(f"move_selected_wall_points_absolute: Moving {len(self.parent_app.selected_points)} points")
        print(f"  Start pos: ({start_x}, {start_y}), Current pos: ({mouse_x}, {mouse_y})")
        print(f"  Delta: ({dx}, {dy})")
        
        # Track if any points were actually moved
        points_moved = False
            
        # Get the grid size for updating grid coordinates
        grid_size = self.parent_app.uvtt_walls_preview['resolution']['pixels_per_grid']
        if grid_size <= 0:
            grid_size = 70  # Default
        
        # Move only the selected points using initial positions + delta
        for wall_idx, point_idx in self.parent_app.selected_points:
            key = (wall_idx, point_idx)
            if key in self.parent_app.initial_point_positions:
                initial_pos = self.parent_app.initial_point_positions[key]
                print(f"  Moving point {point_idx} on wall {wall_idx}")
                
                # Update pixel coordinates
                if ('_preview_pixels' in self.parent_app.uvtt_walls_preview and 
                    wall_idx < len(self.parent_app.uvtt_walls_preview['_preview_pixels']) and
                    point_idx < len(self.parent_app.uvtt_walls_preview['_preview_pixels'][wall_idx])):
                    
                    point = self.parent_app.uvtt_walls_preview['_preview_pixels'][wall_idx][point_idx]
                    # Set position based on initial position + total delta
                    new_x = float(initial_pos["x"] + dx)
                    new_y = float(initial_pos["y"] + dy)
                    point["x"] = new_x
                    point["y"] = new_y
                    points_moved = True
                    print(f"    Point: ({initial_pos['x']}, {initial_pos['y']}) -> ({new_x}, {new_y})")
                    
                    # Update grid coordinates
                    if ('line_of_sight' in self.parent_app.uvtt_walls_preview and 
                        wall_idx < len(self.parent_app.uvtt_walls_preview['line_of_sight']) and
                        point_idx < len(self.parent_app.uvtt_walls_preview['line_of_sight'][wall_idx])):
                        
                        grid_point = self.parent_app.uvtt_walls_preview['line_of_sight'][wall_idx][point_idx]
                        # Set position based on initial position + total delta converted to grid units
                        grid_point["x"] = float((initial_pos["x"] + dx) / grid_size)
                        grid_point["y"] = float((initial_pos["y"] + dy) / grid_size)
                    
        print(f"move_selected_wall_points_absolute: Moved {len(self.parent_app.selected_points)} points, result={points_moved}")
        return points_moved

    # Removed handle_ctrl_hover_in_edit_mode - no longer needed with simplified selection