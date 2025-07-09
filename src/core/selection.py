import cv2
from sklearn.cluster import KMeans
from PyQt6.QtGui import QColor

from src.utils.geometry import convert_to_image_coordinates, point_to_line_distance, line_segments_intersect

class SelectionManager:
    def __init__(self, app):
        self.app = app
        self.selected_contour_indices = []
        self.selected_light_indices = []

    def has_selection(self):
        """Check if there are any selected contours or lights."""
        return bool(self.selected_contour_indices or self.selected_light_indices)

    def get_selected_lights(self):
        """Get the indices of selected lights."""
        return self.selected_light_indices

    def clear_selection(self):
        """Clear the current selection of contours and lights."""
        self.selected_contour_indices = []
        self.selected_light_indices = []
        
        # If in UVTT preview mode, redraw to remove selection highlights
        if hasattr(self.app, 'uvtt_preview_active') and self.app.uvtt_preview_active:
            self.app.export_panel.display_uvtt_preview()
            return

        # Original logic for non-preview mode
        self.app.selecting = False
        self.app.selection_start_img = None
        self.app.selection_current_img = None
        
        self.app.selecting_colors = False
        self.app.color_selection_start = None
        self.app.color_selection_current = None
        
        if self.app.processed_image is not None and self.app.original_processed_image is not None:
            self.app.processed_image = self.app.original_processed_image.copy()
            self.app.refresh_display()

    def start_selection(self, x, y):
        """Start a selection rectangle at the given coordinates."""
        # Convert to image coordinates
        img_x, img_y = convert_to_image_coordinates(self.app, x, y)
        
        if img_x is None or img_y is None:
            return
            
        if self.app.deletion_mode_enabled:
            # Check if click is on a contour edge
            min_distance = float('inf')
            found_contour_index = -1
            
            for i, contour in enumerate(self.app.current_contours):
                contour_points = contour.reshape(-1, 2)
                
                for j in range(len(contour_points)):
                    p1 = contour_points[j]
                    p2 = contour_points[(j + 1) % len(contour_points)]
                    distance = point_to_line_distance(img_x, img_y, p1[0], p1[1], p2[0], p2[1])
                    
                    # If point is close enough to a line segment
                    if distance < 5 and distance < min_distance:  # Threshold for line detection (pixels)
                        min_distance = distance
                        found_contour_index = i
            
            # If click is on a contour edge, handle as single click
            if found_contour_index != -1:
                self.handle_deletion_click(x, y)
                return
                
            # Otherwise, start a selection
            self.app.selecting = True
            self.app.selection_start_img = (img_x, img_y)
            self.app.selection_current_img = (img_x, img_y)
            self.app.selected_contour_indices = []
            
        elif self.app.color_selection_mode_enabled:
            # Start color selection rectangle
            self.app.selecting_colors = True
            self.app.color_selection_start = (img_x, img_y)
            self.app.color_selection_current = (img_x, img_y)
        elif self.app.thin_mode_enabled:
            # Check if click is on a contour edge
            min_distance = float('inf')
            found_contour_index = -1
            
            for i, contour in enumerate(self.app.current_contours):
                contour_points = contour.reshape(-1, 2)
                
                for j in range(len(contour_points)):
                    p1 = contour_points[j]
                    p2 = contour_points[(j + 1) % len(contour_points)]
                    distance = point_to_line_distance(img_x, img_y, p1[0], p1[1], p2[0], p2[1])
                    
                    # If point is close enough to a line segment
                    if distance < 5 and distance < min_distance:  # Threshold for line detection (pixels)
                        min_distance = distance
                        found_contour_index = i
            
            # If click is on a contour edge, handle as single click
            if found_contour_index != -1:
                self.handle_thinning_click(x, y)
                return
                
            # Otherwise, start a selection for thinning multiple contours
            self.app.selecting = True
            self.app.selection_start_img = (img_x, img_y)
            self.app.selection_current_img = (img_x, img_y)
            self.app.selected_contour_indices = []

    def update_selection(self, x, y):
        """Update the current selection rectangle to the given coordinates."""
        # Convert to image coordinates
        img_x, img_y = convert_to_image_coordinates(self.app, x, y)
        
        if img_x is None or img_y is None:
            return
            
        if self.app.deletion_mode_enabled and self.app.selecting:
            self.app.selection_current_img = (img_x, img_y)
            self.update_selection_display()
            
        elif self.app.color_selection_mode_enabled and self.app.selecting_colors:
            self.app.color_selection_current = (img_x, img_y)
            self.update_color_selection_display()
        elif self.app.thin_mode_enabled and self.app.selecting:
            self.app.selection_current_img = (img_x, img_y)
            self.update_selection_display()

    def update_selection_display(self):
        """Update the display with the selection rectangle and highlighted contours."""
        if not self.app.selecting or self.app.original_processed_image is None:
            return
            
        # Start with the original image
        self.app.processed_image = self.app.original_processed_image.copy()
        
        # Calculate selection rectangle
        x1 = min(self.app.selection_start_img[0], self.app.selection_current_img[0])
        y1 = min(self.app.selection_start_img[1], self.app.selection_current_img[1])
        x2 = max(self.app.selection_start_img[0], self.app.selection_current_img[0])
        y2 = max(self.app.selection_start_img[1], self.app.selection_current_img[1])
        
        # Draw semi-transparent selection rectangle
        overlay = self.app.processed_image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 100, 200), 2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 100, 200), -1)
        cv2.addWeighted(overlay, 0.3, self.app.processed_image, 0.7, 0, self.app.processed_image)        # Find and highlight contours within the selection - only using edge detection
        self.app.selected_contour_indices = []
        
        # Get contours at display resolution for accurate selection highlighting
        if self.app.scale_factor != 1.0 and self.app.original_image is not None:
            # Scale contours to display resolution for accurate selection highlighting
            display_contours = self.app.contour_processor.scale_contours_to_original(self.app.current_contours, self.app.scale_factor)
        else:
            display_contours = self.app.current_contours
        
        for i, contour in enumerate(display_contours):
            contour_points = contour.reshape(-1, 2)
            for j in range(len(contour_points)):
                p1 = contour_points[j]
                p2 = contour_points[(j + 1) % len(contour_points)]
                
                # Check if any part of this line segment is in the selection rectangle
                # First check if either endpoint is in the rectangle
                if ((x1 <= p1[0] <= x2 and y1 <= p1[1] <= y2) or 
                    (x1 <= p2[0] <= x2 and y1 <= p2[1] <= y2)):
                    self.app.selected_contour_indices.append(i)
                    # Highlight with different colors based on mode
                    highlight_color = (0, 0, 255) if self.app.deletion_mode_enabled else (255, 0, 255)  # Red for delete, Magenta for thin
                    cv2.drawContours(self.app.processed_image, [contour], 0, highlight_color, 2)
                    break
                
                # If neither endpoint is in the rectangle, check if the line intersects the rectangle
                # by checking against all four edges of the rectangle
                rect_edges = [
                    ((x1, y1), (x2, y1)),  # Top edge
                    ((x2, y1), (x2, y2)),  # Right edge
                    ((x2, y2), (x1, y2)),  # Bottom edge
                    ((x1, y2), (x1, y1))   # Left edge
                ]
                
                for rect_p1, rect_p2 in rect_edges:
                    if line_segments_intersect(self.app, p1[0], p1[1], p2[0], p2[1], 
                                                  rect_p1[0], rect_p1[1], rect_p2[0], rect_p2[1]):
                        self.app.selected_contour_indices.append(i)
                        # Highlight with different colors based on mode
                        highlight_color = (0, 0, 255) if self.app.deletion_mode_enabled else (255, 0, 255)
                        cv2.drawContours(self.app.processed_image, [contour], 0, highlight_color, 2)
                        break
                # If we've already added this contour, no need to check more line segments
                if i in self.app.selected_contour_indices:
                    break
                    
        # Display the updated image
        self.app.refresh_display()

    def update_color_selection_display(self):
        """Update the display with the color selection rectangle."""
        if not self.app.selecting_colors or self.app.original_processed_image is None:
            return
            
        # Start with the original image
        self.app.processed_image = self.app.original_processed_image.copy()
        
        # Calculate selection rectangle
        x1 = min(self.app.color_selection_start[0], self.app.color_selection_current[0])
        y1 = min(self.app.color_selection_start[1], self.app.color_selection_current[1])
        x2 = max(self.app.color_selection_start[0], self.app.color_selection_current[0])
        y2 = max(self.app.color_selection_start[1], self.app.color_selection_current[1])
        
        # Draw semi-transparent selection rectangle
        overlay = self.app.processed_image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), -1)
        cv2.addWeighted(overlay, 0.3, self.app.processed_image, 0.7, 0, self.app.processed_image)
                    
        # Display the updated image
        self.app.refresh_display()

    def end_selection(self, x, y):
        """Complete the selection and process it according to the current mode."""        # Convert to image coordinates
        img_x, img_y = convert_to_image_coordinates(self.app, x, y)
        
        if img_x is None or img_y is None:
            self.clear_selection()
            return
        
        if self.app.deletion_mode_enabled and self.app.selecting:
            self.app.selection_current_img = (img_x, img_y)
            
            # Calculate selection rectangle
            x1 = min(self.app.selection_start_img[0], self.app.selection_current_img[0])
            y1 = min(self.app.selection_start_img[1], self.app.selection_current_img[1])
            x2 = max(self.app.selection_start_img[0], self.app.selection_current_img[0])
            y2 = max(self.app.selection_start_img[1], self.app.selection_current_img[1])
            
            # Convert rectangle to working coordinates for contour matching if needed
            # The rectangle coordinates are in display image coordinates (full resolution)
            # but contours are in working resolution, so scale down if necessary
            working_x1, working_y1, working_x2, working_y2 = x1, y1, x2, y2
            if self.app.scale_factor != 1.0 and self.app.original_image is not None:
                working_x1 = int(x1 * self.app.scale_factor)
                working_y1 = int(y1 * self.app.scale_factor)
                working_x2 = int(x2 * self.app.scale_factor)
                working_y2 = int(y2 * self.app.scale_factor)
            
            # Find contours within the selection
            self.app.selected_contour_indices = []
            
            for i, contour in enumerate(self.app.current_contours):
                # Check if contour is at least partially within selection rectangle
                for point in contour:
                    px, py = point[0]
                    if working_x1 <= px <= working_x2 and working_y1 <= py <= working_y2:
                        self.app.selected_contour_indices.append(i)
                        break
            
            # If we have selected contours, delete them immediately
            if self.app.selected_contour_indices:
                self.app.contour_processor.delete_selected_contours()
            else:
                # If no contours were selected, just clear the selection
                self.clear_selection()
                
        elif self.app.color_selection_mode_enabled and self.app.selecting_colors:
            self.app.color_selection_current = (img_x, img_y)
            
            # Calculate selection rectangle
            x1 = min(self.app.color_selection_start[0], self.app.color_selection_current[0])
            y1 = min(self.app.color_selection_start[1], self.app.color_selection_current[1])
            x2 = max(self.app.color_selection_start[0], self.app.color_selection_current[0])
            y2 = max(self.app.color_selection_start[1], self.app.color_selection_current[1])
            
            # Make sure we have a valid selection area
            if x1 < x2 and y1 < y2 and x2 - x1 > 5 and y2 - y1 > 5:
                # Extract colors from the selected area
                self.extract_colors_from_selection(x1, y1, x2, y2)
            else:
                print("Selection area too small")            # Clear the selection
            self.clear_selection()
                
        elif self.app.thin_mode_enabled and self.app.selecting:
            self.app.selection_current_img = (img_x, img_y)
            
            # Calculate selection rectangle
            x1 = min(self.app.selection_start_img[0], self.app.selection_current_img[0])
            y1 = min(self.app.selection_start_img[1], self.app.selection_current_img[1])
            x2 = max(self.app.selection_start_img[0], self.app.selection_current_img[0])
            y2 = max(self.app.selection_start_img[1], self.app.selection_current_img[1])
            
            # Convert rectangle to working coordinates for contour matching if needed
            # The rectangle coordinates are in display image coordinates (full resolution)
            # but contours are in working resolution, so scale down if necessary
            working_x1, working_y1, working_x2, working_y2 = x1, y1, x2, y2
            if self.app.scale_factor != 1.0 and self.app.original_image is not None:
                working_x1 = int(x1 * self.app.scale_factor)
                working_y1 = int(y1 * self.app.scale_factor)
                working_x2 = int(x2 * self.app.scale_factor)
                working_y2 = int(y2 * self.app.scale_factor)
            
            # Find contours within the selection
            self.app.selected_contour_indices = []
            
            for i, contour in enumerate(self.app.current_contours):
                # Check if contour is at least partially within selection rectangle
                for point in contour:
                    px, py = point[0]
                    if working_x1 <= px <= working_x2 and working_y1 <= py <= working_y2:
                        self.app.selected_contour_indices.append(i)
                        break
            
            # If we have selected contours, thin them
            if self.app.selected_contour_indices:
                self.app.contour_processor.thin_selected_contours()
            else:
                # If no contours were selected, just clear the selection
                self.clear_selection()

    def extract_colors_from_selection(self, x1, y1, x2, y2):
        """Extract dominant colors from the selected region."""
        if self.app.current_image is None:
            return
            
        # Extract the selected region from the image
        region = self.app.current_image[y1:y2, x1:x2]
        
        if region.size == 0:
            print("Selected region is empty")
            return
            
        # Reshape the region for clustering
        pixels = region.reshape(-1, 3)
        
        # Get the number of colors to extract
        num_colors = self.app.color_count_spinner.value()
        
        # Use K-means clustering to find the dominant colors
        kmeans = KMeans(n_clusters=num_colors, n_init=10)
        kmeans.fit(pixels)
        
        # Get the colors (cluster centers)
        colors = kmeans.cluster_centers_
        
        # Add each color to the color list
        for color in colors:
            bgr_color = color.astype(int)
            qt_color = QColor(bgr_color[2], bgr_color[1], bgr_color[0])  # Convert BGR to RGB
            
            # Add the color with a threshold of 0 (exact match) initially
            item = self.app.detection_panel.add_wall_color_to_list(qt_color, 0)
            
            # Select the new color
            self.app.wall_colors_list.setCurrentItem(item)
            self.app.detection_panel.select_color(item)
        
        print(f"Extracted {num_colors} colors from selected region")
        
        # Update the image with the new colors
        self.app.image_processor.update_image()

    def handle_deletion_click(self, x, y):
        """Handle clicks for deletion mode."""
        if not self.app.current_contours or self.app.current_image is None:
            return
            
        # Convert display coordinates to image coordinates
        img_x, img_y = convert_to_image_coordinates(self.app, x, y)
        
        # Check if coordinates are valid
        if img_x is None or img_y is None:
            return        # Clear any existing selection when handling a single click
        self.app.selection_manager.clear_selection()
        
        # Save state before deleting
        self.app.mask_processor.save_state()
        
        # Use the highlighted contour if available
        if self.app.highlighted_contour_index != -1:
            print(f"Deleting highlighted contour {self.app.highlighted_contour_index}")
            self.app.current_contours.pop(self.app.highlighted_contour_index)
            self.app.highlighted_contour_index = -1  # Reset highlight
            self.app.contour_processor.update_display_from_contours()
            return
        
        # Convert to working coordinates for contour matching if needed
        # img_x, img_y are in display image coordinates (full resolution)
        # but contours are in working resolution, so scale down if necessary
        working_x, working_y = img_x, img_y
        if self.app.scale_factor != 1.0 and self.app.original_image is not None:
            working_x = int(img_x * self.app.scale_factor)
            working_y = int(img_y * self.app.scale_factor)
        
        # Find contours where the click is on or near an edge
        min_distance = float('inf')
        closest_contour_index = -1
        
        # Check if click is on or near a contour edge
        for i, contour in enumerate(self.app.current_contours):
            contour_points = contour.reshape(-1, 2)
            
            for j in range(len(contour_points)):
                p1 = contour_points[j]
                p2 = contour_points[(j + 1) % len(contour_points)]
                distance = point_to_line_distance(working_x, working_y, p1[0], p1[1], p2[0], p2[1])
                
                # If point is close enough to a line segment
                if distance < 5 and distance < min_distance:  # Threshold for line detection (pixels)
                    min_distance = distance
                    closest_contour_index = i
        
        # If click is on or near an edge, delete that contour
        if closest_contour_index != -1:
            print(f"Deleting contour {closest_contour_index} (edge clicked)")
            self.app.current_contours.pop(closest_contour_index)
            self.app.contour_processor.update_display_from_contours()
            return

    def handle_thinning_click(self, x, y):
        """Handle clicks for thinning mode."""
        if not self.app.current_contours or self.app.current_image is None:
            return
            
        # Convert display coordinates to image coordinates
        img_x, img_y = convert_to_image_coordinates(self.app, x, y)
        
        # Check if coordinates are valid
        if img_x is None or img_y is None:
            return        # Clear any existing selection when handling a single click
        self.app.selection_manager.clear_selection()
        
        # Save state before modifying
        self.app.mask_processor.save_state()
        
        # Use the highlighted contour if available
        if self.app.highlighted_contour_index != -1:
            print(f"Thinning highlighted contour {self.app.highlighted_contour_index}")
            contour = self.app.current_contours[self.app.highlighted_contour_index]
            thinned_contour = self.app.contour_processor.thin_selected_contour(contour)
            self.app.current_contours[self.app.highlighted_contour_index] = thinned_contour
            self.app.highlighted_contour_index = -1  # Reset highlight
            self.app.contour_processor.update_display_from_contours()
            return
            
        # Convert to working coordinates for contour matching if needed
        # img_x, img_y are in display image coordinates (full resolution)
        # but contours are in working resolution, so scale down if necessary
        working_x, working_y = img_x, img_y
        if self.app.scale_factor != 1.0 and self.app.original_image is not None:
            working_x = int(img_x * self.app.scale_factor)
            working_y = int(img_y * self.app.scale_factor)
            
        # Find contours where the click is on or near an edge
        min_distance = float('inf')
        closest_contour_index = -1
        
        # Check if click is on or near a contour edge
        for i, contour in enumerate(self.app.current_contours):
            contour_points = contour.reshape(-1, 2)
            
            for j in range(len(contour_points)):
                p1 = contour_points[j]
                p2 = contour_points[(j + 1) % len(contour_points)]
                distance = point_to_line_distance(working_x, working_y, p1[0], p1[1], p2[0], p2[1])
                
                # If point is close enough to a line segment
                if distance < 5 and distance < min_distance:  # Threshold for line detection (pixels)
                    min_distance = distance
                    closest_contour_index = i
        
        # If click is on or near an edge, thin that contour
        if closest_contour_index != -1:
            print(f"Thinning contour {closest_contour_index} (edge clicked)")
            contour = self.app.current_contours[closest_contour_index]
            thinned_contour = self.app.contour_processor.thin_selected_contour(contour)
            self.app.current_contours[closest_contour_index] = thinned_contour
            self.app.contour_processor.update_display_from_contours()
            return
