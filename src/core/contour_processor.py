import cv2
import numpy as np

from src.wall_detection.detector import draw_walls
from src.wall_detection.mask_editor import thin_contour, thicken_contour

class ContourProcessor:
    def __init__(self, app):
        self.app = app
    

    def scale_contours_to_original(self, contours, scale_factor):
        """Scale contours back to the original image size."""
        if scale_factor == 1.0:
            # No scaling needed
            return contours
            
        scaled_contours = []
        for contour in contours:
            # Create a scaled copy of the contour
            scaled_contour = contour.copy().astype(np.float32)
            scaled_contour /= scale_factor  # Scale coordinates
            scaled_contours.append(scaled_contour.astype(np.int32))
        
        return scaled_contours

    def scale_contours_to_working(self, contours, scale_factor):
        """Scale contours to the working image size."""
        if scale_factor == 1.0:
            # No scaling needed
            return contours
            
        scaled_contours = []
        for contour in contours:
            # Create a scaled copy of the contour
            scaled_contour = contour.copy().astype(np.float32)
            scaled_contour *= scale_factor  # Scale coordinates
            scaled_contours.append(scaled_contour.astype(np.int32))
        
        return scaled_contours

    def update_display_from_contours(self):
        """Update the display with the current contours."""
        if self.app.current_image is not None and self.app.current_contours:
            # Use bg-removed preview as base image when active
            base_image = self.app.image_processor._get_display_base_image(self.app.current_image)

            # Handle scaling properly - display on full-resolution image if available
            if self.app.scale_factor != 1.0 and self.app.original_image is not None:
                # Scale contours to original resolution for display
                display_contours = self.scale_contours_to_original(self.app.current_contours, self.app.scale_factor)
                self.app.processed_image = draw_walls(base_image, display_contours)
            else:
                self.app.processed_image = draw_walls(base_image, self.app.current_contours)
            
            # Re-draw lights if they exist and light detection is enabled
            if (hasattr(self.app, 'current_lights') and self.app.current_lights and 
                hasattr(self.app, 'enable_light_detection') and self.app.enable_light_detection.isChecked()):
                from src.wall_detection.light_detector import draw_lights_on_image
                
                # Scale lights to match the display image if necessary
                lights_to_draw = self.app.current_lights.copy()
                if self.app.scale_factor != 1.0 and self.app.original_image is not None:
                    # Scale light positions to match the original image size
                    for light in lights_to_draw:
                        if "position" in light:
                            # Convert from grid coordinates back to pixels in working image
                            pixel_x = light["position"]["x"] * 70.0  # Convert grid to pixels
                            pixel_y = light["position"]["y"] * 70.0
                            
                            # Scale to original image size
                            scaled_x = pixel_x * self.app.scale_factor
                            scaled_y = pixel_y * self.app.scale_factor
                            
                            # Convert back to grid coordinates for drawing
                            light["position"]["x"] = scaled_x / 70.0
                            light["position"]["y"] = scaled_y / 70.0
                
                # Draw the lights on the processed image
                self.app.processed_image = draw_lights_on_image(
                    self.app.processed_image,
                    lights_to_draw,
                    grid_size=70.0,
                    show_range=False,  # Don't show range circles in detection mode
                    alpha=0.8  # More visible in detection mode
                )
            
            self.app.original_processed_image = self.app.processed_image.copy()
            self.app.refresh_display()
        elif self.app.current_image is not None:
            # Use bg-removed preview as base when active, otherwise original
            display_image = self.app.image_processor._get_display_base_image(self.app.current_image)
            self.app.processed_image = display_image
            
            # Re-draw lights even when no contours exist
            if (hasattr(self.app, 'current_lights') and self.app.current_lights and 
                hasattr(self.app, 'enable_light_detection') and self.app.enable_light_detection.isChecked()):
                from src.wall_detection.light_detector import draw_lights_on_image
                
                # Scale lights to match the display image if necessary
                lights_to_draw = self.app.current_lights.copy()
                if self.app.scale_factor != 1.0 and self.app.original_image is not None:
                    # Scale light positions to match the original image size
                    for light in lights_to_draw:
                        if "position" in light:
                            # Convert from grid coordinates back to pixels in working image
                            pixel_x = light["position"]["x"] * 70.0  # Convert grid to pixels
                            pixel_y = light["position"]["y"] * 70.0
                            
                            # Scale to original image size
                            scaled_x = pixel_x * self.app.scale_factor
                            scaled_y = pixel_y * self.app.scale_factor
                            
                            # Convert back to grid coordinates for drawing
                            light["position"]["x"] = scaled_x / 70.0
                            light["position"]["y"] = scaled_y / 70.0
                
                # Draw the lights on the processed image
                self.app.processed_image = draw_lights_on_image(
                    self.app.processed_image,
                    lights_to_draw,
                    grid_size=70.0,
                    show_range=False,  # Don't show range circles in detection mode
                    alpha=0.8  # More visible in detection mode
                )
            
            self.app.original_processed_image = self.app.processed_image.copy()
            self.app.refresh_display()

    def delete_selected_contours(self):
        """Delete the selected contours from the current image."""
        if not self.app.selected_contour_indices:
            return
        
        # Save state before modifying
        self.app.mask_processor.save_state()
        
        # Delete selected contours
        for index in sorted(self.app.selected_contour_indices, reverse=True):
            if 0 <= index < len(self.app.current_contours):
                print(f"Deleting contour {index} (area: {cv2.contourArea(self.app.current_contours[index])})")
                self.app.current_contours.pop(index)
        
        # Clear selection and update display
        self.app.selection_manager.clear_selection()
        self.update_display_from_contours()

    def thin_selected_contour(self, contour):
        """Thin a single contour using morphological thinning.

        Returns a list of contours (thinning may split one contour into multiple).
        """
        # Create a mask for the contour
        mask = np.zeros(self.app.current_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Apply the thinning operation using the imported function
        # Pass the current target width and max iterations settings
        thinned_contours = thin_contour(mask, target_width=self.app.target_width, max_iterations=self.app.max_iterations)

        if thinned_contours is not None:
            return thinned_contours
        else:
            return [contour]

    def thin_selected_contours(self):
        """Thin the selected contours."""
        if not self.app.selected_contour_indices:
            return

        # Save state before modifying
        self.app.mask_processor.save_state()

        # Build new contour list, replacing thinned contours (which may split into multiple)
        new_contours = []
        thinned_indices = set(self.app.selected_contour_indices)

        for idx, contour in enumerate(self.app.current_contours):
            if idx in thinned_indices:
                new_contours.extend(self.thin_selected_contour(contour))
            else:
                new_contours.append(contour)

        self.app.current_contours = new_contours

        # Clear selection and update display
        self.app.selection_manager.clear_selection()
        self.update_display_from_contours()

    def thicken_selected_contour(self, contour):
        """Thicken a single contour using morphological dilation.

        Returns a list of contours (dilation may merge nearby regions).
        """
        mask = np.zeros(self.app.current_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        thickened_contours = thicken_contour(mask, target_width=self.app.target_width, max_iterations=self.app.max_iterations)

        if thickened_contours is not None:
            return thickened_contours
        else:
            return [contour]

    def thicken_selected_contours(self):
        """Thicken the selected contours."""
        if not self.app.selected_contour_indices:
            return

        self.app.mask_processor.save_state()

        new_contours = []
        thickened_indices = set(self.app.selected_contour_indices)

        for idx, contour in enumerate(self.app.current_contours):
            if idx in thickened_indices:
                new_contours.extend(self.thicken_selected_contour(contour))
            else:
                new_contours.append(contour)

        self.app.current_contours = new_contours

        self.app.selection_manager.clear_selection()
        self.update_display_from_contours()

