import os
import cv2
import urllib.request
import requests
import io
import numpy as np

from PyQt6.QtWidgets import QFileDialog, QMessageBox, QApplication
from src.wall_detection.image_utils import load_image, convert_to_rgb, save_image
from src.wall_detection.detector import detect_walls, draw_walls, merge_contours, split_edge_contours, remove_hatching_lines
from src.wall_detection.mask_editor import blend_image_with_mask
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QColor
from src.utils.performance import PerformanceTimer, debounce, ImageCache, fast_hash

class ImageProcessor:
    def __init__(self, app):
        self.app = app
        # Initialize performance optimizations
        self.detection_cache = ImageCache(max_size=8)
        self.last_detection_params = None
        
        # Create debounced version of update_image
        self.debounced_update = debounce(delay_ms=250)(self._update_image_internal)

    def update_image(self):
        """Update the displayed image based on the current settings (debounced)."""
        # Use debounced version to prevent rapid successive calls
        self.debounced_update()

    def _update_image_internal(self):
        """Internal update method with performance optimizations."""
        if self.app.current_image is None:
            return

        with PerformanceTimer("Full image update"):
            # Get slider values
            blur = self.app.sliders["Smoothing"]['slider'].value()
            
            # Handle special case for blur=1 (no blur) and ensure odd values
            if blur > 1 and blur % 2 == 0:
                blur += 1
        
        canny1 = self.app.sliders["Edge Sensitivity"]['slider'].value()
        canny2 = self.app.sliders["Edge Threshold"]['slider'].value()
        edge_margin = self.app.sliders["Edge Margin"]['slider'].value()
        
        # Get min_merge_distance as a float value
        min_merge_distance = self.app.sliders["Min Merge Distance"]['slider'].value() * 0.1
        
        # Calculate min area based on mode (percentage or pixels)
        min_area_value = self.app.sliders["Min Area"]['slider'].value()
        image_area = self.app.current_image.shape[0] * self.app.current_image.shape[1]
        
        if hasattr(self.app, 'using_pixels_mode') and self.app.using_pixels_mode:
            # Min area is in pixels (1 to 1000)
            min_area = min_area_value
            working_min_area = min_area
            # For display/logging purposes
            min_area_percentage = (min_area / image_area) * 100.0
        else:
            # Min area is a percentage (0.0001% to 1%)
            min_area_percentage = min_area_value * 0.001
            min_area = int(image_area * min_area_percentage / 100.0)
            working_min_area = min_area
        
        # If we're working with a scaled image, the min area needs to be scaled too
        if self.app.scale_factor != 1.0 and not (hasattr(self.app, 'using_pixels_mode') and self.app.using_pixels_mode):
            # Only scale min_area if we're using percentages
            working_min_area = int(min_area * self.app.scale_factor * self.app.scale_factor)
        elif self.app.scale_factor != 1.0:
            # If using pixels mode, scale the pixels to the working image size
            working_min_area = int(min_area * self.app.scale_factor * self.app.scale_factor)
        
        # Working image that we'll pass to the detection function
        processed_image = self.app.current_image.copy()
        
        # Apply hatching removal if enabled
        if self.app.remove_hatching_checkbox.isChecked():
            # Convert QColor to BGR tuple for OpenCV
            hatching_color_bgr = (
                self.app.hatching_color.blue(),
                self.app.hatching_color.green(),
                self.app.hatching_color.red()
            )
            print(f"Removing hatching lines: Color={hatching_color_bgr}, Threshold={self.app.hatching_threshold:.1f}, Width={self.app.hatching_width}")
            
            # Apply the hatching removal
            from src.wall_detection.detector import remove_hatching_lines
            processed_image = remove_hatching_lines(
                processed_image, 
                hatching_color_bgr, 
                self.app.hatching_threshold, 
                self.app.hatching_width
            )
        
        # Set up color detection parameters with per-color thresholds
        wall_colors_with_thresholds = None
        default_threshold = 0
        
        if self.app.color_detection_radio.isChecked() and self.app.wall_colors_list.count() > 0:
            # Extract all colors and thresholds from the list widget
            wall_colors_with_thresholds = []
            for i in range(self.app.wall_colors_list.count()):
                item = self.app.wall_colors_list.item(i)
                color_data = item.data(Qt.ItemDataRole.UserRole)
                color = color_data["color"]
                threshold = color_data["threshold"]
                
                # Convert Qt QColor to OpenCV BGR color and pair with threshold
                bgr_color = (
                    color.blue(),
                    color.green(),
                    color.red()
                )
                wall_colors_with_thresholds.append((bgr_color, threshold))
            
            print(f"Using {len(wall_colors_with_thresholds)} colors for detection with individual thresholds")
          # Debug output of parameters
        if hasattr(self.app, 'using_pixels_mode') and self.app.using_pixels_mode:
            print(f"Parameters: min_area={min_area} pixels (working: {working_min_area}), "
                  f"blur={blur}, canny1={canny1}, canny2={canny2}, edge_margin={edge_margin}")
        else:
            print(f"Parameters: min_area={min_area} (working: {working_min_area}, {min_area_percentage:.4f}% of image), "
                  f"blur={blur}, canny1={canny1}, canny2={canny2}, edge_margin={edge_margin}")

        # Create cache key for detection parameters
        detection_params = {
            'working_min_area': working_min_area,
            'blur': blur,
            'canny1': canny1,
            'canny2': canny2,
            'edge_margin': edge_margin,
            'wall_colors': wall_colors_with_thresholds,
            'default_threshold': default_threshold,
            'merge_contours': self.app.merge_contours.isChecked(),
            'min_merge_distance': min_merge_distance,
            'hatching_enabled': self.app.remove_hatching_checkbox.isChecked(),
            'hatching_params': (self.app.hatching_color.rgb(), self.app.hatching_threshold, self.app.hatching_width) if self.app.remove_hatching_checkbox.isChecked() else None,
            'image_hash': fast_hash(processed_image.tobytes()[:1000])  # Hash first 1KB for speed
        }
        
        cache_key = fast_hash(tuple(sorted(detection_params.items())))
        
        # Check cache first
        cached_result = self.detection_cache.get(cache_key)
        if cached_result is not None and self.last_detection_params == detection_params:
            print("[CACHE] Using cached detection result")
            contours = cached_result
        else:
            # Process the image directly with detect_walls
            with PerformanceTimer("Wall detection"):
                contours = detect_walls(
                    processed_image,
                    min_contour_area=working_min_area,
                    max_contour_area=None,
                    blur_kernel_size=blur,
                    canny_threshold1=canny1,
                    canny_threshold2=canny2,
                    edge_margin=edge_margin,
                    wall_colors=wall_colors_with_thresholds,
                    color_threshold=default_threshold
                )
            
            # Cache the result
            self.detection_cache.put(cache_key, contours.copy() if contours else [])
            self.last_detection_params = detection_params
        
        print(f"Detected {len(contours)} contours before merging")

        # Merge before Min Area if specified
        if self.app.merge_contours.isChecked():
            contours = merge_contours(
                processed_image, 
                contours, 
                min_merge_distance=min_merge_distance
            )
            print(f"After merge before min area: {len(contours)} contours")
        
        # Filter contours by area BEFORE splitting edges
        contours = [c for c in contours if cv2.contourArea(c) >= working_min_area]
        print(f"After min area filter: {len(contours)} contours")

        # Split contours that touch image edges AFTER area filtering, but only if not in color detection mode
        if not self.app.color_detection_radio.isChecked():
            split_contours = split_edge_contours(processed_image, contours)

            # Use a much lower threshold for split contours to keep them all
            # Use absolute minimum value instead of relative to min_area
            min_split_area = 5.0 * (self.app.scale_factor * self.app.scale_factor)  # Scale with image
            filtered_contours = []
            
            # Keep track of how many contours were kept vs filtered
            kept_count = 0
            filtered_count = 0
            
            for contour in split_contours:
                area = cv2.contourArea(contour)
                if area >= min_split_area:
                    filtered_contours.append(contour)
                    kept_count += 1
                else:
                    filtered_count += 1
            
            contours = filtered_contours
            print(f"After edge splitting: kept {kept_count}, filtered {filtered_count} tiny fragments")        # Save the current contours for interactive editing (these are at working resolution)
        self.app.current_contours = contours

        # Ensure contours are not empty
        if not contours:
            print("No contours found after processing.")
            # Display original full-resolution image without contours
            display_image = self.app.original_image.copy() if self.app.original_image is not None else processed_image
            self.app.processed_image = display_image
            # Disable export button when no contours are found
            self.app.export_uvtt_button.setEnabled(False)
        else:
            # Enable export button when contours are successfully detected
            self.app.export_uvtt_button.setEnabled(True)
            # Scale contours up to original resolution for display
            if self.app.scale_factor != 1.0 and self.app.original_image is not None:
                # Scale contours to original resolution
                display_contours = self.app.contour_processor.scale_contours_to_original(contours, self.app.scale_factor)
                # Draw contours on the original full-resolution image
                display_image = draw_walls(self.app.original_image, display_contours)
            else:
                # No scaling needed or no original image available
                display_image = draw_walls(processed_image, contours)
            
            self.app.processed_image = display_image

        # Save the original image for highlighting
        if self.app.processed_image is not None:
            self.app.original_processed_image = self.app.processed_image.copy()
            
        # Clear any existing selection when re-detecting
        self.app.selection_manager.clear_selection()
        # Reset highlighted contour when re-detecting
        self.app.highlighted_contour_index = -1
          # Display the image with grid overlay
        self.app.refresh_display()
        
    def display_image(self, image, preserve_view=False, region=None):
        """Display an image on the image label.
        
        Args:
            image: The image to display (numpy array)
            preserve_view: Whether to preserve the current zoom/pan state
            region: Optional tuple (x, y, width, height) for updating just a specific region
        """
        # Only proceed if the image label exists and has a valid size
        if not hasattr(self.app, 'image_label') or self.app.image_label.width() <= 0 or self.app.image_label.height() <= 0:
            return
        
        # If region is specified and the image label supports region updates, use that method
        if region is not None and hasattr(self.app.image_label, 'update_region'):
            x, y, width, height = region
            # Extract the region from the image
            region_image = image[y:y+height, x:x+width].copy()
            # Convert to RGB for consistent processing
            region_rgb = convert_to_rgb(region_image)
            # Update just that region
            self.app.image_label.update_region(region_rgb, x, y, width, height)
            return
            
        rgb_image = convert_to_rgb(image)
        height, width, channel = rgb_image.shape
        bytes_per_line = channel * width
        q_image = QImage(rgb_image.data.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # If the image label supports zoom and pan, use the new method
        if hasattr(self.app.image_label, 'set_base_pixmap'):
            self.app.image_label.set_base_pixmap(pixmap, preserve_view=preserve_view)
        else:
            # Fallback to original method
            self.app.image_label.setPixmap(pixmap.scaled(self.app.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def open_image(self):
        """Open an image file and prepare scaled versions for processing."""
        file_path, _ = QFileDialog.getOpenFileName(self.app, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if file_path:
            # Get file extension
            file_extension = os.path.splitext(file_path)[1].lower()
            is_webp = file_extension == '.webp'
            
            # Load the original full-resolution image
            print(f"Loading image: {file_path}")
            self.app.original_image = load_image(file_path)
            
            # Verify image loaded correctly
            if self.app.original_image is None:
                QMessageBox.critical(self.app, "Error", f"Failed to load image: {file_path}")
                return
                
            # Log the image dimensions and type info for debugging
            h, w = self.app.original_image.shape[:2]
            channels = self.app.original_image.shape[2] if len(self.app.original_image.shape) > 2 else 1
            print(f"Image loaded: {w}x{h}, {channels} channels, {self.app.original_image.dtype}")
            
            # For WebP files, log whether conversion was applied
            if is_webp:
                print(f"WebP image detected: {file_path}")
            
            # Clear history when loading a new image
            self.app.history.clear()
            self.app.mask_processor.undo_button.setEnabled(False)
            
            # Create a scaled down version for processing if needed
            self.app.current_image, self.app.scale_factor = self.create_working_image(self.app.original_image)
            
            print(f"Image prepared: Original size {self.app.original_image.shape}, Working size {self.app.current_image.shape}, Scale factor {self.app.scale_factor}")
            
            # Reset the mask layer when loading a new image to prevent dimension mismatch
            self.app.mask_layer = None
            self.app.uvtt_walls_preview = None

            self.app.export_panel.set_controls_enabled(True)
            
            # Reset button states when loading a new image
            self.app.export_uvtt_button.setEnabled(False)
            
            # Reset the current overlays and detected contours
            self.app.current_contours = None
            self.app.edges_overlay = None
            
            # Display the original image immediately (centered/fit to window)
            self.display_image(self.app.original_image, preserve_view=False)
            
            # Update the image display (run detection and overlays)
            self.update_image()

    def load_image_from_url(self):
        """Load an image from a URL in the clipboard."""
        # Get clipboard content
        clipboard = QApplication.clipboard()
        clipboard_text = clipboard.text().strip()
        
        # Check if it's a valid URL
        if not clipboard_text:
            QMessageBox.warning(self.app, "Invalid URL", "Clipboard is empty")
            return
            
        try:
            # Check if it's a valid URL
            parsed_url = urllib.parse.urlparse(clipboard_text)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                QMessageBox.warning(self.app, "Invalid URL", f"The clipboard does not contain a valid URL:\n{clipboard_text}")
                return
            
            # Download image from URL
            self.app.setStatusTip(f"Downloading image from {clipboard_text}...")
            response = requests.get(clipboard_text, stream=True, timeout=10)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            # Check if content type is an image
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                QMessageBox.warning(self.app, "Invalid Content", f"The URL does not point to an image (Content-Type: {content_type})")
                return
                
            # Convert response content to an image
            image_data = io.BytesIO(response.content)
            image_array = np.frombuffer(image_data.read(), dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if img is None:
                QMessageBox.warning(self.app, "Loading Error", "Could not decode image from URL")
                return
            
            # Load the image into the application
            self.app.original_image = img
            self.app.current_image, self.app.scale_factor = self.create_working_image(self.app.original_image)
            
            print(f"Image loaded from URL: Original size {self.app.original_image.shape}, Working size {self.app.current_image.shape}, Scale factor {self.app.scale_factor}")
            
            # Reset the mask layer when loading a new image to prevent dimension mismatch
            self.app.mask_layer = None
            self.app.uvtt_walls_preview = None

            self.app.export_panel.set_controls_enabled(True)
            
            # Reset button states when loading a new image
            self.app.export_uvtt_button.setEnabled(False)
            self.app.save_uvtt_button.setEnabled(False)
            self.app.cancel_uvtt_button.setEnabled(False)
            self.app.copy_uvtt_button.setEnabled(False)
            
            # Update the display
            self.app.setStatusTip(f"Image loaded from URL. Size: {img.shape[1]}x{img.shape[0]}")
            
            # Display the original image immediately (centered/fit to window)
            self.display_image(self.app.original_image, preserve_view=False)
            
            # Update the image display (run detection and overlays)
            self.update_image()
            
        except requests.exceptions.RequestException as e:
            QMessageBox.warning(self.app, "Download Error", f"Failed to download the image:\n{str(e)}")
        except Exception as e:
            QMessageBox.warning(self.app, "Error", f"Failed to load image from URL:\n{str(e)}")

    def create_working_image(self, image):
        """Create a working copy of the image, scaling it down if it's too large."""
        # Check if we should use full resolution
        if self.app.high_res_checkbox.isChecked():
            return image.copy(), 1.0
            
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Calculate scale factor if image is larger than the maximum working dimension
        max_dim = max(width, height)
        if max_dim <= self.app.max_working_dimension:
            # Image is already small enough - use as is
            return image.copy(), 1.0
        
        # Calculate scale factor and new dimensions
        scale_factor = self.app.max_working_dimension / max_dim
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Resize the image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return resized, scale_factor

    def save_image(self):
        """Save the processed image at full resolution."""
        if self.app.original_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self.app, "Save Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
            if file_path:
                if self.app.edit_mask_mode_enabled and self.app.mask_layer is not None:
                    # Save image with mask overlay
                    if self.app.scale_factor != 1.0:
                        # Scale mask to original resolution
                        orig_h, orig_w = self.app.original_image.shape[:2]
                        full_res_mask = cv2.resize(self.app.mask_layer, (orig_w, orig_h), 
                                               interpolation=cv2.INTER_NEAREST)
                    else:
                        full_res_mask = self.app.mask_layer
                        
                    # Blend mask with original image
                    result = blend_image_with_mask(self.app.original_image, full_res_mask)
                    
                    # Save the result
                    save_image(result, file_path)
                    print(f"Saved image with mask overlay to {file_path}")
                else:
                    # Normal save with contours
                    if self.app.original_image is not None and self.app.current_contours:
                        # Scale contours back to original image size if needed
                        if self.app.scale_factor != 1.0:
                            full_res_contours = self.app.contour_processor.scale_contours_to_original(self.app.current_contours, self.app.scale_factor)
                        else:
                            full_res_contours = self.app.current_contours
                            
                        # Draw walls on the original high-resolution image
                        high_res_result = draw_walls(self.app.original_image, full_res_contours)
                        
                        # Save the high-resolution result
                        save_image(high_res_result, file_path)
                        print(f"Saved high-resolution image ({self.app.original_image.shape[:2]}) to {file_path}")
                    else:
                        # Just save the current view if no contours
                        save_image(self.app.original_image, file_path)
                        print(f"Saved original image to {file_path}")

    def reload_working_image(self):
        """Reload the working image when resolution setting changes."""
        if self.app.original_image is None:
            return
        
        # Recreate the working image with the current checkbox state
        self.app.current_image, self.app.scale_factor = self.create_working_image(self.app.original_image)
        print(f"Resolution changed: Working size {self.app.current_image.shape}, Scale factor {self.app.scale_factor}")
        
        # Update the image with new resolution
        self.update_image()
