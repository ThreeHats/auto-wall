import sys
import os
import cv2
import numpy as np
import traceback

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.wall_detection.image_utils import load_image, convert_to_rgb
from src.wall_detection.detector import create_color_mask

def test_webp_loading():
    """
    Test WebP loading functionality with robust error handling.
    This can be used to diagnose WebP loading issues.
    """
    print("Testing WebP file loading...")
    
    # Path to a test WebP file - you'll need to provide a valid path
    test_webp_file = input("Enter path to a WebP file for testing: ")
    
    if not os.path.exists(test_webp_file):
        print(f"Error: File {test_webp_file} does not exist")
        return False
    
    # Ask for a corresponding PNG file for comparison
    test_png_file = input("Enter path to a corresponding PNG file for comparison (or press Enter to skip comparison): ")
    
    print(f"Testing file: {test_webp_file}")
    
    # Test loading with different methods to see what works
    try:
        print("\nMethod 1: Standard OpenCV imread")
        img1 = cv2.imread(test_webp_file)
        if img1 is None:
            print("  Result: FAILED - Could not load WebP with standard imread")
        else:
            print(f"  Result: SUCCESS - Loaded image with shape {img1.shape}")
    except Exception as e:
        print(f"  Result: FAILED - Exception during imread: {e}")
        traceback.print_exc()
    
    try:
        print("\nMethod 2: OpenCV imdecode from bytes")
        with open(test_webp_file, 'rb') as f:
            img_bytes = f.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img2 = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        if img2 is None:
            print("  Result: FAILED - Could not load WebP with imdecode")
        else:
            print(f"  Result: SUCCESS - Loaded image with shape {img2.shape}")
    except Exception as e:
        print(f"  Result: FAILED - Exception during imdecode: {e}")
        traceback.print_exc()
    
    try:
        print("\nMethod 3: Using enhanced load_image function")
        img3 = load_image(test_webp_file)
        if img3 is None:
            print("  Result: FAILED - Could not load WebP with enhanced load_image")
        else:
            print(f"  Result: SUCCESS - Loaded image with shape {img3.shape}")
    except Exception as e:
        print(f"  Result: FAILED - Exception during load_image: {e}")
        traceback.print_exc()
    
    try:
        print("\nMethod 4: Loading + RGB conversion test")
        img4 = load_image(test_webp_file)
        if img4 is None:
            print("  Result: FAILED - Could not load WebP in RGB conversion test")
        else:
            rgb_img = convert_to_rgb(img4)
            print(f"  Result: SUCCESS - Loaded and converted image with shape {rgb_img.shape}")
            
            # Try saving the image to verify it's valid
            test_output = "webp_test_output.png"
            cv2.imwrite(test_output, img4)
            print(f"  Saved test image to {test_output}")
    except Exception as e:
        print(f"  Result: FAILED - Exception during RGB conversion: {e}")
        traceback.print_exc()
    
    # Add color detection test
    try:
        print("\nMethod 5: Color detection test on WebP")
        img5 = load_image(test_webp_file)
        if img5 is None:
            print("  Result: FAILED - Could not load WebP for color detection test")
        else:
            # Get a sample of colors to test
            test_colors = [
                (0, 0, 0),    # Pure black
                (255, 255, 255),  # Pure white
                (128, 128, 128),  # Gray
                (0, 0, 255),      # Red (in BGR)
                (0, 255, 0),      # Green (in BGR)
                (255, 0, 0)       # Blue (in BGR)
            ]
            
            # Test color masks for each test color
            print("  Color detection results for WebP:")
            for color in test_colors:
                mask = create_color_mask(img5, color, threshold=20)
                pixel_count = np.count_nonzero(mask)
                percent = (pixel_count / (mask.shape[0] * mask.shape[1])) * 100
                print(f"    BGR {color}: {pixel_count} pixels ({percent:.2f}%)")
                
                # Save color mask for visualization
                color_filename = f"webp_color_{color[0]}_{color[1]}_{color[2]}.png"
                cv2.imwrite(color_filename, mask)
                print(f"    Saved color mask to {color_filename}")
    except Exception as e:
        print(f"  Result: FAILED - Exception during color detection: {e}")
        traceback.print_exc()
    
    # Compare with PNG if provided
    if test_png_file and os.path.exists(test_png_file):
        try:
            print("\nPerforming PNG comparison test")
            png_img = load_image(test_png_file)
            webp_img = load_image(test_webp_file)
            
            print(f"  PNG image shape: {png_img.shape}")
            print(f"  WebP image shape: {webp_img.shape}")
            
            # Test if sizes match
            if png_img.shape != webp_img.shape:
                print("  WARNING: Image dimensions don't match!")
            
            # Compare color detection between PNG and WebP
            print("\n  Color detection comparison between PNG and WebP:")
            for color in [(0, 0, 0), (255, 255, 255), (128, 128, 128), (0, 0, 255), (0, 255, 0), (255, 0, 0)]:
                png_mask = create_color_mask(png_img, color, threshold=20)
                webp_mask = create_color_mask(webp_img, color, threshold=20)
                
                png_count = np.count_nonzero(png_mask)
                webp_count = np.count_nonzero(webp_mask)
                
                png_percent = (png_count / (png_mask.shape[0] * png_mask.shape[1])) * 100
                webp_percent = (webp_count / (webp_mask.shape[0] * webp_mask.shape[1])) * 100
                
                # Calculate difference between the masks
                diff_mask = cv2.absdiff(png_mask, webp_mask)
                diff_pixels = np.count_nonzero(diff_mask)
                diff_percent = (diff_pixels / (png_mask.shape[0] * png_mask.shape[1])) * 100
                
                print(f"    BGR {color}:")
                print(f"      PNG: {png_count} pixels ({png_percent:.2f}%)")
                print(f"      WebP: {webp_count} pixels ({webp_percent:.2f}%)")
                print(f"      Difference: {diff_pixels} pixels ({diff_percent:.2f}%)")
                
                # Save difference mask
                diff_filename = f"color_diff_{color[0]}_{color[1]}_{color[2]}.png"
                cv2.imwrite(diff_filename, diff_mask)
                print(f"      Saved difference mask to {diff_filename}")
                
            # Create a visual diff between the two images
            if png_img.shape == webp_img.shape:
                # Calculate the absolute difference
                diff_img = cv2.absdiff(png_img, webp_img)
                
                # Enhance difference visibility
                diff_enhanced = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)
                
                # Save the difference images
                cv2.imwrite("format_diff_raw.png", diff_img)
                cv2.imwrite("format_diff_enhanced.png", diff_enhanced)
                print("  Saved visual difference images to format_diff_raw.png and format_diff_enhanced.png")
                
                # Calculate difference statistics
                mean_diff = np.mean(diff_img)
                max_diff = np.max(diff_img)
                print(f"  Mean pixel difference: {mean_diff:.2f}")
                print(f"  Max pixel difference: {max_diff}")
                
                # Check for exact matches
                exact_match = np.all(png_img == webp_img)
                print(f"  Images are pixel-perfect match: {'Yes' if exact_match else 'No'}")
                
                # Check percent of pixels that are identical
                identical_pixels = np.sum(np.all(diff_img == 0, axis=2) if len(diff_img.shape) > 2 else (diff_img == 0))
                total_pixels = png_img.shape[0] * png_img.shape[1]
                identical_percent = (identical_pixels / total_pixels) * 100
                print(f"  Identical pixels: {identical_percent:.2f}%")
        except Exception as e:
            print(f"  PNG comparison failed: {e}")
            traceback.print_exc()
    
    print("\nWebP support test complete.")
    return True

if __name__ == "__main__":
    try:
        test_webp_loading()
    except Exception as e:
        print(f"Unhandled exception: {e}")
        traceback.print_exc()