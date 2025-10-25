import cv2
import numpy as np
from xml.dom import minidom


def export_contours_to_svg(contours, image_shape, output_path, scale_factor=1.0, 
                          simplify_tolerance=0.001, stroke_width=2, stroke_color="black"):
    """
    Export OpenCV contours to SVG format.
    
    Args:
        contours: List of OpenCV contours (numpy arrays)
        image_shape: Tuple of (height, width) of the original image
        output_path: Path where to save the SVG file
        scale_factor: Factor to scale the coordinates (default: 1.0)
        simplify_tolerance: Tolerance for contour simplification (default: 0.001)
        stroke_width: Width of the stroke in SVG (default: 2)
        stroke_color: Color of the stroke (default: "black")
    
    Returns:
        bool: True if export was successful, False otherwise
    """
    try:
        height, width = image_shape[:2]
        
        # Create SVG document
        doc = minidom.Document()
        svg = doc.createElement('svg')
        svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
        svg.setAttribute('width', str(int(width * scale_factor)))
        svg.setAttribute('height', str(int(height * scale_factor)))
        svg.setAttribute('viewBox', f'0 0 {int(width * scale_factor)} {int(height * scale_factor)}')
        doc.appendChild(svg)
        
        # Add a style element for better control
        style = doc.createElement('style')
        style.appendChild(doc.createTextNode(f"""
            .wall-contour {{
                fill: none;
                stroke: {stroke_color};
                stroke-width: {stroke_width};
                stroke-linecap: round;
                stroke-linejoin: round;
            }}
        """))
        svg.appendChild(style)
        
        # Process each contour
        for i, contour in enumerate(contours):
            if contour is None or len(contour) < 2:
                continue
                
            # Simplify contour if tolerance is specified
            if simplify_tolerance > 0:
                epsilon = simplify_tolerance * cv2.arcLength(contour, True)
                contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convert contour to path
            path_data = contour_to_svg_path(contour, scale_factor)
            
            if path_data:
                # Create path element
                path = doc.createElement('path')
                path.setAttribute('d', path_data)
                path.setAttribute('class', 'wall-contour')
                path.setAttribute('id', f'contour-{i}')
                svg.appendChild(path)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doc.toprettyxml(indent='  '))
        
        return True
        
    except Exception as e:
        print(f"Error exporting SVG: {e}")
        return False


def contour_to_svg_path(contour, scale_factor=1.0):
    """
    Convert OpenCV contour to SVG path data.
    
    Args:
        contour: OpenCV contour (numpy array)
        scale_factor: Factor to scale coordinates
    
    Returns:
        str: SVG path data string
    """
    if contour is None or len(contour) < 2:
        return ""
    
    path_data = []
    
    # Reshape contour if needed
    if len(contour.shape) == 3:
        points = contour.reshape(-1, 2)
    else:
        points = contour
    
    if len(points) == 0:
        return ""
    
    # Start with Move command
    first_point = points[0]
    x, y = first_point[0] * scale_factor, first_point[1] * scale_factor
    path_data.append(f"M {x:.2f} {y:.2f}")
    
    # Add Line commands for subsequent points
    for point in points[1:]:
        x, y = point[0] * scale_factor, point[1] * scale_factor
        path_data.append(f"L {x:.2f} {y:.2f}")
    
    # Close the path if it's a closed contour
    if len(points) > 2:
        # Check if the contour is closed (first and last points are close)
        first_point = points[0]
        last_point = points[-1]
        distance = np.linalg.norm(first_point - last_point)
        if distance < 5:  # Threshold for considering it closed
            path_data.append("Z")
    
    return " ".join(path_data)


def export_contours_to_svg_with_layers(contours, image_shape, output_path, 
                                      scale_factor=1.0, simplify_tolerance=0.001):
    """
    Export contours to SVG with different layers based on contour properties.
    
    Args:
        contours: List of OpenCV contours
        image_shape: Tuple of (height, width) of the original image
        output_path: Path where to save the SVG file
        scale_factor: Factor to scale coordinates
        simplify_tolerance: Tolerance for contour simplification
    
    Returns:
        bool: True if export was successful, False otherwise
    """
    try:
        height, width = image_shape[:2]
        
        # Create SVG document
        doc = minidom.Document()
        svg = doc.createElement('svg')
        svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
        svg.setAttribute('width', str(int(width * scale_factor)))
        svg.setAttribute('height', str(int(height * scale_factor)))
        svg.setAttribute('viewBox', f'0 0 {int(width * scale_factor)} {int(height * scale_factor)}')
        doc.appendChild(svg)
        
        # Add styles for different types of walls
        style = doc.createElement('style')
        style.appendChild(doc.createTextNode("""
            .outer-wall {
                fill: none;
                stroke: #2c3e50;
                stroke-width: 3;
                stroke-linecap: round;
                stroke-linejoin: round;
            }
            .inner-wall {
                fill: none;
                stroke: #34495e;
                stroke-width: 2;
                stroke-linecap: round;
                stroke-linejoin: round;
            }
            .small-detail {
                fill: none;
                stroke: #7f8c8d;
                stroke-width: 1;
                stroke-linecap: round;
                stroke-linejoin: round;
            }
        """))
        svg.appendChild(style)
        
        # Separate contours by size/type
        large_contours = []
        medium_contours = []
        small_contours = []
        
        for contour in contours:
            if contour is None or len(contour) < 2:
                continue
                
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area > 5000 or perimeter > 500:
                large_contours.append(contour)
            elif area > 1000 or perimeter > 200:
                medium_contours.append(contour)
            else:
                small_contours.append(contour)
        
        # Create groups for different contour types
        contour_groups = [
            (large_contours, 'outer-wall', 'outer-walls'),
            (medium_contours, 'inner-wall', 'inner-walls'),
            (small_contours, 'small-detail', 'details')
        ]
        
        for contour_list, css_class, group_id in contour_groups:
            if not contour_list:
                continue
                
            # Create group element
            group = doc.createElement('g')
            group.setAttribute('id', group_id)
            group.setAttribute('class', css_class)
            svg.appendChild(group)
            
            # Add paths to group
            for i, contour in enumerate(contour_list):
                # Simplify contour if tolerance is specified
                if simplify_tolerance > 0:
                    epsilon = simplify_tolerance * cv2.arcLength(contour, True)
                    contour = cv2.approxPolyDP(contour, epsilon, True)
                
                # Convert contour to path
                path_data = contour_to_svg_path(contour, scale_factor)
                
                if path_data:
                    # Create path element
                    path = doc.createElement('path')
                    path.setAttribute('d', path_data)
                    path.setAttribute('id', f'{group_id}-{i}')
                    group.appendChild(path)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doc.toprettyxml(indent='  '))
        
        return True
        
    except Exception as e:
        print(f"Error exporting layered SVG: {e}")
        return False
