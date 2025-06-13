import math

def point_to_line_distance(self, x, y, x1, y1, x2, y2):
    """Calculate the distance from point (x,y) to line segment (x1,y1)-(x2,y2)."""
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

def line_segments_intersect(self, x1, y1, x2, y2, x3, y3, x4, y4):
    """Check if two line segments (x1,y1)-(x2,y2) and (x3,y3)-(x4,y4) intersect."""
    # Calculate the direction vectors
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3
    
    # Calculate the determinant
    d = dx1 * dy2 - dy1 * dx2
    
    # If determinant is zero, lines are parallel and don't intersect
    if d == 0:
        return False
        
    # Calculate the parameters for the intersection point
    t1 = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / d
    t2 = ((x1 - x3) * dy1 - (y1 - y3) * dx1) / (-d)
    
    # Check if the intersection point lies on both line segments
    return 0 <= t1 <= 1 and 0 <= t2 <= 1

def convert_to_image_coordinates(self, display_x, display_y):
    """Convert display coordinates to image coordinates."""
    if self.current_image is None:
        return None, None
        
    img_height, img_width = self.current_image.shape[:2]
    display_width = self.image_label.width()
    display_height = self.image_label.height()
    
    # Calculate scaling and offset (centered in the label)
    scale_x = display_width / img_width
    scale_y = display_height / img_height
    scale = min(scale_x, scale_y)
    
    offset_x = (display_width - img_width * scale) / 2
    offset_y = (display_height - img_height * scale) / 2
    
    # Convert to image coordinates
    img_x = int((display_x - offset_x) / scale)
    img_y = int((display_y - offset_y) / scale)
    
    # Check if click is out of bounds
    if img_x < 0 or img_x >= img_width or img_y < 0 or img_y >= img_height:
        return None, None
        
    return img_x, img_y

