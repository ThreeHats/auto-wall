import os
import sys
from PIL import Image, ImageDraw, ImageFont

def create_splash_screen(width=600, height=400, 
                        background_color=(40, 40, 45),
                        text="Auto-Wall", 
                        text_color=(255, 255, 255),
                        include_progress_bar=True,
                        logo_path=None,
                        output_path="splash.png"):
    """
    Create a custom splash screen image.
    
    Args:
        width: Width of the splash screen
        height: Height of the splash screen
        background_color: Background color as RGB tuple
        text: Text to display on the splash screen
        text_color: Text color as RGB tuple
        include_progress_bar: Whether to include a progress bar shape
        logo_path: Path to logo image to include (optional)
        output_path: Path where the splash image will be saved
    """
    # Create a new image with the specified background color
    image = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, falling back to default if not available
    try:
        font_size = height // 8
        font = ImageFont.truetype("Arial", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Calculate text position
    text_width, text_height = draw.textsize(text, font=font)
    text_x = (width - text_width) // 2
    text_y = (height - text_height) // 2 - 50  # Higher up to make room for progress bar
    
    # Add logo if provided
    if logo_path and os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path)
            # Resize logo to fit nicely
            logo_max_size = min(width, height) // 3
            logo.thumbnail((logo_max_size, logo_max_size))
            
            # Position logo above text
            logo_x = (width - logo.width) // 2
            logo_y = text_y - logo.height - 20
            
            # Paste the logo
            image.paste(logo, (logo_x, logo_y), logo if logo.mode == 'RGBA' else None)
        except Exception as e:
            print(f"Error loading logo: {e}")
    
    # Draw the title text
    draw.text((text_x, text_y), text, font=font, fill=text_color)
    
    # Add a progress bar shape
    if include_progress_bar:
        bar_height = 10
        bar_width = width * 0.7
        bar_x = (width - bar_width) // 2
        bar_y = height - 50
        
        # Draw empty bar outline
        draw.rectangle(
            [(bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height)],
            outline=text_color
        )
    
    # Save the image
    image.save(output_path)
    print(f"Splash screen created at: {output_path}")

if __name__ == "__main__":
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "splash.png")
    
    # Look for an icon.ico file that could be converted to a logo
    logo_path = os.path.join(script_dir, "icon.ico")
    if not os.path.exists(logo_path):
        logo_path = None
    
    # Create the splash screen
    create_splash_screen(
        text="Auto-Wall",
        output_path=output_path,
        logo_path=logo_path
    )
