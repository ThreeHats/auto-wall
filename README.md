# auto-wall Project

## Overview
The auto-wall project is a tool designed for detecting and drawing walls on battle maps for tabletop games. It includes both manual drawing capabilities and automated wall detection from images.

## Project Structure
```
auto-wall
├── src
│   ├── wall_detection
│   │   ├── __init__.py
│   │   ├── detector.py       # Core wall detection algorithms
│   │   └── image_utils.py    # Image preprocessing functions
│   ├── gui
│   │   ├── __init__.py
│   │   ├── app.py            # Main application window
│   │   └── drawing.py        # Drawing utilities
│   └── utils
│       ├── __init__.py
│       └── file_handlers.py  # Save/load functionality
├── scripts
│   ├── draw_walls.py         # Original manual drawing script
│   └── detect_walls.py       # Auto detection script
├── tests
│   ├── __init__.py
│   ├── test_detector.py
│   └── test_image_utils.py
├── data
│   ├── input                 # For storing input images
│   └── output                # For storing processed maps
├── requirements.txt
└── README.md
```

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/auto-wall.git
   ```
2. Navigate to the project directory:
   ```
   cd auto-wall
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
- To run the manual drawing script:
  ```
  python scripts/draw_walls.py
  ```
- To run the automated wall detection script:
  ```
  python scripts/detect_walls.py
  ```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.