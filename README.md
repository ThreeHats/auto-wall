# Auto-Wall

![Auto-Wall](resources/icon.ico)

**Auto-Wall** is a powerful tool for processing battle maps for virtual tabletop games. It automatically detects walls, obstacles, and lights in your maps and converts them into VTT-compatible wall data.

Watch the video:
[![video link](https://img.youtube.com/vi/gqkEIWwuJX4/maxresdefault.jpg)](https://youtu.be/gqkEIWwuJX4)

[![GitHub Release](https://img.shields.io/github/v/release/ThreeHats/auto-wall?style=flat&label=Latest)](https://github.com/ThreeHats/auto-wall/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Discord server](https://discord.gg/HUzEnZy8uJ)

## What's New

- **Switched to UVTT format:** Full compatibility with FoundryVTT, Roll20, Arkenforge, Fantasy Grounds, and more.
- **Edit Walls Mode:** Added a new mode for editing walls, doors, and lights.
- **Doors:** Doors can now be manually drawn in the wall editing mode.
- **Lights:** Lights have their own detection settings and can be added to your maps.
- **Improved UI:** Button names have been updated for clarity.
- **Drawing Improvements:** Hold `Control` while drawing walls or doors to create a continuous line.

### Importing UVTT Files

- **FoundryVTT:** Use the [Universal Battlemap Importer](https://foundryvtt.com/packages/uvtt-importer) by Moo Man.
- **Roll20:** Use the [UniversalVTTImporter](https://github.com/shdwjk/UniversalVTTImporter) by The Aaron.
- **Arkenforge & Fantasy Grounds:** Both platforms support UVTT files natively—simply import your exported file.

---

## Features

### Automated Wall, Door, and Light Detection
- **Edge Detection:** Automatically finds walls using edge detection algorithms
- **Color-Based Detection:** Extract walls based on specific colors
- **Adjustable Detection Parameters:** Fine-tune detection sensitivity and results
- **High-Resolution Processing:** Option to process at full resolution for more accurate results
- **Light Detection:** Dedicated settings for detecting lights (currently not saved in presets)
- **Door Drawing:** Manually add doors in edit mode

### Editing and Refinement
- **Interactive Deletion:** Click to remove unwanted walls
- **Color Picking:** Extract colors directly from the map for better detection
- **Mask Editing:** Draw or erase walls manually with various drawing tools
- **Wall Thinning:** Automatically reduce wall thickness for cleaner results
- **Contour Merging:** Connect and simplify wall segments
- **Edit Walls Mode:** Draw, erase, and modify walls, doors, and lights

### Universal VTT (UVTT) Integration

- **Save as UVTT:** Export maps and wall data in the Universal VTT (UVTT) format, compatible with many virtual tabletops.
- **Workflow:** Export your map as UVTT from Auto-Wall, then follow the instructions for your VTT of choice to import walls, doors, lights, and map images.

## Installation

### Windows Executable
1. Download the latest release from [autowallvtt.com](autowallvtt.com) or the [Releases page](https://github.com/ThreeHats/auto-wall/releases)
2. Run `Auto-Wall.exe`

### From Source
1. Clone the repository:
   ```
   git clone https://github.com/ThreeHats/auto-wall.git
   ```

2. Install dependencies:
   ```
   cd auto-wall
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python auto_wall.py
   ```

## Quick Start Guide

1. **Open an image:** Click "Open Image" or "Load from URL" to import your battle map
2. **Navigate the view:** Use scroll wheel to zoom, right-click and drag to pan
3. **Pick detection mode:**
   - For maps with clear lines, use edge detection (default)
   - For maps with distinct colors for the walls or background, enable "Color Detection" and select colors
4. **Fine-tune results:** Use the sliders to adjust detection sensitivity
5. **Clean up walls:** Switch to "Edit Walls" mode to delete, draw, or modify walls, doors, and lights
6. **Manual drawing:** Click "Start Drawing", and use the drawing tools to make any final adjustments
5. **Generate walls:** Click "Generate Walls", adjust your settings, and click "Ok"
5. **Clean up walls:** After generating walls, you can delete, draw, or modify walls, doors, and lights
7. **Export to UVTT:** Click "Save File" to save your newly walled and lit map to a .uvtt file

## Usage Guide

### Detection Modes

#### Edge Detection (Default)
Best for maps with clear wall lines:
- **Edge Sensitivity & Edge Threshold:** Controls edge detection precision - higher sensitivity finds more edges, higher threshold filters out noise
- **Min Area:** Filter out small artifacts
- **Smoothing:** Reduce noise (higher values = more smoothing)
- **Edge Margin:** Exclude detection near image edges

#### Color Detection
Best for maps with distinct colors for the walls or background:
1. Enable "Color Detection"
2. Add colors by:
   - Clicking "Add Color" and selecting from color picker
   - Using "Color Pick" mode to select colors directly from the image
3. Adjust threshold for each color to control matching sensitivity

#### Light Detection
- Adjust light detection settings to automatically find light sources on your map (settings not yet saved in presets).

### Editing Tools

#### Edit Walls Mode
- Draw, erase, and modify walls, doors, and lights
- Hold `Control` while drawing to create continuous lines

#### Deletion Mode
- Click on walls to remove them
- Drag to select and delete multiple walls

#### Edit Mask Mode
Edit the "baked" mask layer:
- **Draw/Erase:** Toggle between adding and removing walls
- **Brush Size:** Control the size of your editing tool
- **Tools:** Choose between brush, line, rectangle, circle, ellipse, or fill tool

#### Thinning Mode
Reduces wall thickness:
- Select thick walls to thin them
- Adjust target width and iteration count for desired results

### Exporting to UVTT

1. Click "Export to UVTT"
2. Configure export settings:
   - **Simplification Tolerance:** Controls how much wall details are simplified (0 = full detail)
   - **Maximum Wall Segment Length:** Limits how long each wall segment can be
   - **Maximum Number of Generation Points:** Caps the total number of generated points for performance
   - **Point Merge Distance:** Connects nearby wall endpoints to fix gaps and lessen the number of walls
   - **Angle Tolerance:** Determines when walls at different angles should merge
   - **Maximum Straight Gap to Connect:** Maximum distance to bridge between straight nearby walls
   - **Grid Snapping:** Optional alignment to a grid for precise positioning

3. Preview the results
4. Save or copy the UVTT file

## Building from Source

To create an executable:

1. Ensure PyInstaller is installed: `pip install pyinstaller`
2. Run the build script: `build.py`
3. Find the executable in the `dist` folder

## Contributing

Contributions are welcome! Please feel free to:

1. Report bugs and request features using the issue tracker
2. Submit pull requests with fixes and improvements
3. Suggest new detection algorithms or optimizations

## License

Auto-Wall is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Uses [OpenCV](https://opencv.org/) for image processing
- Uses [scikit-learn](https://scikit-learn.org/) for color clustering
- Uses [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) for the user interface

---

*Made with ❤️ for the TTRPG community*

---

## Upcoming Changes

- Light detection settings will be saved in presets in the next release.
- The UI is being redesigned for better usability.
- Dedicated executables for MacOS and Linux are planned.
