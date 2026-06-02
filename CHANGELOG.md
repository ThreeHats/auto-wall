# Changelog

All notable changes to Auto-Wall are documented here.

## [Unreleased]

### Added
- Background removal panel and processing pipeline (`rembg` / ONNX Runtime)
- `requirements-gpu.txt` for optional CUDA 12.x GPU acceleration
- **Help → Open Log Folder** menu item — opens the log directory in the system file explorer on all platforms
- CI builds now trigger on pull requests to `main` (artifacts only, no release)
- Beta pre-release support: push a `v*.*.*-beta.*` tag to create a GitHub pre-release

### Changed
- `onnxruntime` version constraint (`>=1.19.0,<2.0.0`)
- Windows distributable switched from single-file to one-directory format (faster startup, no antivirus extraction delay)

### Fixed
- Non-standard characters no longer crash the file picker
- `pymatting` package data missing in Windows executable (caused background removal to fail)

---

## [1.3.9] — macOS build fixes

### Fixed
- macOS Intel native runner for CI builds
- macOS build verify command
- Corrected label text

## [1.3.4]

### Fixed
- Color Pick tool broken in Color Detection mode

## [1.3.3]

### Fixed
- Wall tools UI bug

## [1.3.2]

### Added
- Light detection results now visible in the UI

### Changed
- UVTT grid size now derived from the grid overlay size

## [1.3.1] — UI overhaul

### Added
- New icons throughout the interface
- macOS build support

### Changed
- Full UI overhaul with reorganised controls
- Small UI tweaks and layout fixes

## [1.2.1]

### Changed
- Renamed export/action buttons for clarity

## [1.2.0]

### Added
- Light source detection and editing
- Door/portal drawing
- Wall segment editing mode (Walls Mode)
- Switched export format to UVTT (Universal VTT)

### Changed
- Improved wall edit mode interactions

## [1.1.1]

### Fixed
- Build script corrections

## [1.1.0]

### Added
- Zoom and pan on the image viewer
- Full-resolution image display

### Changed
- Major performance improvements to detection pipeline and drawing tools
- Significant internal refactor across most modules (geometry, contour processor, mask processor, image viewer, selection manager, detection panel, export panel, preset manager)

## [1.0.0]

### Added
- Detection and export settings presets
- Show/hide toggles for edge and color detection settings panels
- Radio buttons to switch detection mode

### Changed
- Rearranged controls layout
- Performance improvements
- Better WebP file support

## [0.9.9]

### Added
- Pixel mode for minimum area control (finer granularity)

### Changed
- Small layout adjustments

### Fixed
- Hatching pattern removal

## [0.9.8]

### Added
- WebP image format support

## [0.9.6]

### Added
- Hover highlight when thinning contours
- Minimum area now scales relative to total image size
- Select-by-edge to pick contours inside other contours

### Fixed
- Exporting to Foundry multiple times no longer ignores the edited mask
- Brush outline preserved correctly

## [0.9.1]

### Added
- Edge detection (Canny) and color detection pipeline
- Contour extraction and wall segment generation
- Paint/mask editing with brush tools
- UVTT export for FoundryVTT, Roll20, Fantasy Grounds, and Arkenforge
- Detection preset system
- CI build pipeline (Windows, Linux, macOS)
