"""
Mode Manager - Centralized tool and mode state management

This module provides a clean, centralized way to manage the application's
different tools and modes, replacing the scattered mode flags and logic.
"""
from enum import Enum
from typing import Optional, Callable, Dict, Any
from PyQt6.QtCore import QObject, pyqtSignal


class ToolType(Enum):
    """Enumeration of available tools."""
    DETECT = "detect"
    PAINT = "paint"
    UVTT_EDITOR = "uvtt_editor"


class DetectionMode(Enum):
    """Enumeration of detection modes."""
    EDGE = "edge"
    COLOR = "color"


class SelectMode(Enum):
    """Enumeration of selection tool modes."""
    DELETE = "delete"
    THIN = "thin"


class PaintMode(Enum):
    """Enumeration of paint tool modes."""
    DRAW = "draw"
    ERASE = "erase"


class DrawingTool(Enum):
    """Enumeration of drawing tools."""
    BRUSH = "brush"
    LINE = "line"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    FILL = "fill"


class UVTTMode(Enum):
    """Enumeration of UVTT editor modes."""
    DRAW_WALLS = "draw_walls"
    EDIT_WALLS = "edit_walls"
    DELETE_WALLS = "delete_walls"
    DRAW_PORTALS = "draw_portals"


class ModeManager(QObject):
    """
    Centralized mode management for the Auto-Wall application.
    
    This class provides a single source of truth for the current tool,
    detection mode, and tool-specific settings.
    """
    
    # Signals emitted when modes change
    tool_changed = pyqtSignal(ToolType)
    detection_mode_changed = pyqtSignal(DetectionMode)
    select_mode_changed = pyqtSignal(SelectMode)
    paint_mode_changed = pyqtSignal(PaintMode)
    drawing_tool_changed = pyqtSignal(DrawingTool)
    uvtt_mode_changed = pyqtSignal(UVTTMode)
    
    def __init__(self):
        super().__init__()
        
        # Current state
        self._current_tool = ToolType.DETECT
        self._detection_mode = DetectionMode.EDGE
        self._select_mode = SelectMode.DELETE
        self._paint_mode = PaintMode.DRAW
        self._drawing_tool = DrawingTool.BRUSH
        self._uvtt_mode = UVTTMode.DRAW_WALLS
        
        # Tool-specific settings storage
        self._settings: Dict[ToolType, Dict[str, Any]] = {
            ToolType.DETECT: {
                'light_detection_enabled': False,
                'mode': DetectionMode.EDGE,
                'high_res_processing': False,
                'merge_contours': True
            },
            ToolType.SELECT: {
                'mode': SelectMode.DELETE,
                'target_width': 5,
                'max_iterations': 3
            },
            ToolType.PAINT: {
                'mode': PaintMode.DRAW,
                'drawing_tool': DrawingTool.BRUSH,
                'brush_size': 10
            },
            ToolType.COLOR_PICKER: {
                'num_colors': 3,
                'tolerance': 20
            },
            ToolType.UVTT_EDITOR: {
                'mode': UVTTMode.DRAW_WALLS,
                'grid_size': 70,
                'show_grid': False
            }
        }
    
    # === Tool Management ===
    
    @property
    def current_tool(self) -> ToolType:
        """Get the currently active tool."""
        return self._current_tool
    
    @current_tool.setter
    def current_tool(self, tool: ToolType):
        """Set the currently active tool."""
        if self._current_tool != tool:
            self._current_tool = tool
            self.tool_changed.emit(tool)
    
    # === Detection Mode Management ===
    
    @property
    def detection_mode(self) -> DetectionMode:
        """Get the current detection mode."""
        return self._detection_mode
    
    @detection_mode.setter
    def detection_mode(self, mode: DetectionMode):
        """Set the current detection mode."""
        if self._detection_mode != mode:
            self._detection_mode = mode
            self._settings[ToolType.DETECT]['mode'] = mode
            self.detection_mode_changed.emit(mode)
    
    # === Selection Mode Management ===
    
    @property
    def select_mode(self) -> SelectMode:
        """Get the current selection mode."""
        return self._select_mode
    
    @select_mode.setter
    def select_mode(self, mode: SelectMode):
        """Set the current selection mode."""
        if self._select_mode != mode:
            self._select_mode = mode
            self._settings[ToolType.SELECT]['mode'] = mode
            self.select_mode_changed.emit(mode)
    
    # === Paint Mode Management ===
    
    @property
    def paint_mode(self) -> PaintMode:
        """Get the current paint mode."""
        return self._paint_mode
    
    @paint_mode.setter
    def paint_mode(self, mode: PaintMode):
        """Set the current paint mode."""
        if self._paint_mode != mode:
            self._paint_mode = mode
            self._settings[ToolType.PAINT]['mode'] = mode
            self.paint_mode_changed.emit(mode)
    
    @property
    def drawing_tool(self) -> DrawingTool:
        """Get the current drawing tool."""
        return self._drawing_tool
    
    @drawing_tool.setter
    def drawing_tool(self, tool: DrawingTool):
        """Set the current drawing tool."""
        if self._drawing_tool != tool:
            self._drawing_tool = tool
            self._settings[ToolType.PAINT]['drawing_tool'] = tool
            self.drawing_tool_changed.emit(tool)
    
    # === UVTT Mode Management ===
    
    @property
    def uvtt_mode(self) -> UVTTMode:
        """Get the current UVTT mode."""
        return self._uvtt_mode
    
    @uvtt_mode.setter
    def uvtt_mode(self, mode: UVTTMode):
        """Set the current UVTT mode."""
        if self._uvtt_mode != mode:
            self._uvtt_mode = mode
            self._settings[ToolType.UVTT_EDITOR]['mode'] = mode
            self.uvtt_mode_changed.emit(mode)
    
    # === Settings Management ===
    
    def get_tool_setting(self, tool: ToolType, key: str, default=None):
        """Get a tool-specific setting."""
        return self._settings.get(tool, {}).get(key, default)
    
    def set_tool_setting(self, tool: ToolType, key: str, value: Any):
        """Set a tool-specific setting."""
        if tool not in self._settings:
            self._settings[tool] = {}
        self._settings[tool][key] = value
    
    def get_current_tool_settings(self) -> Dict[str, Any]:
        """Get all settings for the currently active tool."""
        return self._settings.get(self._current_tool, {}).copy()
    
    
    # === Status and Info Methods ===
    
    def get_status_tip(self) -> str:
        """Get the appropriate status tip for the current mode."""
        if self.is_detection_tool_active():
            if self.is_edge_detection_active():
                return "Detection Mode: Adjust parameters and click 'Detect Walls' to find edges"
            else:
                return "Color Detection Mode: Select wall colors and detect by color similarity"
                
        elif self.is_selection_tool_active():
            if self.is_delete_mode_active():
                return "Delete Mode: Click inside contours or on lines to delete them"
            else:
                return "Thin Mode: Click on contours to thin them"
                
        elif self.is_paint_tool_active():
            if self._paint_mode == PaintMode.DRAW:
                return "Paint Mode: Draw on the mask layer"
            else:
                return "Erase Mode: Erase from the mask layer"
            
        elif self.is_color_picker_active():
            return "Color Picker: Drag to select colors from the image"
            
        elif self.is_uvtt_editor_active():
            mode_tips = {
                UVTTMode.DRAW_WALLS: "Draw Mode: Click and drag to draw new wall segments",
                UVTTMode.EDIT_WALLS: "Edit Mode: Click and drag wall endpoints to move them",
                UVTTMode.DELETE_WALLS: "Delete Mode: Click on walls to delete them",
                UVTTMode.DRAW_PORTALS: "Portal Mode: Click and drag to draw doors and portals"
            }
            return mode_tips.get(self._uvtt_mode, "UVTT Editor Mode")
        
        return "Ready"
    
    def get_tool_name(self, tool: ToolType = None) -> str:
        """Get the display name for a tool."""
        if tool is None:
            tool = self._current_tool
            
        names = {
            ToolType.DETECT: "Detection",
            ToolType.PAINT: "Paint",
            ToolType.UVTT_EDITOR: "UVTT Editor"
        }
        return names.get(tool, "Unknown")
    
    # === Utility Methods ===
    
    def reset_to_defaults(self):
        """Reset all modes to their default values."""
        self.current_tool = ToolType.DETECT
        self.detection_mode = DetectionMode.EDGE
        self.select_mode = SelectMode.DELETE
        self.paint_mode = PaintMode.DRAW
        self.drawing_tool = DrawingTool.BRUSH
        self.uvtt_mode = UVTTMode.DRAW_WALLS
    
    def switch_to_tool(self, tool: ToolType):
        """Convenience method to switch to a specific tool."""
        self.current_tool = tool