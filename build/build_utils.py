"""
Shared build utilities for Auto-Wall cross-platform builds
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


class BuildError(Exception):
    """Custom exception for build errors"""
    pass


def print_status(message: str) -> None:
    """Print a status message"""
    print(f"[INFO] {message}")


def print_success(message: str) -> None:
    """Print a success message"""
    print(f"[SUCCESS] {message}")


def print_warning(message: str) -> None:
    """Print a warning message"""
    print(f"[WARNING] {message}")


def print_error(message: str) -> None:
    """Print an error message"""
    print(f"[ERROR] {message}")


def run_command(cmd: List[str], check: bool = True, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Run a shell command and return the result"""
    print_status(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(cmd)}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        if check:
            raise BuildError(f"Command failed with exit code {e.returncode}")
        return e


def ensure_directory(path: Path) -> None:
    """Ensure a directory exists, creating it if necessary"""
    path.mkdir(parents=True, exist_ok=True)


def clean_directory(path: Path) -> None:
    """Remove and recreate a directory"""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def get_version() -> str:
    """Get the application version from the main file"""
    try:
        # Try to extract from auto_wall.py
        auto_wall_path = Path("auto_wall.py")
        if auto_wall_path.exists():
            with open(auto_wall_path, "r") as f:
                content = f.read()
                # Look for version string
                for line in content.split("\n"):
                    if "VERSION" in line and "=" in line:
                        version = line.split("=")[-1].strip().strip('"').strip("'")
                        return version
        
        # Fallback version
        return "1.3.0"
    except Exception as e:
        print_warning(f"Could not determine version: {e}")
        return "1.3.0"


def install_python_deps() -> None:
    """Install Python dependencies"""
    print_status("Installing Python dependencies...")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print_error("requirements.txt not found")
        raise BuildError("Missing requirements.txt")
    
    # Install dependencies
    run_command([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ])
    
    # Install build dependencies
    build_deps = ["pyinstaller", "pillow"]
    run_command([
        sys.executable, "-m", "pip", "install"
    ] + build_deps)


def get_pyinstaller_args(platform: str) -> Tuple[List[str], List[str]]:
    """Get PyInstaller arguments for the specified platform"""
    
    # Use semicolon on Windows, colon on other platforms for --add-data
    separator = ";" if platform.lower() == "windows" else ":"
    
    base_args = [
        "auto_wall.py",
        "--name=Auto-Wall",
        "--onefile",
        "--windowed",
        "--clean",
        "--distpath=dist",
        "--workpath=build/pyinstaller_work",
        "--collect-all=cv2",
        "--collect-all=sklearn",
        "--collect-all=numpy",
        "--collect-all=PIL",
        "--paths=src",
        "--noconfirm",
        "--noupx",
        f"--add-data=src/styles/style.qss{separator}src/styles",
        f"--add-data=resources{separator}resources",
    ]
    
    hidden_imports = [
        "sklearn.neighbors._partition_nodes",
        "sklearn.utils._typedefs",
        "sklearn.utils._heap",
        "sklearn.utils._sorting",
        "sklearn.tree._utils",
        "scipy.stats",
        "scipy.sparse.csgraph._validation",
    ]
    
    excluded_modules = [
        "tkinter",
        "test",
        "pytest",
        "doctest",
        "pdb",
        "IPython",
        "jupyter",
        "matplotlib.tests",
        "scipy.tests",
        "pandas",
        "torch",
        "tensorflow",
    ]
    
    # Add platform-specific arguments
    if platform == "windows":
        base_args.extend([
            "--icon=resources/icon.ico",
            "--version-file=Auto-Wall.spec.version" if Path("Auto-Wall.spec.version").exists() else None,
        ])
        base_args = [arg for arg in base_args if arg is not None]
    
    elif platform == "macos":
        base_args.extend([
            "--icon=resources/icon.ico",
            "--osx-bundle-identifier=com.threehats.auto-wall",
        ])
    
    # Add hidden imports
    for imp in hidden_imports:
        base_args.extend(["--hidden-import", imp])
    
    # Add excluded modules
    for mod in excluded_modules:
        base_args.extend(["--exclude-module", mod])
    
    return base_args, hidden_imports


def build_pyinstaller_executable(platform: str) -> Path:
    """Build the PyInstaller executable for the specified platform"""
    print_status(f"Building standalone executable for {platform}...")
    
    # Ensure dist directory exists
    dist_path = Path("dist")
    ensure_directory(dist_path)
    
    # Get PyInstaller arguments
    args, _ = get_pyinstaller_args(platform)
    
    # Build executable
    cmd = [sys.executable, "-m", "PyInstaller"] + args
    run_command(cmd)
    
    # Determine executable name
    if platform == "windows":
        exe_name = "Auto-Wall.exe"
    else:
        exe_name = "Auto-Wall"
    
    exe_path = dist_path / exe_name
    
    if not exe_path.exists():
        raise BuildError(f"Executable not found: {exe_path}")
    
    print_success(f"Executable built: {exe_path}")
    return exe_path


def copy_resources(dest_dir: Path) -> None:
    """Copy application resources to destination directory"""
    resources_dir = Path("resources")
    if resources_dir.exists():
        dest_resources = dest_dir / "resources"
        if dest_resources.exists():
            shutil.rmtree(dest_resources)
        shutil.copytree(resources_dir, dest_resources)


def create_desktop_file(app_dir: Path, executable_name: str = "Auto-Wall") -> None:
    """Create a .desktop file for Linux"""
    desktop_content = f"""[Desktop Entry]
Name=Auto-Wall
Comment=Battle Map Wall Detection Tool
Exec={executable_name}
Icon=auto-wall
Type=Application
Categories=Graphics;Photography;
Terminal=false
StartupWMClass=Auto-Wall
"""
    
    desktop_file = app_dir / "Auto-Wall.desktop"
    with open(desktop_file, "w") as f:
        f.write(desktop_content)
    
    return desktop_file


def get_icon_paths() -> List[Path]:
    """Get available icon file paths"""
    icon_paths = []
    resources_dir = Path("resources")
    
    for icon_file in ["icon.ico", "icon.png", "auto-wall-preview.png"]:
        icon_path = resources_dir / icon_file
        if icon_path.exists():
            icon_paths.append(icon_path)
    
    return icon_paths


def convert_icon_to_png(icon_path: Path, output_path: Path) -> bool:
    """Convert an icon file to PNG format"""
    try:
        from PIL import Image
        with Image.open(icon_path) as img:
            # Convert to RGBA if necessary
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            img.save(output_path, "PNG")
        return True
    except ImportError:
        print_warning("PIL not available for icon conversion")
        return False
    except Exception as e:
        print_warning(f"Icon conversion failed: {e}")
        return False