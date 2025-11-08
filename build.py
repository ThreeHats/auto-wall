#!/usr/bin/env python3
"""
Auto-Wall Cross-Platform Build System

A unified build system that handles building Auto-Wall for Windows, Linux, and macOS.
Replaces the separate platform-specific build scripts with a single, maintainable solution.

Usage:
    python3 build.py                     # Build for current platform
    python3 build.py --platform linux    # Build for specific platform
    python3 build.py --install-deps      # Install build dependencies
    python3 build.py --clean             # Clean build artifacts
"""

import argparse
import platform
import sys
from pathlib import Path

# Add the build directory to the Python path
build_dir = Path(__file__).parent / "build"
sys.path.insert(0, str(build_dir))

from build_utils import (
    BuildError, print_status, print_success, print_error,
    get_version, install_python_deps, build_pyinstaller_executable,
    ensure_directory, clean_directory
)


class AutoWallBuilder:
    """Cross-platform Auto-Wall builder"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.root_dir = Path(__file__).parent  # Root directory
        self.version = get_version()
        
        # Change to root directory for builds
        import os
        os.chdir(self.root_dir)
        
        print(f"Auto-Wall Cross-Platform Build System")
        print(f"======================================")
        print(f"Building Auto-Wall version {self.version}")
        print(f"Detected platform: {self.system}")
        print()
    
    def install_dependencies(self) -> None:
        """Install build dependencies"""
        print_status("Installing build dependencies...")
        install_python_deps()
        
        # Platform-specific dependencies
        if self.system == "linux":
            self._install_linux_deps()
        elif self.system == "windows":
            self._install_windows_deps()
        elif self.system == "darwin":
            self._install_macos_deps()
    
    def _install_linux_deps(self) -> None:
        """Install Linux-specific build dependencies"""
        print_status("Installing Linux build dependencies...")
        # These would typically be installed via package manager
        print("Please ensure the following packages are installed:")
        print("- wget (for downloading appimagetool)")
        print("- fakeroot (for building .deb packages)")
        print("- dpkg-deb (for building .deb packages)")
        print("- gzip (for compressing documentation)")
    
    def _install_windows_deps(self) -> None:
        """Install Windows-specific build dependencies"""
        print_status("Installing Windows build dependencies...")
        # Future: Add Windows installer creation tools
        print("Windows build dependencies: None additional required")
    
    def _install_macos_deps(self) -> None:
        """Install macOS-specific build dependencies"""
        print_status("Installing macOS build dependencies...")
        # Future: Add macOS-specific tools
        print("macOS build dependencies: None additional required")
    
    def clean_build_artifacts(self) -> None:
        """Clean all build artifacts"""
        print_status("Cleaning build artifacts...")
        
        artifacts_to_clean = [
            "build/pyinstaller_work",
            "dist",
            "__pycache__",
            "*.spec",
            "Auto-Wall.AppDir",
            "auto-wall_debian",
            "appimagetool-x86_64.AppImage",
        ]
        
        for artifact in artifacts_to_clean:
            artifact_path = Path(artifact)
            if artifact_path.exists():
                if artifact_path.is_dir():
                    clean_directory(artifact_path)
                else:
                    artifact_path.unlink()
                print_status(f"Removed: {artifact}")
        
        # Clean Python cache
        import subprocess
        try:
            subprocess.run(["find", ".", "-name", "*.pyc", "-delete"], check=False)
            subprocess.run(["find", ".", "-name", "__pycache__", "-type", "d", "-exec", "rm", "-rf", "{}", "+"], check=False)
        except FileNotFoundError:
            pass  # find command not available (Windows)
        
        print_success("Build artifacts cleaned")
    
    def build_for_platform(self, target_platform: str = None, **kwargs) -> None:
        """Build for specified platform"""
        platform_name = target_platform or self.system
        
        print_status(f"Building for {platform_name}...")
        
        # For all platforms, use the standard approach with platform-specific builders
        # macOS doesn't need PyInstaller executable first - it handles PyInstaller internally
        if platform_name not in ["darwin", "macos"]:
            executable_path = build_pyinstaller_executable(platform_name)
        
        # Import platform-specific builder
        if platform_name == "linux":
            from platforms.linux import LinuxBuilder
            builder = LinuxBuilder(self.version)
            results = builder.build_all(
                executable_path, 
                skip_appimage=kwargs.get('skip_appimage', False),
                skip_deb=kwargs.get('skip_deb', False)
            )
        elif platform_name == "windows":
            from platforms.windows import WindowsBuilder
            builder = WindowsBuilder(self.version)
            results = builder.build_all(executable_path)
        elif platform_name in ["darwin", "macos"]:
            from platforms.macos import MacOSBuilder
            builder = MacOSBuilder(self.version)
            results = builder.build_all()
            # Set executable path for summary
            executable_path = Path("dist/Auto-Wall.app/Contents/MacOS/Auto-Wall")
        else:
            raise BuildError(f"Unsupported platform: {platform_name}")
        
        # Print summary
        self._print_build_summary(executable_path, results)
    
    def _print_build_summary(self, executable_path: Path, results: list) -> None:
        """Print build completion summary"""
        print()
        print("=" * 50)
        print("BUILD COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        print(f"Standalone executable: {executable_path}")
        
        for package_type, package_path in results:
            print(f"{package_type}: {package_path}")
        
        print()
        print("Build commands:")
        print("  python3 build.py --install-deps    # Install dependencies")
        print("  python3 build.py --clean           # Clean artifacts")
        print("  python3 build.py --platform linux  # Build for Linux")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Auto-Wall Cross-Platform Build System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 build.py                     # Build for current platform
  python3 build.py --platform linux    # Build for Linux
  python3 build.py --install-deps      # Install build dependencies
  python3 build.py --clean             # Clean build artifacts
  
Linux-specific options:
  python3 build.py --skip-appimage     # Skip AppImage creation
  python3 build.py --skip-deb          # Skip Debian package creation
        """
    )
    
    parser.add_argument(
        '--platform', 
        choices=['linux', 'windows', 'macos', 'darwin'], 
        help='Target platform (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--install-deps', 
        action='store_true',
        help='Install build dependencies and exit'
    )
    
    parser.add_argument(
        '--clean', 
        action='store_true',
        help='Clean build artifacts and exit'
    )
    
    # Linux-specific options
    parser.add_argument(
        '--skip-appimage', 
        action='store_true',
        help='Skip AppImage creation (Linux only)'
    )
    
    parser.add_argument(
        '--skip-deb', 
        action='store_true',
        help='Skip Debian package creation (Linux only)'
    )
    
    return parser


def main():
    """Main build entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        builder = AutoWallBuilder()
        
        if args.install_deps:
            builder.install_dependencies()
            return
        
        if args.clean:
            builder.clean_build_artifacts()
            return
        
        # Normalize platform name
        platform_name = args.platform
        if platform_name == "darwin":
            platform_name = "macos"
        
        # Build for platform
        builder.build_for_platform(
            target_platform=platform_name,
            skip_appimage=args.skip_appimage,
            skip_deb=args.skip_deb
        )
        
    except BuildError as e:
        print_error(f"Build failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print_error("Build cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
