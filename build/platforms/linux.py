"""
Linux-specific build functionality for Auto-Wall
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from build_utils import (
    BuildError, print_status, print_success, print_warning, 
    run_command, ensure_directory, clean_directory, 
    create_desktop_file, get_icon_paths, convert_icon_to_png
)


class LinuxBuilder:
    """Linux platform builder"""
    
    def __init__(self, version: str):
        self.version = version
        self.dist_path = Path("dist")
        ensure_directory(self.dist_path)
    
    def build_all(self, executable_path: Path, skip_appimage: bool = False, skip_deb: bool = False) -> List[tuple]:
        """Build all Linux packages"""
        results = []
        
        if not skip_appimage:
            try:
                appimage_path = self.create_appimage(executable_path)
                if appimage_path:
                    results.append(("AppImage", appimage_path))
            except Exception as e:
                print_warning(f"AppImage creation failed: {e}")
        
        if not skip_deb:
            try:
                deb_path = self.create_debian_package(executable_path)
                results.append(("Debian Package", deb_path))
            except Exception as e:
                print_warning(f"Debian package creation failed: {e}")
        
        return results
    
    def create_appimage(self, executable_path: Path) -> Optional[Path]:
        """Create AppImage package"""
        print_status("Creating AppImage...")
        
        # Download appimagetool if needed
        appimagetool_path = Path("appimagetool-x86_64.AppImage")
        if not appimagetool_path.exists():
            print_status("Downloading appimagetool...")
            run_command([
                "wget", "-O", str(appimagetool_path),
                "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
            ])
            appimagetool_path.chmod(0o755)
        
        # Create AppDir structure
        appdir = Path("Auto-Wall.AppDir")
        clean_directory(appdir)
        
        # Create directory structure
        bin_dir = appdir / "usr" / "bin"
        apps_dir = appdir / "usr" / "share" / "applications"
        icons_dir = appdir / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps"
        
        ensure_directory(bin_dir)
        ensure_directory(apps_dir)
        ensure_directory(icons_dir)
        
        # Copy executable
        shutil.copy2(executable_path, bin_dir / "Auto-Wall")
        
        # Create desktop files
        desktop_file = create_desktop_file(appdir)
        if desktop_file:
            shutil.copy2(desktop_file, apps_dir / "auto-wall.desktop")
        
        # Handle icons
        self._setup_appimage_icons(appdir, icons_dir)
        
        # Create AppRun script
        self._create_apprun_script(appdir)
        
        # Build AppImage
        appimage_name = f"Auto-Wall-{self.version}-x86_64.AppImage"
        env = os.environ.copy()
        env["ARCH"] = "x86_64"
        
        try:
            run_command([
                f"./{appimagetool_path}", str(appdir), appimage_name
            ], check=False)  # appimagetool sometimes returns non-zero even on success
        except BuildError:
            pass  # Ignore appimagetool exit code issues
        
        if Path(appimage_name).exists():
            Path(appimage_name).chmod(0o755)
            # Move to dist folder
            final_path = self.dist_path / appimage_name
            shutil.move(appimage_name, final_path)
            print_success(f"AppImage created: {final_path}")
            return final_path
        else:
            print_warning("AppImage creation may have failed")
            return None
    
    def _setup_appimage_icons(self, appdir: Path, icons_dir: Path) -> None:
        """Setup icons for AppImage"""
        icon_paths = get_icon_paths()
        
        for icon_path in icon_paths:
            if icon_path.suffix.lower() == '.png':
                shutil.copy2(icon_path, icons_dir / "auto-wall.png")
                break
            elif icon_path.suffix.lower() == '.ico':
                png_path = appdir / "auto-wall.png"
                if convert_icon_to_png(icon_path, png_path):
                    shutil.copy2(png_path, icons_dir / "auto-wall.png")
                    break
    
    def _create_apprun_script(self, appdir: Path) -> None:
        """Create AppRun script for AppImage"""
        apprun_content = '''#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
export PATH="${HERE}/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${HERE}/usr/lib:${LD_LIBRARY_PATH}"
cd "${HERE}"
exec "${HERE}/usr/bin/Auto-Wall" "$@"
'''
        
        apprun_path = appdir / "AppRun"
        with open(apprun_path, "w") as f:
            f.write(apprun_content)
        
        apprun_path.chmod(0o755)
    
    def create_debian_package(self, executable_path: Path) -> Path:
        """Create .deb package"""
        print_status("Creating Debian package...")
        
        # Create package structure
        pkg_dir = Path("auto-wall_debian")
        clean_directory(pkg_dir)
        
        # Create directory structure
        bin_dir = pkg_dir / "usr" / "bin"
        apps_dir = pkg_dir / "usr" / "share" / "applications"
        icons_dir = pkg_dir / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps"
        doc_dir = pkg_dir / "usr" / "share" / "doc" / "auto-wall"
        debian_dir = pkg_dir / "DEBIAN"
        
        for directory in [bin_dir, apps_dir, icons_dir, doc_dir, debian_dir]:
            ensure_directory(directory)
        
        # Copy executable
        shutil.copy2(executable_path, bin_dir / "auto-wall")
        
        # Create desktop file
        desktop_content = """[Desktop Entry]
Name=Auto-Wall
Comment=Battle Map Wall Detection Tool
Exec=auto-wall
Icon=auto-wall
Type=Application
Categories=Graphics;Photography;
Terminal=false
StartupWMClass=Auto-Wall
"""
        with open(apps_dir / "auto-wall.desktop", "w") as f:
            f.write(desktop_content)
        
        # Handle icons
        self._setup_debian_icons(icons_dir)
        
        # Setup documentation
        self._setup_debian_docs(doc_dir)
        
        # Create control files
        self._create_debian_control_files(debian_dir, pkg_dir)
        
        # Build package
        deb_name = f"auto-wall_{self.version}_amd64.deb"
        run_command(["fakeroot", "dpkg-deb", "--build", str(pkg_dir), deb_name])
        
        # Clean up
        shutil.rmtree(pkg_dir)
        
        if Path(deb_name).exists():
            # Move to dist folder
            final_path = self.dist_path / deb_name
            shutil.move(deb_name, final_path)
            print_success(f"Debian package created: {final_path}")
            return final_path
        else:
            raise BuildError("Failed to create Debian package")
    
    def _setup_debian_icons(self, icons_dir: Path) -> None:
        """Setup icons for Debian package"""
        icon_paths = get_icon_paths()
        
        for icon_path in icon_paths:
            if icon_path.suffix.lower() == '.png':
                shutil.copy2(icon_path, icons_dir / "auto-wall.png")
                break
            elif icon_path.suffix.lower() == '.ico':
                if convert_icon_to_png(icon_path, icons_dir / "auto-wall.png"):
                    break
    
    def _setup_debian_docs(self, doc_dir: Path) -> None:
        """Setup documentation for Debian package"""
        # Copy documentation files
        for doc_file in ["README.md", "LICENSE"]:
            src_path = Path(doc_file)
            if src_path.exists():
                dest_name = "README" if doc_file == "README.md" else doc_file.lower()
                shutil.copy2(src_path, doc_dir / dest_name)
        
        # Create changelog
        changelog_content = f"""auto-wall ({self.version}) stable; urgency=medium

  * Version {self.version} release

 -- Auto-Wall Team <auto-wall@example.com>  {subprocess.check_output(['date', '-R']).decode().strip()}
"""
        
        changelog_path = doc_dir / "changelog"
        with open(changelog_path, "w") as f:
            f.write(changelog_content)
        
        # Compress changelog
        run_command(["gzip", "-9", str(changelog_path)])
    
    def _create_debian_control_files(self, debian_dir: Path, pkg_dir: Path) -> None:
        """Create Debian control files"""
        # Get installed size
        size_bytes = sum(f.stat().st_size for f in pkg_dir.rglob('*') if f.is_file())
        size_kb = (size_bytes // 1024) + 1
        
        # Create control file
        control_content = f"""Package: auto-wall
Version: {self.version}
Section: graphics
Priority: optional
Architecture: amd64
Depends: python3, python3-tk, libc6
Installed-Size: {size_kb}
Maintainer: Auto-Wall Team <auto-wall@example.com>
Description: Battle Map Wall Detection Tool
 Auto-Wall is a powerful tool for detecting and managing walls in battle maps
 for tabletop RPGs. It provides automated wall detection, manual editing tools,
 and export capabilities for various VTT platforms.
Homepage: https://github.com/ThreeHats/auto-wall
"""
        
        with open(debian_dir / "control", "w") as f:
            f.write(control_content)
        
        # Create postinst script
        postinst_content = """#!/bin/bash
set -e

# Update desktop database
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database -q /usr/share/applications
fi

# Update icon cache
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -q /usr/share/icons/hicolor
fi

exit 0
"""
        
        postinst_path = debian_dir / "postinst"
        with open(postinst_path, "w") as f:
            f.write(postinst_content)
        postinst_path.chmod(0o755)
        
        # Create postrm script
        postrm_content = """#!/bin/bash
set -e

case "$1" in
    remove|purge)
        # Update desktop database
        if command -v update-desktop-database >/dev/null 2>&1; then
            update-desktop-database -q /usr/share/applications
        fi
        
        # Update icon cache
        if command -v gtk-update-icon-cache >/dev/null 2>&1; then
            gtk-update-icon-cache -q /usr/share/icons/hicolor
        fi
        ;;
esac

exit 0
"""
        
        postrm_path = debian_dir / "postrm"
        with open(postrm_path, "w") as f:
            f.write(postrm_content)
        postrm_path.chmod(0o755)