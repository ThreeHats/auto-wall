"""
macOS-specific build functionality for Auto-Wall
"""

import plistlib
import shutil
from pathlib import Path
from typing import List

from build_utils import (
    BuildError, print_status, print_success, 
    ensure_directory, get_icon_paths
)


class MacOSBuilder:
    """macOS platform builder"""
    
    def __init__(self, version: str):
        self.version = version
        self.dist_path = Path("dist")
        ensure_directory(self.dist_path)
    
    def build_all(self, executable_path: Path) -> List[tuple]:
        """Build all macOS packages"""
        results = []
        
        try:
            app_path = self.create_app_bundle(executable_path)
            results.append(("macOS App Bundle", app_path))
        except Exception as e:
            print_status(f"App bundle creation failed: {e}")
        
        try:
            dmg_path = self.create_dmg(executable_path)
            results.append(("DMG Image", dmg_path))
        except Exception as e:
            print_status(f"DMG creation failed: {e}")
        
        return results
    
    def create_app_bundle(self, executable_path: Path) -> Path:
        """Create macOS .app bundle"""
        print_status("Creating macOS .app bundle...")
        
        # Create bundle structure
        bundle_name = "Auto-Wall.app"
        bundle_path = self.dist_path / bundle_name
        
        if bundle_path.exists():
            shutil.rmtree(bundle_path)
        
        # Create directory structure
        contents_dir = bundle_path / "Contents"
        macos_dir = contents_dir / "MacOS"
        resources_dir = contents_dir / "Resources"
        
        ensure_directory(macos_dir)
        ensure_directory(resources_dir)
        
        # Copy executable
        shutil.copy2(executable_path, macos_dir / "Auto-Wall")
        
        # Create Info.plist
        self._create_info_plist(contents_dir)
        
        # Copy icons
        self._setup_macos_icons(resources_dir)
        
        # Copy resources
        resources_src = Path("resources")
        if resources_src.exists():
            shutil.copytree(resources_src, resources_dir / "resources")
        
        print_success(f"macOS app bundle created: {bundle_path}")
        return bundle_path
    
    def _create_info_plist(self, contents_dir: Path) -> None:
        """Create Info.plist for app bundle"""
        plist_data = {
            'CFBundleName': 'Auto-Wall',
            'CFBundleDisplayName': 'Auto-Wall',
            'CFBundleIdentifier': 'com.threehats.auto-wall',
            'CFBundleVersion': self.version,
            'CFBundleShortVersionString': self.version,
            'CFBundlePackageType': 'APPL',
            'CFBundleSignature': 'AWAL',
            'CFBundleExecutable': 'Auto-Wall',
            'CFBundleIconFile': 'auto-wall.icns',
            'LSMinimumSystemVersion': '10.14.0',
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,
        }
        
        plist_path = contents_dir / "Info.plist"
        with open(plist_path, 'wb') as f:
            plistlib.dump(plist_data, f)
    
    def _setup_macos_icons(self, resources_dir: Path) -> None:
        """Setup icons for macOS app bundle"""
        # For now, just copy available icons
        # Future: Convert to .icns format
        icon_paths = get_icon_paths()
        
        for icon_path in icon_paths:
            if icon_path.suffix.lower() == '.ico':
                # Copy as placeholder until we implement proper .icns conversion
                shutil.copy2(icon_path, resources_dir / "auto-wall.icns")
                break
    
    def create_dmg(self, executable_path: Path) -> Path:
        """Create DMG disk image (placeholder for future implementation)"""
        # Future: Use hdiutil to create proper DMG
        print_status("DMG creation not yet implemented - creating ZIP instead")
        
        # Create ZIP as fallback
        zip_name = f"Auto-Wall-{self.version}-macOS.zip"
        zip_path = self.dist_path / zip_name
        
        # Create temporary directory
        temp_dir = Path("Auto-Wall-macOS")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        temp_dir.mkdir()
        
        # Copy executable and docs
        shutil.copy2(executable_path, temp_dir / "Auto-Wall")
        
        for doc_file in ["README.md", "LICENSE"]:
            doc_path = Path(doc_file)
            if doc_path.exists():
                shutil.copy2(doc_path, temp_dir / doc_file)
        
        # Create ZIP
        shutil.make_archive(str(self.dist_path / f"Auto-Wall-{self.version}-macOS"), 'zip', str(temp_dir))
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        if zip_path.exists():
            print_success(f"macOS ZIP created: {zip_path}")
            return zip_path
        else:
            raise BuildError("Failed to create macOS ZIP")