"""
macOS-specific build functionality for Auto-Wall
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import List

from build_utils import (
    BuildError, print_status, print_success, 
    ensure_directory
)


def build_macos_app():
    """Build Auto-Wall for macOS using PyInstaller with native app bundle creation"""
    try:
        print_status("Building Auto-Wall for macOS")
        
        # Clean previous builds
        dirs_to_clean = ['dist']
        for dir_name in dirs_to_clean:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print_status(f"Removed {dir_name}/")
        
        # Clean only PyInstaller work directory, not the entire build folder
        pyinstaller_work = 'build/pyinstaller_work'
        if os.path.exists(pyinstaller_work):
            shutil.rmtree(pyinstaller_work)
            print_status(f"Removed {pyinstaller_work}/")
        
        os.makedirs('dist', exist_ok=True)
        
        # Check if main script exists
        if not os.path.exists('auto_wall.py'):
            raise BuildError("Main script auto_wall.py not found!")
        
        print_status("Configuring PyInstaller")
        
        # PyQt6 modules that cause conflicts on macOS - exclude them
        excluded_modules = [
            'PyQt6.QtBluetooth', 'PyQt6.QtNfc', 'PyQt6.QtPositioning', 'PyQt6.QtLocation',
            'PyQt6.QtNetworkAuth', 'PyQt6.QtQml', 'PyQt6.QtQuick', 'PyQt6.QtQuick3D',
            'PyQt6.QtQuickWidgets', 'PyQt6.QtWebChannel', 'PyQt6.QtWebEngine', 
            'PyQt6.QtWebEngineCore', 'PyQt6.QtWebEngineWidgets', 'PyQt6.QtWebSockets',
            'PyQt6.Qt3D', 'PyQt6.Qt3DAnimation', 'PyQt6.Qt3DCore', 'PyQt6.Qt3DExtras',
            'PyQt6.Qt3DInput', 'PyQt6.Qt3DLogic', 'PyQt6.Qt3DRender', 'PyQt6.QtCharts',
            'PyQt6.QtDataVisualization', 'PyQt6.QtDesigner', 'PyQt6.QtHelp',
            'PyQt6.QtMultimedia', 'PyQt6.QtMultimediaWidgets', 'PyQt6.QtPdf',
            'PyQt6.QtPdfWidgets', 'PyQt6.QtSensors', 'PyQt6.QtSpatialAudio',
            'PyQt6.QtSql', 'PyQt6.QtStateMachine', 'PyQt6.QtSvg', 'PyQt6.QtSvgWidgets',
            'PyQt6.QtTextToSpeech', 'PyQt6.QtVirtualKeyboard', 'PyQt6.Qt'
        ]

        # Essential modules for the application
        hidden_imports = [
            'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets', 'cv2', 'numpy',
            'sklearn.neighbors._typedefs', 'sklearn.neighbors._partition_nodes',
            'sklearn.utils._heap', 'sklearn.utils._sorting', 'sklearn.tree._utils'
        ]

        # Build PyInstaller command - using native macOS app bundle creation
        cmd = [
            'pyinstaller', '--clean', '--windowed', '--name', 'Auto-Wall',
            '--workpath', 'build/pyinstaller_work', '--distpath', 'dist'
        ]

        # Add all module exclusions
        for module in excluded_modules:
            cmd.extend(['--exclude-module', module])

        # Add hidden imports for essential modules
        for module in hidden_imports:
            cmd.extend(['--hidden-import', module])

        # Add icon if available
        icon_path = 'resources/icon.ico'
        if os.path.exists(icon_path):
            cmd.extend(['--icon', icon_path])

        # Add resource files that the application needs
        resources_dir = 'resources'
        if os.path.exists(resources_dir):
            cmd.extend(['--add-data', f'{resources_dir}{os.pathsep}resources'])

        # Add data directory if it exists
        data_dir = 'data'
        if os.path.exists(data_dir):
            cmd.extend(['--add-data', f'{data_dir}{os.pathsep}data'])

        # Add src directory for any Python modules
        src_dir = 'src'
        if os.path.exists(src_dir):
            cmd.extend(['--add-data', f'{src_dir}{os.pathsep}src'])

        # Add main script
        cmd.append('auto_wall.py')
        
        print_status("Running PyInstaller")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise BuildError(f"PyInstaller failed: {result.stderr}")
        
        print_success("PyInstaller build successful")
        
        # Check if the app was created (PyInstaller creates .app bundle)
        app_dir = 'dist/Auto-Wall.app'
        
        if os.path.exists(app_dir):
            print_success(f"Application built successfully at: {app_dir}")
            
            # Sign the application
            sign_result = sign_application(app_dir)
            if not sign_result:
                print_status("Code signing failed, continuing anyway")
            
            # Create DMG
            print_status("Creating DMG")
            dmg_result = create_dmg(app_dir)
            if dmg_result:
                print_success("Complete macOS build finished")
                print_success(f"App bundle: {app_dir}")
                print_success("DMG installer: dist/Auto-Wall-macOS.dmg")
            
            return True
        else:
            raise BuildError(f"Application not found at expected location: {app_dir}")
            
    except Exception as e:
        raise BuildError(f"Build failed: {e}")


def sign_application(app_dir):
    """Sign the application with ad-hoc signing"""
    try:
        print_status("Code signing application")
        
        # For .app bundles, sign the entire bundle
        target = app_dir
        
        # Sign with ad-hoc signature (self-signed, no developer account needed)
        cmd = [
            'codesign', '--force', '--sign', '-', target
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print_success("Application signed successfully (ad-hoc)")
            
            # Verify the signature
            verify_cmd = ['codesign', '--verify', '--verbose', target]
            verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
            
            if verify_result.returncode == 0:
                print_success("Code signature verified")
            else:
                print_status(f"Signature verification warning: {verify_result.stderr}")
                
            return True
        else:
            print_status(f"Code signing failed: {result.stderr}")
            return False
            
    except Exception as e:
        print_status(f"Code signing error: {e}")
        return False


def create_dmg(app_dir):
    """Create a professional drag-and-drop DMG installer"""
    try:
        dmg_name = "Auto-Wall-macOS.dmg"
        dmg_path = f"dist/{dmg_name}"
        
        # Remove existing DMG
        if os.path.exists(dmg_path):
            os.remove(dmg_path)
        
        # Create a temporary directory for the DMG contents
        dmg_staging_dir = "dist/dmg_staging"
        if os.path.exists(dmg_staging_dir):
            shutil.rmtree(dmg_staging_dir)
        os.makedirs(dmg_staging_dir)
        
        print_status("Setting up DMG layout")
        
        # Copy the app bundle to staging area
        app_bundle_name = os.path.basename(app_dir)
        staging_app_path = os.path.join(dmg_staging_dir, app_bundle_name)
        
        print_status(f"Copying {app_bundle_name} to DMG staging area")
        shutil.copytree(app_dir, staging_app_path, symlinks=True, dirs_exist_ok=True)
        print_success("App bundle copied successfully")
        
        # Create a symbolic link to Applications folder for drag-and-drop
        applications_link = os.path.join(dmg_staging_dir, "Applications")
        os.symlink("/Applications", applications_link)
        
        # Create DMG from the staging directory
        print_status("Creating installer DMG")
        cmd = [
            'hdiutil', 'create',
            '-srcfolder', dmg_staging_dir,
            '-volname', 'Auto-Wall Installer',
            '-format', 'UDZO',      # Compressed format
            '-imagekey', 'zlib-level=9',  # Maximum compression
            '-fs', 'HFS+',          # File system
            dmg_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print_success(f"DMG created: {dmg_path}")
            
            # Check if DMG actually exists and has content
            if os.path.exists(dmg_path):
                size = os.path.getsize(dmg_path)
                size_mb = size / (1024 * 1024)
                print_success(f"DMG size: {size_mb:.1f} MB")
                
                if size > 0:
                    # Clean up staging directory
                    if os.path.exists(dmg_staging_dir):
                        shutil.rmtree(dmg_staging_dir)
                    return True
                else:
                    raise BuildError("DMG is empty - something went wrong")
            else:
                raise BuildError("DMG file was not created")
        else:
            raise BuildError(f"hdiutil failed: {result.stderr}")
            
    except Exception as e:
        raise BuildError(f"DMG creation error: {e}")


class MacOSBuilder:
    """macOS platform builder that uses the working PyInstaller approach"""
    
    def __init__(self, version: str):
        self.version = version
        self.dist_path = Path("dist")
        ensure_directory(self.dist_path)
    
    def build_all(self) -> List[tuple]:
        """Build all macOS packages using the proven approach"""
        results = []
        
        # Use the working build function
        success = build_macos_app()
        
        if success:
            app_path = Path("dist/Auto-Wall.app")
            dmg_path = Path("dist/Auto-Wall-macOS.dmg")
            
            if app_path.exists():
                results.append(("macOS App Bundle", app_path))
            if dmg_path.exists():
                results.append(("DMG Installer", dmg_path))
        else:
            raise BuildError("macOS build failed")
        
        return results