"""
Windows-specific build functionality for Auto-Wall
"""

import shutil
from pathlib import Path
from typing import List

from build_utils import (
    BuildError, print_status, print_success, 
    ensure_directory
)


class WindowsBuilder:
    """Windows platform builder"""
    
    def __init__(self, version: str):
        self.version = version
        self.dist_path = Path("dist")
        ensure_directory(self.dist_path)
    
    def build_all(self, executable_path: Path) -> List[tuple]:
        """Build all Windows packages"""
        results = []
        
        # For Windows, we mainly just have the executable
        # Could add installer creation here in the future
        results.append(("Windows Executable", executable_path))
        
        # Create ZIP distribution
        try:
            zip_path = self.create_zip_distribution(executable_path)
            results.append(("ZIP Distribution", zip_path))
        except Exception as e:
            print_status(f"ZIP creation failed: {e}")
        
        return results
    
    def create_zip_distribution(self, executable_path: Path) -> Path:
        """Create a ZIP distribution with executable and resources"""
        print_status("Creating ZIP distribution...")
        
        # Create temporary directory for ZIP contents
        zip_dir = Path("Auto-Wall-Windows")
        if zip_dir.exists():
            shutil.rmtree(zip_dir)
        
        zip_dir.mkdir()
        
        # Copy executable
        shutil.copy2(executable_path, zip_dir / "Auto-Wall.exe")
        
        # Copy documentation
        for doc_file in ["README.md", "LICENSE"]:
            doc_path = Path(doc_file)
            if doc_path.exists():
                shutil.copy2(doc_path, zip_dir / doc_file)
        
        # Create ZIP file
        zip_name = f"Auto-Wall-{self.version}-Windows"
        zip_path = self.dist_path / f"{zip_name}.zip"
        
        shutil.make_archive(str(self.dist_path / zip_name), 'zip', str(zip_dir))
        
        # Clean up
        shutil.rmtree(zip_dir)
        
        if zip_path.exists():
            print_success(f"ZIP distribution created: {zip_path}")
            return zip_path
        else:
            raise BuildError("Failed to create ZIP distribution")
    
    def create_installer(self, executable_path: Path) -> Path:
        """Create Windows installer (placeholder for future implementation)"""
        # Future: Use NSIS, Inno Setup, or WiX to create installer
        raise NotImplementedError("Windows installer creation not yet implemented")