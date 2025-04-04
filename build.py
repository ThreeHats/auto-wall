import os
import sys
import shutil
import subprocess

print("Building Auto-Wall executable...")

# Try to use PyInstaller module directly
try_module = True
try:
    if try_module:
        import PyInstaller.__main__
        print("Successfully imported PyInstaller module")
        use_module = True
    else:
        use_module = False
except ImportError as e:
    print(f"WARNING: Could not import PyInstaller module: {e}")
    print("Will try using PyInstaller command line instead")
    use_module = False

# Get the current directory
root_dir = os.path.abspath(os.path.dirname(__file__))

# Define output directory
output_dir = os.path.join(root_dir, 'dist', 'Auto-Wall')

# Clean any previous build artifacts
if os.path.exists(output_dir):
    print(f"Cleaning previous build at {output_dir}")
    shutil.rmtree(output_dir, ignore_errors=True)

# Create directory for additional files
os.makedirs(output_dir, exist_ok=True)

# Define PyInstaller arguments
pyinstaller_args = [
    'auto_wall.py',                # Script to package
    '--name=Auto-Wall',            # Name of the application
    '--onedir',                    # Create a directory containing an executable
    '--windowed',                  # Windows GUI application (no console)
    '--clean',                     # Clean PyInstaller cache
    f'--distpath={os.path.join(root_dir, "dist")}',  # Output directory
    f'--workpath={os.path.join(root_dir, "build")}', # Work directory
    '--add-data=src/gui/style.qss;src/gui/',  # Add style sheet resource
    '--paths=src',                 # Add source paths
    '--noconfirm',                 # Replace output directory without confirmation
]

# Add specific imports that PyInstaller might miss
hidden_imports = [
    '--hidden-import=sklearn.neighbors._partition_nodes',
    '--hidden-import=sklearn.utils._typedefs',
    '--hidden-import=sklearn.utils._heap',
    '--hidden-import=sklearn.utils._sorting',
    '--hidden-import=sklearn.neighbors._dist_metrics',
    '--hidden-import=scipy.stats'
]
pyinstaller_args.extend(hidden_imports)

# Add icon if available
icon_path = os.path.join(root_dir, 'resources', 'icon.ico')
if os.path.exists(icon_path):
    pyinstaller_args.append(f'--icon={icon_path}')
    
print("Running PyInstaller with arguments:", pyinstaller_args)

try:
    if use_module:
        # Run PyInstaller via the module
        PyInstaller.__main__.run(pyinstaller_args)
    else:
        # Run PyInstaller via command line as fallback
        cmd = ['pyinstaller'] + pyinstaller_args
        print(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
    print(f"Build completed! Executable is located at: {os.path.join(output_dir, 'Auto-Wall.exe')}")

    # Copy sample data (optional)
    sample_data_dir = os.path.join(root_dir, 'data')
    if os.path.exists(sample_data_dir):
        output_data_dir = os.path.join(output_dir, 'data')
        print(f"Copying sample data to {output_data_dir}")
        shutil.copytree(sample_data_dir, output_data_dir, dirs_exist_ok=True)

    # Copy README
    readme_path = os.path.join(root_dir, 'README.md')
    if os.path.exists(readme_path):
        shutil.copy(readme_path, os.path.join(output_dir, 'README.md'))

    print("Build process completed successfully!")
except Exception as e:
    print(f"ERROR: Build process failed: {e}")
    sys.exit(1)
