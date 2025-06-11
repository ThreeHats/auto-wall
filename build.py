import glob
import os
import shutil
import subprocess
import sys

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
root_dir = os.getcwd()

# Define output directory
output_dir = os.path.join(root_dir, 'dist', 'Auto-Wall')

# Clean any previous build artifacts
if os.path.exists(output_dir):
    print(f"Cleaning previous build at {output_dir}")
    shutil.rmtree(output_dir, ignore_errors=True)

# Clean build cache and spec file to force fresh build
build_dir = os.path.join(root_dir, 'build')
if os.path.exists(build_dir):
    print(f"Cleaning build cache at {build_dir}")
    shutil.rmtree(build_dir, ignore_errors=True)

spec_file = os.path.join(root_dir, 'Auto-Wall.spec')
if os.path.exists(spec_file):
    print(f"Removing existing spec file {spec_file}")
    os.remove(spec_file)

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
    f'--add-data=src/gui/style.qss{os.pathsep}src/gui/',  # Add style sheet resource
    f'--add-data=resources{os.pathsep}resources/',  # Add resources directory (splash screen, icons, etc.)
    '--paths=src',                 # Add source paths
    '--noconfirm',                 # Replace output directory without confirmation
    '--noupx',                     # Disable UPX compression to speed up build
    '--log-level=INFO',            # Reduce log verbosity
]

# Add specific imports that PyInstaller might miss
hidden_imports = [
    '--hidden-import=sklearn.neighbors._partition_nodes',
    '--hidden-import=sklearn.utils._typedefs',
    '--hidden-import=sklearn.utils._heap',
    '--hidden-import=sklearn.utils._sorting',
    '--hidden-import=sklearn.neighbors._dist_metrics',
    '--hidden-import=sklearn.tree._partitioner',
    '--hidden-import=sklearn.tree._criterion',
    '--hidden-import=sklearn.tree._splitter',
    '--hidden-import=sklearn.tree._utils',
    '--hidden-import=sklearn.manifold._barnes_hut_tsne',
    '--hidden-import=sklearn.neighbors._quad_tree',
    '--hidden-import=sklearn.ensemble._base',
    '--hidden-import=sklearn.ensemble._forest',
    '--hidden-import=sklearn.tree',
    '--hidden-import=scipy.stats',
    '--hidden-import=scipy.sparse.csgraph._validation',
    '--hidden-import=scipy.special.cython_special'
]
pyinstaller_args.extend(hidden_imports)

# Exclude unnecessary modules to speed up build and prevent import conflicts
# These might not all be needed, but excluuding them anyway.
excludes = [
    '--exclude-module=tkinter',
    '--exclude-module=test',
    '--exclude-module=pytest',
    '--exclude-module=doctest',
    '--exclude-module=pdb',
    '--exclude-module=profile',
    '--exclude-module=cProfile',
    '--exclude-module=pstats',
    '--exclude-module=IPython',
    '--exclude-module=jupyter',
    '--exclude-module=notebook',
    '--exclude-module=spyder',
    '--exclude-module=sympy',
    '--exclude-module=sage',
    '--exclude-module=matplotlib.tests',
    '--exclude-module=scipy.tests',
    '--exclude-module=sklearn.tests',
    '--exclude-module=numpy.tests',
    '--exclude-module=PIL.tests',
    '--exclude-module=cv2.tests',
    '--exclude-module=pandas',
    '--exclude-module=statsmodels',
    '--exclude-module=seaborn',
    '--exclude-module=plotly',
    '--exclude-module=nltk',
    '--exclude-module=nltk.collocations',
    '--exclude-module=nltk.metrics',
    '--exclude-module=nltk.util',
    '--exclude-module=gensim',
    '--exclude-module=spacy',
    '--exclude-module=transformers',
    '--exclude-module=torch',
    '--exclude-module=tensorflow',
]
pyinstaller_args.extend(excludes)

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
        
    print(f"\n\nBuild completed! Executable is located at: {glob.glob(f'{output_dir}/Auto-Wall*')[0]}")

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
