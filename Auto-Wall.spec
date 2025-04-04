# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs

block_cipher = None

# Explicitly collect NumPy and its dependencies
numpy_imports = collect_submodules('numpy')
opencv_imports = collect_submodules('cv2')

# More comprehensive collection of scikit-learn modules
sklearn_imports = []
sklearn_imports.extend(collect_submodules('sklearn.cluster'))
sklearn_imports.extend(collect_submodules('sklearn.tree'))
sklearn_imports.extend(collect_submodules('sklearn.neighbors'))
sklearn_imports.extend(collect_submodules('sklearn.manifold'))
sklearn_imports.extend(collect_submodules('sklearn.utils'))
sklearn_imports.extend(collect_submodules('sklearn.metrics'))

scipy_imports = collect_submodules('scipy.stats')
scipy_imports.extend(collect_submodules('scipy.sparse'))
scipy_imports.extend(collect_submodules('scipy.special'))

# Create a list of packages to exclude - these are large and likely not needed
excludes = [
    'torch', 'tensorflow', 'transformers', 'IPython', 
    'jupyter', 'jedi', 'sphinx', 'spyder', 'nbconvert', 'nbformat',
    'lxml', 'docutils', 'timm', 'torchvision', 'nltk',
]

# Add additional data files needed by the application
extra_data_files = [
    ('src/gui/style.qss', 'src/gui')  # Make sure this path is correct
]

# Collect NumPy DLL files explicitly
numpy_dlls = collect_dynamic_libs('numpy')

# Create a runtime hook to fix NumPy docstring issues
runtime_hook_content = """
# Fix NumPy docstring issues
import os
import sys

try:
    import numpy
    import types

    # Patch add_docstring to handle non-string docstrings
    if hasattr(numpy.core.function_base, 'add_docstring'):
        original_add_docstring = numpy.core.function_base.add_docstring

        def patched_add_docstring(obj, docstring):
            if docstring is None:
                docstring = ""
            elif not isinstance(docstring, str):
                docstring = str(docstring)
            return original_add_docstring(obj, docstring)

        numpy.core.function_base.add_docstring = patched_add_docstring
except Exception as e:
    with open(os.path.join(sys._MEIPASS, "numpy_hook_error.log"), "w") as f:
        f.write(f"Error in NumPy hook: {e}\\n")
"""

# Create a startup hook that will help diagnose errors
startup_hook_content = """
# Startup diagnostics hook
import os
import sys
import traceback

# Redirect stdout and stderr to files in case console is hidden
log_dir = os.path.dirname(sys.executable)
sys.stdout = open(os.path.join(log_dir, "stdout.log"), "w")
sys.stderr = open(os.path.join(log_dir, "stderr.log"), "w")

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"sys.path: {sys.path}")
print(f"Working directory: {os.getcwd()}")
print(f"_MEIPASS: {getattr(sys, '_MEIPASS', 'Not in PyInstaller bundle')}")

# Set up global exception handler to log uncaught exceptions
def global_exception_handler(exctype, value, tb):
    error_msg = ''.join(traceback.format_exception(exctype, value, tb))
    print(f"UNCAUGHT EXCEPTION: {error_msg}")
    with open(os.path.join(log_dir, "uncaught_exception.log"), "w") as f:
        f.write(error_msg)
    sys.__excepthook__(exctype, value, tb)

sys.excepthook = global_exception_handler

# Try importing key packages to verify they're available
try:
    import numpy
    print(f"NumPy version: {numpy.__version__}")
except Exception as e:
    print(f"Error importing NumPy: {e}")

try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except Exception as e:
    print(f"Error importing OpenCV: {e}")

try:
    import sklearn
    print(f"scikit-learn version: {sklearn.__version__}")
except Exception as e:
    print(f"Error importing scikit-learn: {e}")

try:
    from PyQt6.QtWidgets import QApplication
    print("PyQt6 imported successfully")
except Exception as e:
    print(f"Error importing PyQt6: {e}")
"""

# Write the runtime hooks to files
runtime_hook_path = os.path.join(SPECPATH, 'numpy_hook.py')
with open(runtime_hook_path, 'w') as f:
    f.write(runtime_hook_content)

startup_hook_path = os.path.join(SPECPATH, 'startup_hook.py')
with open(startup_hook_path, 'w') as f:
    f.write(startup_hook_content)

a = Analysis(
    ['auto_wall.py'],
    pathex=['src'],
    binaries=numpy_dlls,
    datas=extra_data_files,
    hiddenimports=[
        'PyQt6.QtWidgets',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'numpy',
        'numpy.core.function_base',
        'numpy.core._multiarray_umath',
        'numpy.core._multiarray_tests',
        'numpy.core._dtype_ctypes',
        'numpy.random',
        'pandas',  # Add pandas explicitly since it's used by sklearn
        'sklearn.neighbors._partition_nodes',
        'sklearn.utils._typedefs',
        'sklearn.utils._heap',
        'sklearn.utils._sorting',
        'sklearn.cluster.k_means_',
        'sklearn.neighbors._kd_tree',
        'sklearn.neighbors._quad_tree',
        'sklearn.metrics.pairwise',
        'sklearn.tree._partitioner',
        'sklearn.tree._criterion',
        'sklearn.tree._splitter',
        'sklearn.tree._utils',
        'sklearn.manifold._t_sne',
        'sklearn.manifold._barnes_hut_tsne',
        'scipy.special',
        'scipy.integrate',
        'scipy.linalg',
        'scipy.sparse.linalg',
        'scipy.sparse.csgraph',
    ] + numpy_imports + opencv_imports + sklearn_imports + scipy_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[runtime_hook_path, startup_hook_path],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    optimize=1,  # Use moderate optimization for better performance
)

# Don't filter out NumPy or OpenCV binaries
def _filter_binary(x):
    return not any(name in x[0].lower() for name in 
        ['torch', 'tensorflow', 'cuda'])

a.binaries = [x for x in a.binaries if _filter_binary(x)]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Auto-Wall',
    debug=False,  # Disable debug mode for production
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Enable UPX compression for smaller executable size
    console=False,  # Hide the console window in production
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,  # Enable UPX compression for smaller executable size
    upx_exclude=[],
    name='Auto-Wall',
)
