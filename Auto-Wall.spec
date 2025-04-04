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

# Create a list of packages to exclude - these are large and likely not needed
excludes = [
    'torch', 'tensorflow', 'transformers', 'pandas', 'notebook', 'IPython', 
    'jupyter', 'jedi', 'sphinx', 'spyder', 'nbconvert', 'nbformat',
    'lxml', 'docutils', 'pyarrow', 'timm', 'torchvision', 'nltk',
    'datasets', 'lightning', 'pydantic'
]

# Add additional data files needed by the application
extra_data_files = [
    ('src/gui/style.qss', 'src/gui')  # Make sure this path is correct
]

# Collect NumPy DLL files explicitly
numpy_dlls = collect_dynamic_libs('numpy')

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
        'numpy.core._multiarray_umath',
        'numpy.core._multiarray_tests',
        'numpy.core._dtype_ctypes',
        'numpy.random',
        'sklearn.neighbors._partition_nodes',
        'sklearn.utils._typedefs',
        'sklearn.utils._heap',
        'sklearn.utils._sorting',
        'sklearn.cluster.k_means_',
        'sklearn.neighbors._kd_tree',
        'sklearn.metrics.pairwise',
        'sklearn.tree._partitioner',  # Add the missing module
        'sklearn.tree._criterion',
        'sklearn.tree._splitter',
        'sklearn.tree._utils',
        'sklearn.manifold._t_sne',
        'sklearn.manifold._barnes_hut_tsne',
        'sklearn.neighbors._quad_tree',
        'scipy.special',
        'scipy.integrate',
    ] + numpy_imports + opencv_imports + sklearn_imports + scipy_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Don't filter out NumPy or OpenCV binaries
def _filter_binary(x):
    return not any(name in x[0].lower() for name in 
        ['torch', 'tensorflow', 'cuda', 'mkl'])

a.binaries = [x for x in a.binaries if _filter_binary(x)]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Auto-Wall',
    debug=True,  # Enable debug mode
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Disable UPX compression to avoid potential issues
    console=True,  # Keep console window open to see errors
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
    upx=False,  # Disable UPX compression to avoid potential issues
    upx_exclude=[],
    name='Auto-Wall',
)
