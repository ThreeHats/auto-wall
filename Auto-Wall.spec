# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['auto_wall.py'],
    pathex=['src'],
    binaries=[],
    datas=[('src/gui/style.qss', 'src/gui/'), ('resources', 'resources/')],
    hiddenimports=['sklearn.neighbors._partition_nodes', 'sklearn.utils._typedefs', 'sklearn.utils._heap', 'sklearn.utils._sorting', 'sklearn.neighbors._dist_metrics', 'sklearn.tree._partitioner', 'sklearn.tree._criterion', 'sklearn.tree._splitter', 'sklearn.tree._utils', 'sklearn.manifold._barnes_hut_tsne', 'sklearn.neighbors._quad_tree', 'sklearn.ensemble._base', 'sklearn.ensemble._forest', 'sklearn.tree', 'scipy.stats', 'scipy.sparse.csgraph._validation', 'scipy.special.cython_special'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'test', 'pytest', 'doctest', 'pdb', 'profile', 'cProfile', 'pstats', 'IPython', 'jupyter', 'notebook', 'spyder', 'sympy', 'sage', 'matplotlib.tests', 'scipy.tests', 'sklearn.tests', 'numpy.tests', 'PIL.tests', 'cv2.tests', 'pandas', 'statsmodels', 'seaborn', 'plotly', 'nltk', 'nltk.collocations', 'nltk.metrics', 'nltk.util', 'gensim', 'spacy', 'transformers', 'torch', 'tensorflow'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Auto-Wall',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:\\Users\\Noah\\Documents\\GitHub\\auto-wall\\resources\\icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Auto-Wall',
)
