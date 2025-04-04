
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
        f.write(f"Error in NumPy hook: {e}\n")
