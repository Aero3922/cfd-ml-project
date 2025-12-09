# run_test.py — quick environment smoke test
import sys
try:
    import numpy as np
    import tensorflow as tf
    import meshio, pyvista, vtk, pyevtk, skimage
except Exception as e:
    print("IMPORT ERROR:", e)
    raise SystemExit(1)

print("python:", sys.executable)
print("numpy:", np.__version__)
print("tf:", tf.__version__)
print("TF devices:", tf.config.list_physical_devices())
print("meshio:", meshio.__version__)
print("pyvista:", pyvista.__version__)
print("vtk:", vtk.vtkVersion().GetVTKVersion())
print("pyevtk: import OK")
print("scikit-image:", skimage.__version__)
print("Environment smoke test: OK")
