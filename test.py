import sys
try:
    import cv2
    print("cv2 import OK ->", cv2.__file__, "cv2", cv2.__version__)
except Exception as e:
    print("cv2 import ERROR:")
    import traceback; traceback.print_exc()
try:
    import numpy as np
    print("numpy", np.__version__)
except Exception as e:
    print("numpy import ERROR:", e)
try:
    import torch
    print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
except Exception as e:
    print("torch import ERROR:", e)
print("sys.executable:", sys.executable)