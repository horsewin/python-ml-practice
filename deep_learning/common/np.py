# -*- coding: utf-8 -*-
GPU = False

if GPU:
    # import cupy as np
    print("NO CUDA:{}".format(GPU))
else:
    import numpy as np

