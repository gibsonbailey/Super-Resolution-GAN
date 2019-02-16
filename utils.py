import numpy as np

def reduce_resolution(ims, reduction_factor=2):
    O, M, N, P = ims.shape
    K = reduction_factor
    L = reduction_factor

    MK = M // K
    NL = N // L
    return ims.reshape(-1, MK, K, NL, L, P).mean(axis=(2, 4)).reshape(O, MK, NL, P)
