import numpy as np

def reduce_resolution(ims):
    O, M, N, P = ims.shape
    K = 2
    L = 2

    MK = M // K
    NL = N // L
    return ims.reshape(-1, MK, K, NL, L, P).mean(axis=(2, 4)).reshape(O, MK, NL, P)
