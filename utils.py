import numpy as np
import tensorflow as tf
from keras.layers import Lambda

def reduce_resolution(ims, reduction_factor=2):
    O, M, N, P = ims.shape
    K = reduction_factor
    L = reduction_factor

    MK = M // K
    NL = N // L
    return ims.reshape(-1, MK, K, NL, L, P).mean(axis=(2, 4)).reshape(O, MK, NL, P)

'''SubpizelConv2D from https://github.com/twairball/keras-subpixel-conv/blob/master/subpixel.py'''
def SubpixelConv2D(scale=2):
    
    def subpixel(x):
        return tf.depth_to_space(x, scale)

    return Lambda(subpixel)

def InstanceNormalization():
    def IN(x):
        return tf.contrib.layers.instance_norm(x)
        
    return Lambda(IN)