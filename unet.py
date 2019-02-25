from keras.layers import Input, Conv2D, Conv2DTranspose, \
                         concatenate, LeakyReLU, \
                         BatchNormalization, Activation, \
                         PReLU
from keras.regularizers import Regularizer

# local imports
from utils import InstanceNormalization

LEAKY_RELU_ALPHA = 0.2
BATCH_NORM_MOMENTUM = 0.8

# Attempting to generalize the Unet architecture in a way similar to 
# https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
def unet_cell(inputs,
              transpose_conv=False,
              num_filters=16,
              kernel_size=3,
              strides=1,
              padding='same', # Be careful modifying this default
              kernel_initializer='glorot_uniform',
              kernel_regularizer=None,
              activation='leaky_relu',
              batch_normalization=True, 
              instance_normalization=False, 
              conv_first=True):
    """2D Convolution -> Batch Normalization -> Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer

        ## Conv2D features:
        transpose_conv (bool): use Conv2DTranspose instead of Conv2D
        num_filters (int): number of filters used by Conv2D
        kernel_size (int): square kernel dimension
        strides (int): square stride dimension
        padding (str): one of 'same' or 'valid'
        kernel_initializer (string): method used to initialize kernel
        kernel_regularizer (keras.regularizers.Regularizer): 
            method used to constrain (regularize) kernel values or None
        
        ## Other cell features
        activation (string): name of activation function to be used or None
        batch_normalization (bool): whether to use batch normalization
        conv_first (bool): conv -> bn         -> activation, if True; 
                           bn   -> activation -> conv,       if False

    # Returns
        x (tensor): tensor as input to the next layer
    """

    # Validate arguments
    if kernel_regularizer is not None:
        if not isinstance(kernel_regularizer, Regularizer):
            raise TypeError("Argument `kernel_regularizer` must be "
                            "type %s or None." % repr(Regularizer))

    # Determine which convolutional layer to use:
    if transpose_conv:
        conv = Conv2DTranspose(num_filters,
                               kernel_size=kernel_size, # TODO: Check what kernel_size, strides to use
                               strides=strides,
                               padding=padding,
                               kernel_initializer=kernel_initializer,
                               kernel_regularizer=kernel_regularizer)
    else:
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding, # This is not optional for our Unet.
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)

    # Determine which activation function to use:
    if isinstance(activation, str):
        if activation.lower() == 'leaky_relu':
            activation_fn = LeakyReLU(alpha=LEAKY_RELU_ALPHA)
        elif activation.lower() == 'prelu':
            activation_fn = PReLU()
        else:
            activation_fn = Activation(activation)
    else:
        activation_fn = None

    x = inputs
    if conv_first:
        x = conv(x)
        if activation_fn is not None:
            x = activation_fn(x)
        if batch_normalization:
            x = BatchNormalization(momentum=BATCH_NORM_MOMENTUM)(x)
        if instance_normalization:
            x = InstanceNormalization()(x)
    else:
        if activation_fn is not None:
            x = activation_fn(x)
        if batch_normalization:
            x = BatchNormalization(momentum=BATCH_NORM_MOMENTUM)(x)
        if instance_normalization:
            x = InstanceNormalization()(x)
        x = conv(x)
    return x
