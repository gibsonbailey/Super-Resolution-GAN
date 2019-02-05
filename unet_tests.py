# Ensure that our Unet class produces the expected network

# TODO: Finish Unet API, then supply test-case parameters.

from keras.models import Model
from keras.layers import Input, UpSampling2D, concatenate, Conv2D
from keras.utils import plot_model
from keras.regularizers import l1, l2, l1_l2

import numpy as np

# Local imports
from unet import unet_cell

unet_cell_input_options = dict(
    transpose_conv=[True, False],
    num_filters=[1, 2, 3, 16, 32, 64],
    kernel_size=[1, 2, 3, 4, 5],
    strides=[1, 2, 3, 4, 5],
    kernel_initializer=['he_normal', 'glorot_uniform'],
    kernel_regularizer=[l1(0.01), l2(0.01), l1_l2(l1=0.01, l2=0.01)],
    activation=['leaky_relu', 'relu', 'tanh', 'elu'],
    batch_normalization=[True, False],
    conv_first=[True, False]
)

def generate_kwargs(options):
    kwargs = dict()
    for key, options in options.items():
        choice = np.random.choice(len(options))
        kwargs[key] = options[choice]
    return kwargs

# Randomly choose values for each option, then plot the resulting model
# Note: redirect the output of this script to a text file to save the parameters
def test_unet_cell_kwargs(num_tests):
    for i in range(num_tests):
        kwargs = generate_kwargs(unet_cell_input_options)

        print('-'*50)
        print(f"Sample key-word args {i}:")
        for key, val in kwargs.items():
            print(f"    {key}={repr(val)}")

        print(f"\nSaving as model{i}.png")

        x = Input(shape=(28, 28, 1))
        y = unet_cell(x, **kwargs)
        model = Model(inputs=x, outputs=y)

        plot_model(model,
               to_file=f"model_images/model{i}.png",
               show_shapes=True,
               show_layer_names=True,
               rankdir='TB' # Create a vertical plot
        )

        print('='*50)

def unet_constructor(image_shape, num_layers, num_cells_per_layer, initial_num_filters=16):
    n = initial_num_filters # Rename for brevity

    inputs = Input(shape=image_shape)
    x = UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='nearest')(inputs)

    # Contraction
    carry_forward_tensors = dict()
    for i in range(num_layers - 1):
        if i == 0: # First layer is special case
            for j in range(num_cells_per_layer):
                x = unet_cell(x, num_filters=n, kernel_size=3, strides=1)
        else:
            x = unet_cell(x, num_filters=n*2**i, kernel_size=3, strides=2) # First cell has stride 2
            for j in range(num_cells_per_layer - 1):
                x = unet_cell(x, num_filters=n*2**i, kernel_size=3, strides=1)
        carry_forward_tensors[i] = x

    # Bottom layer
    x = unet_cell(x, num_filters=n*2**(i+1), kernel_size=3, strides=2)
    for j in range(num_cells_per_layer - 1):
        x = unet_cell(x, num_filters=n*2**(i+1), kernel_size=3, strides=1)

    # Expansion
    for i in reversed(range(num_layers - 1)):
        x = unet_cell(x, transpose_conv=True, num_filters=n*2**i, kernel_size=3, strides=2)
        x = concatenate([carry_forward_tensors[i], x])
        for j in range(num_cells_per_layer - 1):
            x = unet_cell(x, num_filters=n*2**i, kernel_size=3, strides=1)

    # Output
    x = Conv2D(filters=1, kernel_size=1, strides=1, padding='same', activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    plot_model(model,
       to_file=f"model_images/unet_constructor_{num_layers}_{num_cells_per_layer}.png",
       show_shapes=True,
       show_layer_names=True,
       rankdir='TB' # Create a vertical plot
    )

def unet_parametrized_constructor(inputs, 
                                  num_filters_in_layer, 
                                  num_cells_in_layer,
                                  bottleneck_before_concat=False):
    x = inputs

    carry_forward_tensors = []

    assert len(num_filters_in_layer) == len(num_cells_in_layer)

    # Zip the parameters that apply to the contraction-side (exclude last 
    # parameter for bottom layer)
    contraction_filters_and_cells = zip(num_filters_in_layer[:-1],
                                        num_cells_in_layer[:-1])

    # Contraction
    for i, (num_filters, num_cells) in enumerate(contraction_filters_and_cells):
        if i == 0:
            # First layer does not downsample
            for j in range(num_cells):
                x = unet_cell(x, num_filters=num_filters) 
                # Note: We make extensive use of default values.
        else:
            # Downsample first
            x = unet_cell(x, num_filters=num_filters, strides=2)
            for j in range(num_cells - 1):
                x = unet_cell(x, num_filters=num_filters)
        carry_forward_tensors.append(x)

    # Bottom layer
    num_filters_at_bottom = num_filters_in_layer[-1]
    num_cells_at_bottom = num_cells_in_layer[-1]

    x = unet_cell(x, num_filters=num_filters_at_bottom, strides=2)
    for i in range(num_cells_at_bottom - 1):
        x = unet_cell(x, num_filters=num_filters_at_bottom)

    # Expansion
    filters_cells_and_carrys = reversed(list(zip(num_filters_in_layer[:-1],
                                                 num_cells_in_layer[:-1],
                                                 carry_forward_tensors)))
    for num_filters, num_cells, carry in filters_cells_and_carrys:
        x = unet_cell(x, transpose_conv=True, num_filters=num_filters, strides=2)
        if bottleneck_before_concat:
            x = unet_cell(x, num_filters=num_filters//2, kernel_size=1)
            carry = unet_cell(carry, num_filters=num_filters//2, kernel_size=1)
        x = concatenate([x, carry])
        for _ in range(num_cells - 1):
            x = unet_cell(x, num_filters=num_filters)

    # Output
    x = unet_cell(x, num_filters=1, kernel_size=1, activation='sigmoid',
                  batch_normalization=False)

    model = Model(inputs=inputs, outputs=x)
    plot_model(model,
       to_file=f"model_images/unet_parametrized_constructor.png",
       show_shapes=True,
       show_layer_names=True,
       rankdir='TB' # Create a vertical plot
    )

if __name__=="__main__":

    num_filters_in_layer = [2**i for i in range(4, 9)]
    num_cells_in_layer = [3] * len(num_filters_in_layer)

    unet_parametrized_constructor(
        Input(shape=(28*4, 28*4, 1)),
        num_filters_in_layer,
        num_cells_in_layer,
        bottleneck_before_concat=True
    )
