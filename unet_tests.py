# Ensure that our Unet class produces the expected network

# TODO: Finish Unet API, then supply test-case parameters.

from keras.models import Model
from keras.layers import Input
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
for i in range(20):
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