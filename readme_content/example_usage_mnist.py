from keras.models import load_model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load the model
generator = load_model('./models/generator_mnist.h5')

# Load MNIST images
(X_train, _), (_, _) = mnist.load_data()
X_train = np.reshape(X_train, (-1, 28, 28, 1))
X_train = X_train / 255.0

# Upsample the first five images
images = X_train[0:5]
upsampled_images = generator.predict(images)

# Create a figure showing the results
n_images = len(images)
fig, axes = plt.subplots(n_images, 2)
fig.suptitle("Original (left) vs. Upsampled (right)")
for ((ax1, ax2), original, upsampled) in zip(axes, images, upsampled_images):
	ax1.imshow(original[:, :, 0])
	ax2.imshow(upsampled[:, :, 0])
plt.show()