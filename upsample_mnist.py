from keras.models import Model
from keras.layers import Input
from keras.optimizers import *
from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt

# Local imports
from builders import build_generator, build_discriminator

def reduce_resolution(ims):
    O, M, N, P = ims.shape
    K = 2
    L = 2

    MK = M // K
    NL = N // L
    return ims.reshape(-1, MK, K, NL, L).mean(axis=(2, 4)).reshape(O, MK, NL, P)

def upsize(ims):
    return ims.repeat(2, axis = 1).repeat(2, axis = 2)

if __name__ == "__main__":
	(X_train, _), (_, _) = mnist.load_data()
	X_train = np.reshape(X_train, (-1, 28, 28, 1))/255

	image_size = 28

	optimizer = Adam(0.001, 0.5)
	generator = build_generator(image_size)
	generator.compile(loss='binary_crossentropy', optimizer=optimizer)

	optimizer1 = Adam(0.0002, 0.5)
	discriminator = build_discriminator(image_size)
	discriminator.compile(loss='binary_crossentropy', optimizer=optimizer1, metrics=['accuracy'])

	discriminator.trainable = False

	# The generator takes noise as input and generated imgs
	z = Input(shape=(None, None, 1))
	img = generator(z)
	# The discriminator takes generated images as input and determines validity
	valid = discriminator([z, img])

	# The combined model  (stacked generator and discriminator) takes
	# noise as input => generates images => determines validity 
	combined = Model(z, [valid, img])
	combined.compile(loss=['binary_crossentropy', 'mean_absolute_error'], optimizer=optimizer, metrics = ['accuracy'])

	combined.summary()

	epochs = 2
	batch_size = 2
	half_batch = int(batch_size/2)

	for epoch in range(epochs):

	    # ---------------------
	    #  Train Discriminator
	    # ---------------------

	    # Select a random half batch of images
	    idx = np.random.randint(0, X_train.shape[0], half_batch)
	    imgs = X_train[idx]
	    noise = reduce_resolution(imgs)

	    idx1 = np.random.randint(0, X_train.shape[0], half_batch)
	    imgs1 = X_train[idx1]
	    noise1 = reduce_resolution(imgs1)

	    # Generate a half batch of new images
	    gen_imgs = generator.predict(noise1)
	    # Train the discriminator
	    
	    d_loss_real = discriminator.train_on_batch([noise, imgs], np.ones((half_batch, 1)))
	    d_loss_fake = discriminator.train_on_batch([noise1, gen_imgs], np.zeros((half_batch, 1)))
	    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

	    # ---------------------
	    #  Train Generator
	    # ---------------------

	    idx2 = np.random.randint(0, X_train.shape[0], batch_size)
	    noise2 = reduce_resolution(X_train[idx2])

	    # The generator wants the discriminator to label the generated samples
	    # as valid (ones)
	    valid_y = np.array([1] * batch_size)
	    
	    # Train the generator
	    g_loss = combined.train_on_batch(noise2, [valid_y, X_train[idx2]])
	        
	    if epoch % 10 == 0:
	        # Plot the progress
	        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))
	        fig, ax = plt.subplots(1, 3)
	        ax[0].imshow(np.reshape(noise1[0], (14, 14)))
	        ax[1].imshow(np.reshape(gen_imgs[0], (28, 28)))
	        ax[2].imshow(np.reshape(imgs1[0], (28, 28)))
	        plt.show()

	generator.save('./models/generator_mnist.h5')
	discriminator.save('./models/discriminator_mnist.h5')
	combined.save('./models/combined_mnist.h5')
