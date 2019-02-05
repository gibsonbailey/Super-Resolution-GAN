from keras.layers import Input
from keras.utils import plot_model

# Local import
from builders import build_upscaler_v2

if __name__ == "__main__":
    x = Input(shape=(14, 14, 1))
    model = build_upscaler_v2(x, output_size=(28, 28),
        num_cells_in_layer=[3, 3, 3, 3],
        bottleneck_before_concat=False)
    plot_model(model,
        to_file="model_images/upscaler_v2.png",
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB')
    model.summary()