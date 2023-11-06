import tensorflow as tf

WIDTH = 112
HEIGHT = 112

mobile_net = tf.keras.applications.mobilenet.MobileNet(
    input_shape=(WIDTH, HEIGHT, 3),
    include_top=False,
    weights='imagenet',
)
encoder = tf.keras.models.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Flatten()
])
encoder.save('models/base-models/mobile-net')