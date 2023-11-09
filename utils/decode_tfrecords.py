import tensorflow as tf

##
# example schame
# feature_description = {
#     'anchor': tf.io.FixedLenFeature([1024], tf.float32),
#     'positive': tf.io.FixedLenFeature([1024], tf.float32),
#     'negative': tf.io.FixedLenFeature([1024], tf.float32),
# }
##

def decode_tfrecord(path: str, schema: dict):
    return tf.data.TFRecordDataset(path, compression_type='GZIP')\
        .map(lambda example: tf.io.parse_single_example(example, schema))