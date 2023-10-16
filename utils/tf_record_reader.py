import tensorflow as tf


def read_tf_record(path: str, sample = 10):
    files = tf.io.gfile.glob(path)
    raw_dataset = tf.data.TFRecordDataset(files)
    for raw_record in raw_dataset.take(sample):
        print(raw_record)
    