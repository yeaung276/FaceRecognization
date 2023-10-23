import tensorflow as tf
import matplotlib.pyplot as plt


def read_examplegen_output(path: str, compression_type = 'GZIP', sample=10):
    files = tf.io.gfile.glob(path)
    raw_dataset = tf.data.TFRecordDataset(files, compression_type=compression_type)
    for raw_record in raw_dataset.shuffle(sample).take(sample):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy()) # type: ignore
        result = {}
        # example.features.feature is the dictionary
        for key, feature in example.features.feature.items():
            # The values are the Feature objects which contain a `kind` which contains:
            # one of three fields: bytes_list, float_list, int64_list

            kind = feature.WhichOneof('kind')
            result[key] = tf.io.parse_tensor(getattr(feature, kind).value[0], out_type=tf.uint8)
        yield result
        
def inspect_triplets(root: str):
    train_examples = list(read_examplegen_output(root))
    fig, axs = plt.subplots(len(train_examples), 3)
    for idx, example in enumerate(train_examples):
        axs[idx, 0].imshow(example['positive'].numpy())
        axs[idx, 0].set_xticks([])
        axs[idx, 0].set_yticks([])

        axs[idx, 1].imshow(example['anchor'].numpy())
        axs[idx, 1].set_xticks([])
        axs[idx, 1].set_yticks([])

        axs[idx, 2].imshow(example['negative'].numpy())
        axs[idx, 2].set_xticks([])
        axs[idx, 2].set_yticks([])
    plt.show()
    
    