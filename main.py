from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from pipeline.pipeline import create_pipeline
from utils.tf_record_reader import inspect_triplets
from utils.decode_tfrecords import decode_tfrecord
import tensorflow as tf

pipeline = create_pipeline('test', 'mocks/pipeline_root', 'mocks/example_gen', 'mocks/pipeline_metadata')
LocalDagRunner().run(pipeline)

# inspect_triplets('mocks/pipeline_root/TripletExampleGen/examples/53/Split-eval/data_tfrecord-00000-of-00001.gz')
# WIDTH = 112
# HEIGHT = 112
# mobile_net = tf.keras.applications.mobilenet.MobileNet(
#     input_shape=(WIDTH, HEIGHT, 3),
#     include_top=False,
#     weights='imagenet',
# )
# mobile_net.save('models/base_models')
# feature_description = {
#     'anchor': tf.io.FixedLenFeature([1024], tf.float32),
#     'positive': tf.io.FixedLenFeature([1024], tf.float32),
#     'negative': tf.io.FixedLenFeature([1024], tf.float32),
# }
# for i in decode_tfrecord('./mocks/pipeline_root/EmbeddingGen/output/81/Split-train/examples-00000-of-00001.gz', feature_description).batch(125):
#     print(i)
    # check if example 1 and example 2 has same embeddings values for anchor