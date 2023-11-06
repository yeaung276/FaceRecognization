from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from pipeline.pipeline import create_pipeline
from utils.tf_record_reader import inspect_triplets
# import tensorflow as tf

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