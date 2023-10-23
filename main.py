from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from pipeline.pipeline import create_pipeline
from utils.tf_record_reader import inspect_triplets

# pipeline = create_pipeline('test', 'mocks/pipeline_root', 'mocks/example_gen', 'mocks/pipeline_metadata')
# LocalDagRunner().run(pipeline)

# inspect_triplets('mocks/pipeline_root/TripletExampleGen/examples/17/Split-train/data_tfrecord-00000-of-00001.gz')

