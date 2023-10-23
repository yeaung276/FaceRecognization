from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from pipeline.pipeline import create_pipeline
pipeline = create_pipeline('test', 'mocks/pipeline_root', 'mocks/example_gen', 'mocks/pipeline_metadata')

LocalDagRunner().run(pipeline)
# res = read_examplegen_output('mocks/pipeline_root/TripletExampleGen/examples/3/Split-train/data_tfrecord-00000-of-00001.gz')

