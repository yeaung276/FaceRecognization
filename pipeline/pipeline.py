from tfx.orchestration.pipeline import Pipeline
from tfx.orchestration import metadata
from tfx.proto import example_gen_pb2
from pipeline.example_gen.component import TripletExampleGen


def create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     metadata_path: str) -> Pipeline:
  """Creates a three component penguin pipeline with TFX."""
  # Brings data into the pipeline.
  example_gen = TripletExampleGen(
      input_base=data_root, 
      input_config=example_gen_pb2.Input(splits=[ # type: ignore
          example_gen_pb2.Input.Split(name='train', pattern='[0-2]'), # type: ignore
          example_gen_pb2.Input.Split(name='eval', pattern='[3-4]') # type: ignore
          ])
      )

  # Following three components will be included in the pipeline.
  components = [example_gen]

  return Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
      components=components) # type: ignore