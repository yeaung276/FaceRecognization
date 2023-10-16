from tfx import v1 as tfx
import tfx.orchestration.metadata
from pipeline.example_gen.component import TripletExampleGen



def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     metadata_path: str) -> tfx.dsl.Pipeline:
  """Creates a three component penguin pipeline with TFX."""
  # Brings data into the pipeline.
  example_gen = TripletExampleGen(input_base=data_root)

  # Following three components will be included in the pipeline.
  components = [example_gen]

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      metadata_connection_config=tfx.orchestration.metadata
      .sqlite_metadata_connection_config(metadata_path),
      components=components)