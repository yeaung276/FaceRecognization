from tfx.orchestration.pipeline import Pipeline
from tfx.dsl.components.common.importer import Importer
from tfx.types import standard_artifacts
from tfx.orchestration import metadata
from tfx.proto import example_gen_pb2
from tfx.types.channel import Channel

from pipeline.example_gen import TripletExampleGen
from pipeline.embedding_gen import EmbeddingGen

def create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     metadata_path: str) -> Pipeline:
  """Creates a triplet pipeline with TFX."""
  # Brings data into the pipeline.
#   example_gen = TripletExampleGen(
#       input_base=data_root, 
#       input_config=example_gen_pb2.Input(splits=[ # type: ignore
#           example_gen_pb2.Input.Split(name='train', pattern='[0-2]'), # type: ignore
#           example_gen_pb2.Input.Split(name='eval', pattern='[3-4]') # type: ignore
#           ])
#       )
  
  # Convert it into encodings
  encoding_gen = EmbeddingGen(
      examples=Importer(source_uri='mocks/pipeline_root/TripletExampleGen/examples/1', artifact_type=standard_artifacts.Examples),
      model=Importer(source_uri='models/encoder', artifact_type=standard_artifacts.Model)
    )

  # Following three components will be included in the pipeline.
  components = [encoding_gen]

  return Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
      components=components) # type: ignore