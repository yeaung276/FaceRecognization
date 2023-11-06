from tfx.orchestration.pipeline import Pipeline
from tfx.dsl.components.common.importer import Importer
from tfx.types import standard_artifacts
from tfx.orchestration import metadata
from tfx.proto import example_gen_pb2
from tfx.types.channel import Channel
from tfx.types import channel_utils
from typing import Any

from pipeline.example_gen import TripletExampleGen
from pipeline.embedding_gen import EmbeddingGen

def create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     metadata_path: str) -> Pipeline:
  """Creates a triplet pipeline with TFX."""
  # Brings data into the pipeline.
  example_gen = TripletExampleGen(
      input_base=data_root, 
      sample_per_class=10, 
      eval_split_ratio=0.5
      )
  
  # Convert it into encodings
  model = Importer(
    source_uri='models/base-models/mobile-net',
    artifact_type=standard_artifacts.Model
  ).with_id('model_importer')
  
  encoding_gen = EmbeddingGen(
      examples=example_gen.outputs['examples'],
      model=model.outputs['result']
    )

  # Following three components will be included in the pipeline.
  components = [example_gen, model, encoding_gen]

  return Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
      components=components) # type: ignore