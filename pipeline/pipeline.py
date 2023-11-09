from tfx.proto import trainer_pb2
from tfx.orchestration.pipeline import Pipeline
from tfx.dsl.components.common.importer import Importer
from tfx.types import standard_artifacts
from tfx.types import channel
from tfx.orchestration import metadata
from tfx.components import Trainer

from pipeline.example_gen import TripletExampleGen
from pipeline.embedding_gen import EmbeddingGen

def create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     metadata_path: str) -> Pipeline:
  """Creates a triplet pipeline with TFX."""
  # # Brings data into the pipeline.
  # example_gen = TripletExampleGen(
  #     input_base=data_root, 
  #     sample_per_class=10, 
  #     eval_split_ratio=0.5
  #     )
  
  # # Convert it into encodings
  # model = Importer(
  #   source_uri='models/base-models/mobile-net',
  #   artifact_type=standard_artifacts.Model
  # ).with_id('model_importer')
  
  # encoding_gen = EmbeddingGen(
  #     examples=example_gen.outputs['examples'],
  #     model=model.outputs['result'],
  #     batch_size=500
  #   )
  hypermeter = Importer(
    source_uri='pipeline/trainer/hyperparameters',
    artifact_type=standard_artifacts.HyperParameters
  )

  examples = Importer(
      source_uri='mocks/pipeline_root/EmbeddingGen/output/81', 
      artifact_type=standard_artifacts.Examples,
      properties={'split_names': '["train", "eval"]'}
    ).with_id('example_importer')

  trainer = Trainer(
      module_file='./pipeline/trainer/module_file.py',
      examples=examples.outputs['result'],
      hyperparameters=hypermeter.outputs['result'],
      train_args=trainer_pb2.TrainArgs(num_steps=100), # type: ignore
      eval_args=trainer_pb2.EvalArgs(num_steps=5)) # type: ignore

  # Following three components will be included in the pipeline.
  components = [hypermeter, examples, trainer]

  return Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
      components=components) # type: ignore