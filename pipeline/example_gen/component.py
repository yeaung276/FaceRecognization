from typing import Optional, Union

from tfx.dsl.components.base import executor_spec
from tfx.components.example_gen import component
from tfx.components.example_gen.base_example_gen_executor import BaseExampleGenExecutor
from tfx.dsl.placeholder import placeholder
from tfx.orchestration import data_types
from tfx.proto import example_gen_pb2
from tfx.proto import range_config_pb2

from pipeline.example_gen.transforms import _TripletTransform

class Executor(BaseExampleGenExecutor):
  """TFX triplet example gen executor."""

  def GetInputSourceToExamplePTransform(self):
    """Returns PTransform for Triplet TF examples."""
    return _TripletTransform
  
  
class TripletExampleGen(component.FileBasedExampleGen):
  """TFX TripletExampleGen component.

  The triplet examplegen component takes image data, and generates train
  and eval examples for downstream components.


  Component `outputs` contains:
   - `examples`: Channel of type `standard_artifacts.Examples` for output train
                 and eval examples.
  """

  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(Executor)

  def __init__(
      self,
      input_base: Optional[str] = None,
      input_config: Optional[Union[example_gen_pb2.Input, # type: ignore
                                   data_types.RuntimeParameter]] = None,
      output_config: Optional[Union[example_gen_pb2.Output, # type: ignore
                                    data_types.RuntimeParameter]] = None,
      range_config: Optional[Union[placeholder.Placeholder,
                                   range_config_pb2.RangeConfig, # type: ignore
                                   data_types.RuntimeParameter]] = None, 
      triplet_config: Optional[dict] = None,
      ):
      """Construct a TripletExampleGen component.

      Args:
        input_base: an external directory containing the CSV files.
        input_config: An example_gen_pb2.Input instance, providing input
          configuration. If unset, the files under input_base will be treated as a
          single split.
        output_config: An example_gen_pb2.Output instance, providing output
          configuration. If unset, default splits will be 'train' and 'eval' with
          size 2:1.
        range_config: An optional range_config_pb2.RangeConfig instance,
          specifying the range of span values to consider. If unset, driver will
          default to searching for latest span with no restrictions.
      """
      super().__init__(
          input_base=input_base,
          input_config=input_config,
          output_config=output_config,
          custom_config=triplet_config,
          range_config=range_config)