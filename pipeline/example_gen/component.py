from typing import Optional, Union

from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.base import base_beam_component
from tfx.types import channel
from tfx.types import standard_artifacts

from pipeline.example_gen import executor
from pipeline.example_gen import component_specs


class TripletExampleGen(base_beam_component.BaseBeamComponent):
    """TFX TripletExampleGen component.

    The triplet examplegen component takes image data, and generates train
    and eval examples for downstream components.


    Component `outputs` contains:
     - `examples`: Channel of type `standard_artifacts.Examples` for output train
                   and eval examples.
    """

    SPEC_CLASS = component_specs.TripletGenComponentSpec
    EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(executor.Executor)

    def __init__(
        self,
        input_base: Optional[str] = None,
        eval_split_ratio: Optional[float] = None,
        sample_per_class: Optional[int] = 5
    ):
        """Construct a TripletExampleGen component.

        Args:
          input_base: an external directory containing the CSV files.
          sample_per_class: number of sample to choose per one class
          eval_split_ratio: split ratio for eval
        """
        example_artifacts = channel.Channel(type=standard_artifacts.Examples)

        spec = component_specs.TripletGenComponentSpec(
            input_base=input_base,
            eval_split_ratio=eval_split_ratio,
            sample_per_class=sample_per_class,
            examples=example_artifacts)
        super().__init__(spec=spec)
