
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter
from tfx.types.component_spec import ChannelParameter
from tfx.types import standard_artifacts

INPUT_BASE_KEY = 'input_base'
EXAMPLES_KEY = 'examples'
SAMPLE_PER_CLASS = 'sample_per_class'
EVAL_SPLIT_RATIO = 'eval_split_ratio'


class TripletGenComponentSpec(ComponentSpec):
    """File-based ExampleGen component spec."""

    PARAMETERS = {
        INPUT_BASE_KEY: ExecutionParameter(type=str),
        SAMPLE_PER_CLASS: ExecutionParameter(type=int),
        EVAL_SPLIT_RATIO: ExecutionParameter(type=float)
    }
    INPUTS = {}
    OUTPUTS = {
      EXAMPLES_KEY: ChannelParameter(type=standard_artifacts.Examples),
    }
    
    