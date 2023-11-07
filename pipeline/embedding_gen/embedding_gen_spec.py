from tfx.types import component_spec, standard_artifacts

EMBEDDING_GEN_EXAMPLE_KEY = "examples"
EMBEDDING_GEN_MODEL_KEY = "model"
EMBEDDING_GEN_OUTPUT_KEY = "output"
EMBEDDING_GEN_BATCH_KEY = "batch_size"


class EmbeddingGenSpec(component_spec.ComponentSpec):
    """ComponentSpec for Custom TFX Hello World Component."""

    PARAMETERS = {
        # These are parameters that will be passed in the call to
        # create an instance of this component.
        EMBEDDING_GEN_BATCH_KEY: component_spec.ExecutionParameter(type=int, optional=True)
    }
    INPUTS = {
        # This will be a dictionary with input artifacts, including URIs
        EMBEDDING_GEN_EXAMPLE_KEY: component_spec.ChannelParameter(type=standard_artifacts.Examples),
        EMBEDDING_GEN_MODEL_KEY: component_spec.ChannelParameter(type=standard_artifacts.Model),
    }
    OUTPUTS = {
        # This will be a dictionary which this component will populate
        EMBEDDING_GEN_OUTPUT_KEY: component_spec.ChannelParameter(type=standard_artifacts.Examples),
    }
