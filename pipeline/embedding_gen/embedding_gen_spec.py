from tfx.types import component_spec, standard_artifacts

EMBEDDING_GEN_EXAMPLE_KEY = "examples"
EMBEDDING_GEN_MODEL_KEY = "model"


class EmbeddingGenSpec(component_spec.ComponentSpec):
    """ComponentSpec for Custom TFX Hello World Component."""

    PARAMETERS = {
        # These are parameters that will be passed in the call to
        # create an instance of this component.
    }
    INPUTS = {
        # This will be a dictionary with input artifacts, including URIs
        "examples": component_spec.ChannelParameter(type=standard_artifacts.Examples),
        "model": component_spec.ChannelParameter(type=standard_artifacts.Model),
    }
    OUTPUTS = {
        # This will be a dictionary which this component will populate
        #   'output_examples': component_spec.ChannelParameter(type=standard_artifacts.Examples),
    }
