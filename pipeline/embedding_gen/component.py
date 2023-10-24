from tfx.types import standard_artifacts
from tfx.dsl.components.base import base_beam_component, executor_spec
from tfx.types import channel

from pipeline.embedding_gen import executor
from pipeline.embedding_gen import embedding_gen_spec


class EmbeddingGen(base_beam_component.BaseBeamComponent):
    """Custom TFX EmbeddingGen Component."""

    SPEC_CLASS = embedding_gen_spec.EmbeddingGenSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self, examples: channel.BaseChannel, model: channel.BaseChannel):
        """Construct an EmbeddingGen component.

        Args:
          examples: A BaseChannel of type `standard_artifacts.Examples`, usually
            produced by an ExampleGen component. _required_
          model: A BaseChannel of type `standard_artifacts.Model`, usually produced
            by a Trainer component.
        """

        spec = embedding_gen_spec.EmbeddingGenSpec(examples=examples, model=model)
        super().__init__(spec=spec)
