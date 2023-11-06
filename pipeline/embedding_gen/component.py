from typing import Optional

from tfx.types import standard_artifacts, channel, channel_utils
from tfx.dsl.components.base import base_beam_component, executor_spec

from pipeline.embedding_gen import executor
from pipeline.embedding_gen import embedding_gen_spec


class EmbeddingGen(base_beam_component.BaseBeamComponent):
    """Custom TFX EmbeddingGen Component."""

    SPEC_CLASS = embedding_gen_spec.EmbeddingGenSpec
    EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(executor.Executor)

    def __init__(
        self,
        examples: channel.Channel,
        model: channel.BaseChannel,
        output: Optional[channel.Channel] = None,
    ):
        """Construct an EmbeddingGen component.

        Args:
          examples: A BaseChannel of type `standard_artifacts.Examples`, usually
            produced by an ExampleGen component. _required_
          model: A BaseChannel of type `standard_artifacts.Model`, usually produced
            by a Trainer component. __required__
          output: A BasedChannel of type `standard_artifacts.Examples`, for saving generated
          embeddings.
        """
        if not output:
            examples_artifact = standard_artifacts.Examples()
            output_data = channel_utils.as_channel([examples_artifact])
        spec = embedding_gen_spec.EmbeddingGenSpec(
            examples=examples, model=model, output=output_data  # type: ignore
        )
        super().__init__(spec=spec)
