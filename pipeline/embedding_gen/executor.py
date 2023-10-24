from typing import Dict, List, Any

from tfx.dsl.components.base import base_beam_executor
from tfx.types import artifact


class Executor(base_beam_executor.BaseBeamExecutor):
    """Executor for EmbeddingGenComponent."""

    def Do(
        self,
        input_dict: Dict[str, List[artifact.Artifact]],
        output_dict: Dict[str, List[artifact.Artifact]],
        exec_properties: Dict[str, Any],
    ) -> None:
        """Runs batch inference on a given model with given input examples.

        Args:
          input_dict: Input dict from input key to a list of Artifacts.
            - examples: examples for inference.
          output_dict: Output dict from output key to a list of Artifacts.
            - output: bulk inference results.
          exec_properties: A dict of execution properties.
            - model: standard artifact of imported model

        Returns:
          None
        """
        self._log_startup(input_dict, output_dict, exec_properties)
