import logging
import math
from typing import Dict, Any, List, Tuple

import tensorflow as tf

from tfx.types import artifact
from tfx.types import artifact_utils
from tfx.proto import example_gen_pb2
from tfx.dsl.io import fileio
from tfx.dsl.io.filesystem import PathType
from tfx.dsl.components.base import base_beam_executor
from tfx.components.example_gen import write_split
from tfx.components.util import examples_utils

import apache_beam as beam

from pipeline.example_gen import component_specs
from pipeline.example_gen import transforms

SPLITS = ['train', 'eval']


class Executor(base_beam_executor.BaseBeamExecutor):
    """TFX triplet example gen executor."""
    
    def Do(
      self,
      input_dict: Dict[str, List[artifact.Artifact]],
      output_dict: Dict[str, List[artifact.Artifact]],
      exec_properties: Dict[str, Any],
    ) -> None:
        """Take input data source and generates serialized data splits.

        Args:
        output_dict: Output dict from output key to a list of Artifacts.
            - examples: splits of serialized records.
        exec_properties: A dict of execution properties.
            - input_base: an external directory containing the data files.
            - sample_per_class: number of sample per class to choose from.
            - eval_split_ratio: portion to split for eval dataset

        Returns:
        None
        """
        self._log_startup(input_dict, output_dict, exec_properties)


        examples_artifact = artifact_utils.get_single_instance(
            output_dict[component_specs.EXAMPLES_KEY])
        examples_artifact.split_names = artifact_utils.encode_split_names(SPLITS)
        input_base_uri = exec_properties.get(component_specs.INPUT_BASE_KEY)
        sample_per_class = exec_properties.get(component_specs.SAMPLE_PER_CLASS, 5)
        eval_split_ratio = exec_properties.get(component_specs.EVAL_SPLIT_RATIO, 0.5)

        classes = fileio.glob(f"{input_base_uri}/*")
        
        logging.info(f'{len(classes)} found. Generating examples with {sample_per_class} per folder.')
        with self._make_beam_pipeline() as pipeline:
            for split_name, paths in self.get_folder_splits(classes, eval_split_ratio):
                _ = (
                    pipeline
                    | f'InputToRecord[{split_name}]' >> transforms.TripletTransform(paths, sample_per_class)
                    | f'WriteSplit[{split_name}]' >> write_split.WriteSplit(
                        artifact_utils.get_split_uri([examples_artifact], split_name),
                        example_gen_pb2.FORMAT_TFRECORDS_GZIP
                        )
                    )


        for output_examples_artifact in output_dict[component_specs.EXAMPLES_KEY]:
            examples_utils.set_payload_format(
                output_examples_artifact, example_gen_pb2.FORMAT_TF_EXAMPLE)

        for output_examples_artifact in output_dict[component_specs.EXAMPLES_KEY]:
            examples_utils.set_file_format(
                output_examples_artifact,
                write_split.to_file_format_str(example_gen_pb2.FORMAT_TFRECORDS_GZIP))

        logging.info('Examples generated.')
    
    def get_folder_splits(self, folders: List[PathType], eval_ratio: float) -> List[Tuple[str, List[PathType]]]:
        train_cutoff = math.floor(len(folders) * (1-eval_ratio))
        assert (
            len(folders[:train_cutoff]) > 1
        ), f"excepted to have minimum folder of 2 to generate triplet instead get {len(folders[:train_cutoff])} folders.(Train split)"
        assert (
            len(folders[train_cutoff:]) > 1
        ), f"excepted to have minimum folder of 2 to generate triplet instead get {len(folders[train_cutoff:])} folders.(Eval split)"
        return [
            ('train', folders[:train_cutoff]),
            ('eval', folders[train_cutoff:])
            ]
