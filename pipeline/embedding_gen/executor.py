from typing import Dict, List, Any
import os

from tfx.dsl.components.base import base_beam_executor
from tfx.types import artifact, artifact_utils
from tfx.components.util import tfxio_utils
from tfx.utils import io_utils
from tfx_bsl.tfxio import record_based_tfxio
from tfx_bsl.public.beam import run_inference
from tfx_bsl.public.proto import model_spec_pb2
import apache_beam as beam
from apache_beam.ml.inference.base import RunInference, PredictionResult
from apache_beam.ml.inference.base import KeyedModelHandler
from apache_beam.ml.inference.tensorflow_inference import TFModelHandlerTensor

import tensorflow as tf

from pipeline.embedding_gen.do_fns import ParseAndFlatTriplets, ToTripletExample
import pipeline.embedding_gen.embedding_gen_spec as embedding_specs

_TELEMETRY_DESCRIPTORS = ["EmbeddingGen"]
_EXAMPLES_FILE_NAME = "examples"


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
            - model: tensorflow model for inference
          output_dict: Output dict from output key to a list of Artifacts.
            - output: bulk inference results.
          exec_properties: A dict of execution properties.
            - model: standard artifact of imported model

        Returns:
          None
        """
        self._log_startup(input_dict, output_dict, exec_properties)

        if embedding_specs.EMBEDDING_GEN_EXAMPLE_KEY not in input_dict:
            raise ValueError("'examples' is missing in input dict.")

        if embedding_specs.EMBEDDING_GEN_MODEL_KEY not in input_dict:
            raise ValueError("'model' is missing in input dict.")
        model = artifact_utils.get_single_instance(
            input_dict[embedding_specs.EMBEDDING_GEN_MODEL_KEY]
        )
        output = artifact_utils.get_single_instance(
            output_dict[embedding_specs.EMBEDDING_GEN_OUTPUT_KEY]
        )

        self._run_model_inferance(
            input_dict["examples"],
            model,
            output, 
            exec_properties.get(embedding_specs.EMBEDDING_GEN_BATCH_KEY, None)
        )

    def _run_model_inferance(
        self,
        examples: List[artifact.Artifact],
        model: artifact.Artifact,
        output_examples: artifact.Artifact,
        batch_size: int | None
    ) -> None:  # type: ignore
        output_examples.split_names = examples[0].split_names  # type: ignore
        example_uris = {}
        for example_artifact in examples:
            for split in artifact_utils.decode_split_names(example_artifact.split_names):
                example_uris[split] = artifact_utils.get_split_uri([example_artifact],split)

        tfxio_factory = tfxio_utils.get_tfxio_factory_from_artifact(
            examples,
            _TELEMETRY_DESCRIPTORS,
            schema=None,
            read_as_raw_records=True,
            # We have to specify this parameter in order to create a RawRecord TFXIO
            # but we won't use the RecordBatches so the column name of the raw
            # records does not matter.
            raw_record_column_name="unused",
        )

        keyed_model_handler = KeyedModelHandler(TFModelHandlerTensor(model.uri, max_batch_size=batch_size))

        with self._make_beam_pipeline() as pipeline:
            for split, example_uri in example_uris.items():
                output_examples_split_uri = artifact_utils.get_split_uri(
                    [output_examples], split)
                tfxio = tfxio_factory([io_utils.all_files_pattern(example_uri)])
                assert isinstance(
                    tfxio, record_based_tfxio.RecordBasedTFXIO
                ), "Unable to use TFXIO {} as it does not support reading raw records.".format(
                    type(tfxio)
                )
                _ = (
                    pipeline
                    | f"ReadData[{split}]" >> tfxio.RawRecordBeamSource()
                    | f"ParseAndFlatten[{split}]" >> beam.ParDo(ParseAndFlatTriplets())
                    | f"Inference[{split}]" >> RunInference(keyed_model_handler)
                    | f"GroupBy[{split}]" >> beam.GroupByKey()
                    | f"ToTripletExample[{split}]" >> beam.ParDo(ToTripletExample())
                    | f"WriteTFRecord[{split}]"
                    >> beam.io.WriteToTFRecord(
                        os.path.join(output_examples_split_uri, _EXAMPLES_FILE_NAME),
                        file_name_suffix=".gz",
                        coder=beam.coders.ProtoCoder(tf.train.Example), # type: ignore
                    )
                )
