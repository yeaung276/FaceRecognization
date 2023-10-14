
import random
import logging

from typing import Dict, Any

from tfx.types import standard_component_specs
from tfx.dsl.io import fileio
import apache_beam as beam
import tensorflow as tf 


#     # Read each CSV file while maintaining order. This is done in order to group
#     # together multi-line string fields.
#     parsed_csv_lines = (
#         pipeline
#         | 'CreateFilenames' >> beam.Create(csv_files)
#         | 'ReadFromText' >> beam.ParDo(_ReadCsvRecordsFromTextFile())
#         | 'ParseCSVLine' >> beam.ParDo(csv_decoder.ParseCSVLine(delimiter=','))
#         | 'ExtractParsedCSVLines' >> beam.Keys())
#     column_infos = beam.pvalue.AsSingleton(
#         parsed_csv_lines
#         | 'InferColumnTypes' >> beam.CombineGlobally(
#             csv_decoder.ColumnTypeInferrer(column_names, skip_blank_lines=True))
#     )

#     return (parsed_csv_lines
#             |
#             'ToTFExample' >> beam.ParDo(_ParsedCsvToTfExample(), column_infos))


# class Executor(BaseExampleGenExecutor):
#   """Generic TFX CSV example gen executor."""

#   def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
#     """Returns PTransform for CSV to TF examples."""
#     return _CsvToExample