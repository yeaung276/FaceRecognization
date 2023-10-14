
import random
import logging
import json
from typing import Dict, Any

import tensorflow as tf 
from tfx.types import standard_component_specs
from tfx.dsl.io import fileio

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems

class triplet_component_spec:
    SAMPLE_PER_CLASS = 'sample_per_class'

class _ImagesToTriplets(beam.PTransform):
  """Read Images and transform to Triplets."""

  def __init__(self, base_uri: str, sample_per_class = 5):
    """Init method for _ImagesToTriplet."""
    self.input_base_uri = base_uri
    self.sample_per_class = sample_per_class
    
  def _get_triplets(self):
      image_folders = fileio.listdir(self.input_base_uri)
      for folder in image_folders:
          images = fileio.glob(f'{self.input_base_uri}/{folder}/*')
          if len(images) < self.sample_per_class:
              logging.warning(f'folder {folder} only has {len(images)} images.')
          # select anchor images
          anchors = random.choices(images, k = min(self.sample_per_class, len(images)//2))
          for anchor in anchors:
              # select positive match
              positive = random.choice(images)
              while positive in anchors:
                  positive = random.choice(images)
              # select negative match
              negative_folder = random.choice(image_folders)
              while negative_folder == folder:
                  negative_folder = random.choice(image_folders)
              negative = random.choice(fileio.glob(f'{self.input_base_uri}/{negative_folder}/*'))
              # appand triplet to the list of triplet
              yield (positive, anchor, negative)
      
  def expand(self, pipeline: beam.Pipeline):
    logging.info('Processing input image data %s to Triplets.')
    triplets = self._get_triplets()
    return (pipeline | beam.Create(triplets))


class _TripletToExample(beam.PTransform):
    """Read triplets and convert them into TFRecords"""
    
    def __init__(self):
        """Init method for _TripletToExample.

        Args:
          exec_properties: A dict of execution properties.
            - input_base: input dir that contains image data.
        """
        pass
      
    def _process_file(self, path):
        with FileSystems.open(path) as file:
          tensor = tf.io.decode_image(file.read(), channels=3)
          return tensor
      
    def _read_triplets(self, triplet: tuple):
        return tuple(map(self._process_file, triplet))
      
    def _to_example(self, triplet: tuple):
        feature = {
            'positive': self._bytes_feature(triplet[0]),
            'anchor': self._bytes_feature(triplet[1]),
            'negative': self._bytes_feature(triplet[2])
        }
        # Create a Features message using tf.train.Example.
        return tf.train.Example(features=tf.train.Features(feature=feature))
  
    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = tf.io.serialize_tensor(value).numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
              
    def expand(self, pipeline: beam.Pipeline):
        logging.info("Processing triplets into TFExamples")
        examples = (
            pipeline
            | beam.Map(self._read_triplets)
            | beam.Map(self._to_example)
        )
        return examples
      
class _TripletTransform(beam.PTransform):
    """Combine ImageToTriplet and TripletToExample"""
   
    def __init__(self, exec_properties: Dict[str, Any], split_pattern: str):
        """    
        Args:
          exec_properties: A dict of execution properties.
            - input_base: input dir that contains image data.
            - custom_config: a dict containing sample_per_class
          sample_per_class: number of sample from one folder. If number of images
            in the folder is not enought, only half of the files will be selected
        """
        if split_pattern != '*':
            logging.warn(f'split_pattern provided but it will not have any affect on splitting the data')
        self.input_base_uri = exec_properties[standard_component_specs.INPUT_BASE_KEY]
        if exec_properties[standard_component_specs.CUSTOM_CONFIG_KEY] is None:
            logging.info(f'sample_per_class is not specified. Default to 5 sample_per_class.')
            self.sample_per_class = 5
        else:
            config = json.loads(exec_properties[standard_component_specs.CUSTOM_CONFIG_KEY])
            self.sample_per_class = config[triplet_component_spec.SAMPLE_PER_CLASS]
      
    def expand(self, pipeline: beam.Pipeline):
      return (
        pipeline 
        | "ImagesToTriplet" >> _ImagesToTriplets(self.input_base_uri, self.sample_per_class) 
        | "TripletToTFExample" >> _TripletToExample()
      )