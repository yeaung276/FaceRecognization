import random
import logging
from typing import List


import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.dsl.io.filesystem import PathType

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems


class _ImagesToTriplets(beam.PTransform):
    """Read Images and transform to Triplets."""

    def __init__(self, file_locations: List[PathType], sample_per_class=5):
        """Init method for _ImagesToTriplet."""
        self.file_locations = file_locations
        self.sample_per_class = sample_per_class

    def _get_triplets(self):
        for folder in self.file_locations:
            images = fileio.glob(f"{folder}/*")
            if len(images) < self.sample_per_class:
                logging.warning(f"folder {folder} only has {len(images)} images.")
            # select anchor images
            anchors = random.choices(
                images, k=min(self.sample_per_class, len(images) // 2)
            )
            for anchor in anchors:
                # select positive match
                positive = random.choice(images)
                while positive in anchors:
                    positive = random.choice(images)
                # select negative match
                negative_folder = random.choice(self.file_locations)
                while negative_folder == folder:
                    negative_folder = random.choice(self.file_locations)
                negative = random.choice(fileio.glob(f"{negative_folder}/*"))
                # appand triplet to the list of triplet
                yield (positive, anchor, negative)

    def expand(self, pipeline: beam.Pipeline):
        logging.info("Processing input image data %s to Triplets.")
        triplets = self._get_triplets()
        return pipeline | beam.Create(triplets)


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
            "positive": self._bytes_feature(triplet[0]),
            "anchor": self._bytes_feature(triplet[1]),
            "negative": self._bytes_feature(triplet[2]),
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
        examples = pipeline | beam.Map(self._read_triplets) | beam.Map(self._to_example)
        return examples


class TripletTransform(beam.PTransform):
    """Combine ImageToTriplet and TripletToExample"""

    def __init__(self, file_locations: List[PathType], sample_per_class: int):
        """
        Args:
          file_locations: input dirs that contains image data.
          sample_per_class: number of sample from one folder. If number of images
            in the folder is not enought, only half of the files will be selected
        """
        self.file_locations = file_locations
        self.sample_per_class = sample_per_class

    def expand(self, pipeline: beam.Pipeline):
        return (
            pipeline
            | "ImagesToTriplet"
            >> _ImagesToTriplets(self.file_locations, self.sample_per_class)
            | "TripletToTFExample" >> _TripletToExample()
        )
