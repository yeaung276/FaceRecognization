import random
import tensorflow as tf
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to

from pipeline.example_gen.executor import _ImagesToTriplets, _TripletToExample

random.seed(0)


class TestExampleGen:
    def setup(self):
        self.base_dir = 'mocks/example_gen'

    def test_triplet_gen(self):
        with TestPipeline() as p:
            examples = (
            p | 'ToTFExample' >> _ImagesToTriplets(self.base_dir,split_pattern='[0-3]',sample_per_class=1)
            )
            assert_that(examples, equal_to([
                ("mocks/example_gen/1/1.png", "mocks/example_gen/1/2.png", "mocks/example_gen/2/2.png"),
                ("mocks/example_gen/2/2.png", "mocks/example_gen/2/1.png", "mocks/example_gen/1/1.png")
            ]))
            
    def test_tfrecord_gen(self):
        with TestPipeline() as p:
            examples = (
                p
                | 'Triplets' >> beam.Create([("mocks/example_gen/1/1.png", "mocks/example_gen/1/2.png", "mocks/example_gen/2/2.png")])
                | 'TFRec transofrm' >> _TripletToExample()
            )

            def check_fn(value):
                assert len(value) == 1, f"input count and output count not match. output: {len(value)}"
                element = value[0]
                assert len(element.features.feature.items()) == 3, f"not a triplet. Got tuple with length of {len(element.features.feature.items())}"
                for _, itm in element.features.feature.items():
                    bin_str = itm.bytes_list.value[0]
                    tensor = tf.io.parse_tensor(bin_str, out_type=tf.uint8)
                    assert isinstance(tensor, tf.Tensor), f"triplet contain non tensor item. {type(itm)}"
                    assert len(tensor.shape) == 3, f"triplet contain non 3D tensor. {len(itm.shape)}"
                
            
            assert_that(examples, check_fn)
    