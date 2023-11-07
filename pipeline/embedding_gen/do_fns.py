from typing import Tuple, Any, Iterable
import apache_beam as beam
import uuid
import tensorflow as tf


class FlatTriplets(beam.DoFn):
    def mobile_net_preprocessor(self, x):
        x = tf.cast(x, tf.float16)
        x /= 127.5
        x -= 1.0
        return x

    def process(self, element: dict):
        key = uuid.uuid4()
        yield key, self.mobile_net_preprocessor(element["positive"])
        yield key, self.mobile_net_preprocessor(element["anchor"])
        yield key, self.mobile_net_preprocessor(element["negative"])


class ToTriplets(beam.DoFn):
    def process(self, element: Tuple[Any, Iterable[Any]]):
        _, items = element
        yield {
            "positive": list(items)[0],
            "anchor": list(items)[1],
            "negative": list(items)[2],
        }
