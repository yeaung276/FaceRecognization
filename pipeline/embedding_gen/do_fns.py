import uuid
from typing import Tuple, Any, Iterable, Dict
import tensorflow as tf
import apache_beam as beam
from apache_beam.ml.inference.base import RunInference, PredictionResult


class ParseAndFlatTriplets(beam.DoFn):
    def _parse_example(self, raw_example):
        example = tf.train.Example.FromString(raw_example)
        triplet = {}
        for key, value in example.features.feature.items():
            triplet[key] = tf.io.parse_tensor(
                value.bytes_list.value[0], out_type=tf.uint8
            )
        return triplet
    def mobile_net_preprocessor(self, x):
        x = tf.cast(x, tf.float16)
        x /= 127.5 # type: ignore
        x -= 1.0
        return x

    def process(self, raw_example):
        triplet = self._parse_example(raw_example)
        key = uuid.uuid4()
        yield key, self.mobile_net_preprocessor(triplet["positive"])
        yield key, self.mobile_net_preprocessor(triplet["anchor"])
        yield key, self.mobile_net_preprocessor(triplet["negative"])


class ToTripletExample(beam.DoFn):
    def _to_example(self, triplet: Dict[str, PredictionResult]):
        features = {
            "anchor": tf.train.Feature(
                float_list=tf.train.FloatList(value=triplet["anchor"].inference.numpy().tolist())  # type: ignore
            ),
            "positive": tf.train.Feature(
                float_list=tf.train.FloatList(value=triplet["positive"].inference.numpy().tolist())  # type: ignore
            ),
            "negative": tf.train.Feature(
                float_list=tf.train.FloatList(value=triplet["negative"].inference.numpy().tolist())  # type: ignore
            ),
        }
        return tf.train.Example(features=tf.train.Features(feature=features))
    
    def process(self, element: Tuple[Any, Iterable[Any]]):
        _, items = element
        yield self._to_example({
            "positive": list(items)[0],
            "anchor": list(items)[1],
            "negative": list(items)[2],
        })
