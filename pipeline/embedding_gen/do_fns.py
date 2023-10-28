from typing import Tuple, Any, Iterable
import apache_beam as beam
import uuid

class FlatTriplets(beam.DoFn):
    def process(self, element: dict):
        key = uuid.uuid4()
        yield key, element['positive']
        yield key, element['anchor']
        yield key, element['negative']
        
class ToTriplets(beam.DoFn):
    def process(self, element: Tuple[Any, Iterable[Any]]):
        _, items = element
        yield {
            'positive': list(items)[0],
            'anchor': list(items)[1],
            'negative': list(items)[2]
        }