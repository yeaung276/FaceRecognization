from utils.tf_record_reader import read_examplegen_output

class TestTRRecordReader:
    def setup(self):
        self.mock_tfr = 'mocks/pipeline_root/TripletExampleGen/examples/17/Split-train/data_tfrecord-00000-of-00001.gz'

    def test_reader(self):
        result = list(read_examplegen_output(self.mock_tfr))
        assert len(result) == 2, f'excepting result length of 2. instead get {len(result)}'