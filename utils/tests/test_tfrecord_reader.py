from utils.tf_record_reader import read_tf_record

class TestTRRecordReader:
    def setup(self):
        self.mock_tfr = 'data_tfrecord.gz'

    def test_reader(self):
        read_tf_record(self.mock_tfr)