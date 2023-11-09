import os
import pickle
from typing import List, Tuple
import tensorflow as tf
from tensorflow_transform.tf_metadata import schema_utils
from tfx.dsl.io import fileio

from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx_bsl.public import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2

# Since we're not generating or creating a schema, we will instead create
# a feature spec.  Since there are a fairly small number of features this is
# manageable for this dataset.
_FEATURE_SPEC = {
    'anchor': tf.io.FixedLenFeature([1024], tf.float32),
    'positive': tf.io.FixedLenFeature([1024], tf.float32),
    'negative': tf.io.FixedLenFeature([1024], tf.float32),
}


def _input_fn(file_pattern: List[str],
              data_accessor: DataAccessor,
              schema: schema_pb2.Schema, # type: ignore
              batch_size: int = 200) -> tf.data.Dataset:
    """Generates features and label for training.

    Args:
        file_pattern: List of paths or patterns of input tfrecord files.
        data_accessor: DataAccessor for converting input to RecordBatch.
        schema: schema of the input data.
        batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch

    Returns:
        A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size, num_epochs=1),
        schema=schema # type: ignore
    )
    
class DistanceLayer(tf.keras.layers.Layer):
    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, positive, anchor, negative):
      ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
      an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
      return ap_distance , an_distance
    
class TripletLoss(tf.keras.losses.Loss):
    def __init__(self, alpha, name="triplet_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.margin = alpha

    def call(self, ap_distance, an_distance):
        loss = tf.reduce_mean(tf.maximum(ap_distance - an_distance + self.margin, 0.0))
        return loss

    def get_config(self):
        config = {
            'margin': self.margin
        }
        base_config = super().get_config()
        return {**base_config, **config}

def calc_accuracy(ap_d, an_d):
  count = ap_d.shape[0]
  return tf.reduce_sum(tf.cast(ap_d < an_d, dtype=tf.int16))/count


def _build_keras_model(
    intermediate_layer=32,
    output_layer=16,
    dropout_strength=0.4
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Creates a DNN Keras model for classifying penguin data.

    Returns:
        A Keras Model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(intermediate_layer,
                activation='relu'),
        tf.keras.layers.Dropout(dropout_strength),
        tf.keras.layers.Dense(output_layer,
                activation="relu")
    ])

    # Input Layers for the encoding
    anchor_input   = tf.keras.layers.Input(1024, name="anchor")
    positive_input = tf.keras.layers.Input(1024, name="positive")
    negative_input = tf.keras.layers.Input(1024, name="negative")

    ## Generate the encodings (feature vectors) for the images
    encoded_a = model(anchor_input)
    encoded_p = model(positive_input)
    encoded_n = model(negative_input)

    ## Calculate distance between anchor and positive/negative
    distances = DistanceLayer()(encoded_p, encoded_a, encoded_n)

    # Creating the Model
    siamese_model = tf.keras.models.Model(
            inputs  = {'positive': positive_input, 'anchor': anchor_input, 'negative': negative_input},
            outputs = distances,
            name = "Siamese_Network"
        )
    return siamese_model, model


# TFX Trainer will call this function.
def run_fn(fn_args: FnArgs):
    """Train the model based on given args.

    Args:
        fn_args: Holds args used to train the model as name/value pairs.
    """
    # This schema is usually either an output of SchemaGen or a manually-curated
    # version provided by pipeline author. A schema can also derived from TFT
    # graph if a Transform component is used. In the case when either is missing,
    # `schema_from_feature_spec` could be used to generate schema from very simple
    # feature_spec, but the schema returned would be very primitive.
    
    # hyperparameters
    INTERMEDIATE_LAYER=fn_args.hyperparameters.get('intermediate_layer', 32)
    OUTPUT_LAYER=fn_args.hyperparameters.get('output_layer', 16)
    DROPOUT_STRENGTH=fn_args.hyperparameters.get('dropout_strength', 0.4)
    NO_EPOCH = fn_args.hyperparameters.get('epoch', 50)
    BATCH_SIZE = fn_args.hyperparameters.get('batch_size', 125)
    lr = fn_args.hyperparameters.get('lr', 1e-3)
    alpha=fn_args.hyperparameters.get('alpha', 1.0)

    schema = schema_utils.schema_from_feature_spec(_FEATURE_SPEC)
    
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        schema,
        batch_size=BATCH_SIZE)
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        schema,
        batch_size=BATCH_SIZE)

    siamese_model, model = _build_keras_model(
        intermediate_layer=INTERMEDIATE_LAYER,
        output_layer=OUTPUT_LAYER,
        dropout_strength=DROPOUT_STRENGTH
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fn_args.model_run_dir, histogram_freq=1)
    tensorboard_callback.set_model(siamese_model)
    callbacks =  tf.keras.callbacks.CallbackList([
        tensorboard_callback
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-01)
    loss_eval = TripletLoss(alpha=alpha)

    train_loss_metric = tf.keras.metrics.Mean(name="loss")
    train_acc_metric = tf.keras.metrics.Mean(name="acc")
    train_an_distance = tf.keras.metrics.Mean(name="an_d")
    train_ap_distance = tf.keras.metrics.Mean(name="ap_d")

    test_loss_metric = tf.keras.metrics.Mean(name="loss")
    test_acc_metric = tf.keras.metrics.Mean(name="acc")
    test_an_distance = tf.keras.metrics.Mean(name="an_d")
    test_ap_distance = tf.keras.metrics.Mean(name="ap_d")

    metrics = []

    # start training loop
    callbacks.on_train_begin()
    for ep in range(1,NO_EPOCH+1):
        callbacks.on_epoch_begin(ep)
        # training batch
        callbacks.on_train_batch_begin(ep)
        for data in train_dataset:
            with tf.GradientTape() as tape:
                ap_d, an_d = siamese_model(data) # type: ignore
                train_loss = loss_eval(ap_d, an_d)
                grads = tape.gradient(train_loss, siamese_model.trainable_weights)
                optimizer.apply_gradients(
                    zip(grads, siamese_model.trainable_weights)
                )
                train_loss_metric.update_state(train_loss)
                train_ap_distance.update_state(ap_d)
                train_an_distance.update_state(an_d)
                train_acc_metric.update_state(
                    calc_accuracy(ap_d,an_d)
                )
        callbacks.on_train_batch_end(ep)
        
        # testing batch
        callbacks.on_test_batch_begin(ep)
        for data in eval_dataset:
            ap_d, an_d = siamese_model(data) # type: ignore
            test_loss = loss_eval(ap_d, an_d)
            test_loss_metric.update_state(test_loss)
            test_ap_distance.update_state(ap_d)
            test_an_distance.update_state(an_d)
            test_acc_metric.update_state(
                calc_accuracy(ap_d, an_d)
            )
        callbacks.on_test_batch_end(ep)

        metrics.append({
            "train_loss": train_loss_metric.result(),
            "test_loss": test_loss_metric.result(),
            "train_acc": train_acc_metric.result(),
            "test_acc": test_acc_metric.result(),
            "train_ap": train_ap_distance.result(),
            "test_ap": test_ap_distance.result(),
            "train_an": train_an_distance.result(),
            "test_an": test_an_distance.result()
        })
        callbacks.on_epoch_end(ep)
        
        # clearing the metric values
        train_loss_metric.reset_states()
        train_acc_metric.reset_states()
        train_ap_distance.reset_states()
        train_an_distance.reset_states()
        test_loss_metric.reset_states()
        test_acc_metric.reset_states()
        test_ap_distance.reset_states()
        test_an_distance.reset_states()
    callbacks.on_train_end()

    # The result of the training should be saved in `fn_args.serving_model_dir`
    # directory.
    model.save(fn_args.serving_model_dir, save_format='tf')
    with fileio.open(os.path.join(fn_args.model_run_dir, 'metrics.pkl'), 'wb') as file:
        pickle.dump(metrics, file)