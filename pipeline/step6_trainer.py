"""Step 6: train and export a Keras model with the TFX Trainer.

Run from the project root with the course Conda environment:

    conda run -n tfx-env python pipeline/step6_trainer.py

This file serves two purposes: ``run_fn`` is loaded by the Trainer executor,
while ``main`` creates a small local pipeline that tests training through Step 6.
"""

import glob
import os

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components import (
    CsvExampleGen,
    SchemaGen,
    StatisticsGen,
    Trainer,
    Transform,
)
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.proto import example_gen_pb2, trainer_pb2


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PIPELINE_ROOT = os.path.join(PROJECT_DIR, "pipeline_output", "step6_trainer")
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")
PREPROCESSING_MODULE = os.path.join(PROJECT_DIR, "pipeline", "preprocessing.py")
TRAINER_MODULE = os.path.abspath(__file__)

NUMERIC_FEATURES = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]
CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]
LABEL_KEY = "y"
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
TRAIN_STEPS = 500
EVAL_STEPS = 100


def transformed_name(key):
    """Return the feature name produced by preprocessing.py."""
    return key + "_xf"


def _input_fn(file_pattern, tf_transform_output, batch_size):
    """Read compressed transformed examples for Keras training/evaluation."""
    transformed_feature_spec = tf_transform_output.transformed_feature_spec()
    filenames = tf.io.gfile.glob(file_pattern)
    if not filenames:
        raise ValueError("No transformed examples matched: " + file_pattern)

    def parse_examples(serialized_examples):
        features = tf.io.parse_example(
            serialized_examples, transformed_feature_spec
        )
        label = features.pop(transformed_name(LABEL_KEY))
        # The transformed schema stores scalar values, while Keras dense
        # inputs use one value per row with shape (batch, 1).
        features = {
            name: tf.reshape(value, [-1, 1])
            for name, value in features.items()
        }
        return features, tf.reshape(label, [-1, 1])

    dataset = tf.data.TFRecordDataset(
        filenames, compression_type="GZIP"
    )
    return (
        dataset.shuffle(1000)
        .repeat()
        .batch(batch_size)
        .map(parse_examples, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_keras_model():
    """Build a binary classifier using all transformed input features."""
    inputs = {}
    encoded_features = []

    for feature_name in NUMERIC_FEATURES:
        name = transformed_name(feature_name)
        feature_input = tf.keras.Input(shape=(1,), name=name, dtype=tf.float32)
        inputs[name] = feature_input
        encoded_features.append(feature_input)

    for feature_name in CATEGORICAL_FEATURES:
        name = transformed_name(feature_name)
        feature_input = tf.keras.Input(shape=(1,), name=name, dtype=tf.int64)
        inputs[name] = feature_input
        # Vocabulary IDs are already numeric after Transform. Casting lets the
        # compact classroom model consume them alongside z-scored features.
        encoded_features.append(tf.cast(feature_input, tf.float32))

    features = tf.keras.layers.concatenate(encoded_features)
    features = tf.keras.layers.Dense(64, activation="relu")(features)
    features = tf.keras.layers.Dropout(0.2)(features)
    features = tf.keras.layers.Dense(32, activation="relu")(features)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="probabilities")(
        features
    )

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def _serving_signature(model, tf_transform_output):
    """Create a signature that accepts raw serialized tf.Example records."""
    model.tft_layer = tf_transform_output.transform_features_layer()
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    raw_feature_spec.pop(LABEL_KEY)

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[None], dtype=tf.string, name="examples"
            )
        ]
    )
    def serve_tf_examples(serialized_tf_examples):
        raw_features = tf.io.parse_example(
            serialized_tf_examples, raw_feature_spec
        )
        transformed_features = model.tft_layer(raw_features)
        model_inputs = {
            name: tf.reshape(transformed_features[name], [-1, 1])
            for name in model.input_names
        }
        return {"probabilities": model(model_inputs)}

    return serve_tf_examples


def run_fn(fn_args):
    """Train the model and export a SavedModel for serving."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    train_dataset = _input_fn(
        fn_args.train_files,
        tf_transform_output,
        TRAIN_BATCH_SIZE,
    )
    eval_dataset = _input_fn(
        fn_args.eval_files,
        tf_transform_output,
        EVAL_BATCH_SIZE,
    )

    model = build_keras_model()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir,
        update_freq="batch",
    )
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=1,
        callbacks=[tensorboard_callback],
    )

    serving_signature = _serving_signature(model, tf_transform_output)
    model.save(
        fn_args.serving_model_dir,
        save_format="tf",
        signatures={"serving_default": serving_signature},
    )


def latest_model_uri():
    """Return the newest SavedModel artifact created by Trainer."""
    artifact_dirs = glob.glob(
        os.path.join(PIPELINE_ROOT, "Trainer", "model", "*")
    )
    if not artifact_dirs:
        raise RuntimeError("Trainer did not create a model artifact")
    return max(artifact_dirs, key=os.path.getmtime)


def main():
    """Run the pipeline through Trainer and display the exported model URI."""
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(
                    name="train", hash_buckets=2
                ),
                example_gen_pb2.SplitConfig.Split(
                    name="eval", hash_buckets=1
                ),
            ]
        )
    )

    example_gen = CsvExampleGen(
        input_base=DATA_DIR,
        output_config=output_config,
    )
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"],
    )
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=False,
    )
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=PREPROCESSING_MODULE,
    )

    # Trainer consumes both Transform outputs, the inferred schema, and this
    # module's run_fn. Step counts are explicitly part of its configuration.
    trainer = Trainer(
        module_file=TRAINER_MODULE,
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(num_steps=TRAIN_STEPS),
        eval_args=trainer_pb2.EvalArgs(num_steps=EVAL_STEPS),
    )

    trainer_pipeline = pipeline.Pipeline(
        pipeline_name="step6_trainer",
        pipeline_root=PIPELINE_ROOT,
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            transform,
            trainer,
        ],
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            METADATA_PATH
        ),
    )
    LocalDagRunner().run(trainer_pipeline)

    print("\n===== TRAINER RESULTS =====")
    print("Trainer module:", TRAINER_MODULE)
    print("Train steps:", TRAIN_STEPS)
    print("Eval steps:", EVAL_STEPS)
    print("Model artifact URI:", latest_model_uri())
    print("TRAINER COMPLETED SUCCESSFULLY: True")


if __name__ == "__main__":
    main()
