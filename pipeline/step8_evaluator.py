"""Step 8: evaluate the candidate model with TFMA and bless improvements.

Run from the project root with the course Conda environment:

    conda run -n tfx-env python pipeline/step8_evaluator.py

This development pipeline uses persistent ML Metadata. On its first run there
is no baseline, so Evaluator can bless the candidate. Later runs use the most
recent blessed model returned by Resolver as the comparison baseline.
"""

import os

import tensorflow_model_analysis as tfma
from google.protobuf import wrappers_pb2
from tfx.components import (
    CsvExampleGen,
    Evaluator,
    SchemaGen,
    StatisticsGen,
    Trainer,
    Transform,
)
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.proto import example_gen_pb2, trainer_pb2

from step7_resolver import create_resolver


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PIPELINE_ROOT = os.path.join(PROJECT_DIR, "pipeline_output", "step8_evaluator")
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")
PREPROCESSING_MODULE = os.path.join(PROJECT_DIR, "pipeline", "preprocessing.py")
TRAINER_MODULE = os.path.join(PROJECT_DIR, "pipeline", "step6_trainer.py")

LABEL_KEY = "y"
TRAIN_STEPS = 500
EVAL_STEPS = 100


def create_eval_config():
    """Return the TFMA metrics, slices, and candidate blessing criteria."""
    accuracy_threshold = tfma.MetricThreshold(
        change_threshold=tfma.GenericChangeThreshold(
            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
            absolute=wrappers_pb2.DoubleValue(value=0.0001),
        )
    )

    return tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                label_key=LABEL_KEY,
                signature_name="serving_default",
                preprocessing_function_names=["transformed_labels"],
                prediction_key="probabilities",
            )
        ],
        slicing_specs=[
            # Empty SlicingSpec evaluates the complete evaluation dataset.
            tfma.SlicingSpec(),
            tfma.SlicingSpec(feature_keys=["marital"]),
            tfma.SlicingSpec(feature_keys=["job"]),
            tfma.SlicingSpec(feature_keys=["education"]),
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        threshold=accuracy_threshold,
                    ),
                    tfma.MetricConfig(class_name="AUC"),
                ]
            )
        ],
    )


def create_evaluator(examples, model, baseline_model):
    """Create Evaluator wired to examples, candidate, and resolved baseline."""
    return Evaluator(
        examples=examples,
        model=model,
        baseline_model=baseline_model,
        eval_config=create_eval_config(),
    )


def main():
    """Run the local development pipeline through Resolver and Evaluator."""
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
    trainer = Trainer(
        module_file=TRAINER_MODULE,
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(num_steps=TRAIN_STEPS),
        eval_args=trainer_pb2.EvalArgs(num_steps=EVAL_STEPS),
    )

    # Resolver searches this pipeline's persistent metadata database for the
    # latest model that a previous Evaluator execution blessed.
    model_resolver = create_resolver()
    evaluator = create_evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
    )

    evaluator_pipeline = pipeline.Pipeline(
        pipeline_name="step8_evaluator",
        pipeline_root=PIPELINE_ROOT,
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            transform,
            trainer,
            model_resolver,
            evaluator,
        ],
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            METADATA_PATH
        ),
    )
    LocalDagRunner().run(evaluator_pipeline)

    print("\n===== EVALUATOR RESULTS =====")
    print("Label key:", LABEL_KEY)
    print("Metrics: BinaryAccuracy, AUC")
    print("Slices: overall, marital, job, education")
    print("Evaluation artifact:", evaluator.outputs["evaluation"])
    print("Model blessing:", evaluator.outputs["blessing"])
    print("EVALUATOR COMPLETED SUCCESSFULLY: True")


if __name__ == "__main__":
    main()
