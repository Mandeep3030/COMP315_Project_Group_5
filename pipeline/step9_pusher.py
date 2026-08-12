"""Step 9: deploy a candidate model only when Evaluator blesses it.

Run from the project root with the course Conda environment:

    conda run -n tfx-env python pipeline/step9_pusher.py

The Pusher receives Evaluator's ModelBlessing artifact as a gate. If the
candidate fails evaluation, no model is copied into ``serving_model``.
"""

import os

from tfx.components import (
    CsvExampleGen,
    Pusher,
    SchemaGen,
    StatisticsGen,
    Trainer,
    Transform,
)
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2

from step7_resolver import create_resolver
from step8_evaluator import create_evaluator


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PIPELINE_ROOT = os.path.join(PROJECT_DIR, "pipeline_output", "step9_pusher")
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")
PREPROCESSING_MODULE = os.path.join(PROJECT_DIR, "pipeline", "preprocessing.py")
TRAINER_MODULE = os.path.join(PROJECT_DIR, "pipeline", "step6_trainer.py")
SERVING_MODEL_DIR = os.path.join(PROJECT_DIR, "serving_model")

TRAIN_STEPS = 500
EVAL_STEPS = 100


def create_pusher(model, model_blessing, serving_model_dir=SERVING_MODEL_DIR):
    """Create a Pusher gated by Evaluator's model blessing artifact."""
    return Pusher(
        model=model,
        model_blessing=model_blessing,
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )


def main():
    """Run the local development pipeline through conditional deployment."""
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

    model_resolver = create_resolver()
    evaluator = create_evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
    )

    # Pusher examines this blessing before copying the candidate SavedModel.
    # An unblessed model still has an artifact, but Pusher skips deployment.
    pusher = create_pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
    )

    pusher_pipeline = pipeline.Pipeline(
        pipeline_name="step9_pusher",
        pipeline_root=PIPELINE_ROOT,
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            transform,
            trainer,
            model_resolver,
            evaluator,
            pusher,
        ],
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            METADATA_PATH
        ),
    )
    LocalDagRunner().run(pusher_pipeline)

    print("\n===== PUSHER RESULTS =====")
    print("Serving model directory:", SERVING_MODEL_DIR)
    print("Candidate model:", trainer.outputs["model"])
    print("Model blessing:", evaluator.outputs["blessing"])
    print("Pushed model:", pusher.outputs["pushed_model"])
    print("Only an Evaluator-blessed model is copied to the serving directory.")
    print("PUSHER COMPLETED SUCCESSFULLY: True")


if __name__ == "__main__":
    main()
