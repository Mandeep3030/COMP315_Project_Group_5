"""Reusable definition of the complete COMP315 TFX pipeline.

This module only builds and returns a pipeline object. It does not choose a
runner, so the same ``create_pipeline`` function can be used by LocalDagRunner
and the Apache Airflow DAG.
"""

from tfx.components import (
    CsvExampleGen,
    ExampleValidator,
    SchemaGen,
    StatisticsGen,
    Trainer,
    Transform,
)
from tfx.orchestration import metadata
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.proto import example_gen_pb2, trainer_pb2

try:
    # Used when imported as pipeline.pipeline_definition.
    from .step7_resolver import create_resolver
    from .step8_evaluator import create_evaluator, create_eval_config
    from .step9_pusher import create_pusher
except ImportError:
    # Supports loading this module directly from the pipeline directory.
    from step7_resolver import create_resolver
    from step8_evaluator import create_evaluator, create_eval_config
    from step9_pusher import create_pusher


def create_pipeline(
    pipeline_name,
    pipeline_root,
    data_root,
    preprocessing_module,
    trainer_module,
    serving_model_dir,
    metadata_path,
    train_steps=500,
    eval_steps=100,
    eval_config=None,
    train_hash_buckets=2,
    eval_hash_buckets=1,
    beam_pipeline_args=None,
    enable_cache=False,
):
    """Create the complete ExampleGen-to-Pusher TFX pipeline.

    Args:
        pipeline_name: Stable pipeline name shared by repeated runs.
        pipeline_root: Persistent directory for all TFX artifacts.
        data_root: Directory containing only the TFX-compatible input CSV.
        preprocessing_module: Path to the Transform preprocessing module.
        trainer_module: Path to the Trainer module containing ``run_fn``.
        serving_model_dir: Destination where blessed models are deployed.
        metadata_path: Path to the persistent ML Metadata SQLite database.
        train_steps: Number of Trainer training steps.
        eval_steps: Number of Trainer validation steps.
        eval_config: Optional TFMA EvalConfig; uses the Step 8 config by default.
        train_hash_buckets: ExampleGen hash buckets assigned to training.
        eval_hash_buckets: ExampleGen hash buckets assigned to evaluation.
        beam_pipeline_args: Optional Apache Beam runner arguments.
        enable_cache: Whether TFX may reuse cached component executions.

    Returns:
        A TFX 1.12 ``tfx.orchestration.pipeline.Pipeline`` object. This is the
        version's equivalent of the assignment's ``tfx.dsl.Pipeline``.
    """
    if train_hash_buckets < 1 or eval_hash_buckets < 1:
        raise ValueError("ExampleGen hash bucket counts must be positive")
    if train_steps < 1 or eval_steps < 1:
        raise ValueError("Trainer step counts must be positive")

    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(
                    name="train", hash_buckets=train_hash_buckets
                ),
                example_gen_pb2.SplitConfig.Split(
                    name="eval", hash_buckets=eval_hash_buckets
                ),
            ]
        )
    )

    example_gen = CsvExampleGen(
        input_base=data_root,
        output_config=output_config,
    )
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"],
    )
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=False,
    )
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=preprocessing_module,
    )
    # Transform does not consume the anomaly artifact, so use a control edge
    # to ensure validation completes before feature engineering begins.
    transform.add_upstream_node(example_validator)
    trainer = Trainer(
        module_file=trainer_module,
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(num_steps=train_steps),
        eval_args=trainer_pb2.EvalArgs(num_steps=eval_steps),
    )

    # Resolver reads the same persistent SQLite database used by this pipeline.
    # This lets later runs locate the most recently blessed model.
    model_resolver = create_resolver()
    model_resolver.add_upstream_node(trainer)
    if eval_config is None:
        eval_config = create_eval_config()
    evaluator = create_evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )
    pusher = create_pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        serving_model_dir=serving_model_dir,
    )

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    ]

    return tfx_pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_pipeline_args or [],
        enable_cache=enable_cache,
    )
