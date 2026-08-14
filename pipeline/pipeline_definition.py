# COMP_315 | SEC_401 | GROUP_5
# Mandeep Singh | Dennis Kozevnikoff
# =========

"""Build the complete reusable TFX pipeline for Airflow."""

# ===============
# PIPELINE DEFINITION
# ===============

from tfx.orchestration import metadata
from tfx.orchestration import pipeline as tfx_pipeline

from .step1_examplegen import create_example_gen
from .step2_statisticsgen import create_statistics_gen
from .step3_schemagen import create_schema_gen
from .step4_examplevalidator import create_example_validator
from .step5_transform import create_transform
from .step6_trainer import create_trainer
from .step7_resolver import create_resolver
from .step8_evaluator import create_evaluator
from .step9_pusher import create_pusher


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
    """Create the integrated ExampleGen-to-Pusher pipeline."""
    example_gen = create_example_gen(
        data_root,
        train_hash_buckets=train_hash_buckets,
        eval_hash_buckets=eval_hash_buckets,
    )
    statistics_gen = create_statistics_gen(example_gen.outputs["examples"])
    schema_gen = create_schema_gen(statistics_gen.outputs["statistics"])
    example_validator = create_example_validator(
        statistics_gen.outputs["statistics"],
        schema_gen.outputs["schema"],
    )
    transform = create_transform(
        example_gen.outputs["examples"],
        schema_gen.outputs["schema"],
        preprocessing_module,
    )
    # Validation is a required control step before feature engineering.
    transform.add_upstream_node(example_validator)

    trainer = create_trainer(
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        trainer_module=trainer_module,
        train_steps=train_steps,
        eval_steps=eval_steps,
    )
    model_resolver = create_resolver()
    model_resolver.add_upstream_node(trainer)
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
