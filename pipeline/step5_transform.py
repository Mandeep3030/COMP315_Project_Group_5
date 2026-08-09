"""Step 5: apply feature engineering with TensorFlow Transform.

Run from the project root with the course Conda environment:

    conda run -n tfx-env python pipeline/step5_transform.py
"""

import glob
import os

from tfx.components import CsvExampleGen, SchemaGen, StatisticsGen, Transform
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.proto import example_gen_pb2


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PIPELINE_ROOT = os.path.join(PROJECT_DIR, "pipeline_output", "step5_transform")
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")
PREPROCESSING_MODULE = os.path.join(PROJECT_DIR, "pipeline", "preprocessing.py")


def latest_artifact_uri(component_name, output_name):
    """Return the newest artifact URI for a component output."""
    artifact_dirs = glob.glob(
        os.path.join(PIPELINE_ROOT, component_name, output_name, "*")
    )
    if not artifact_dirs:
        raise RuntimeError(
            "{} did not create its {} artifact".format(
                component_name, output_name
            )
        )
    return max(artifact_dirs, key=os.path.getmtime)


def main():
    """Run ExampleGen, StatisticsGen, SchemaGen, and Transform."""
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

    # preprocessing.py scales numeric features, creates categorical vocabulary
    # IDs, and converts the yes/no target into a binary integer label.
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=PREPROCESSING_MODULE,
    )

    transform_pipeline = pipeline.Pipeline(
        pipeline_name="step5_transform",
        pipeline_root=PIPELINE_ROOT,
        components=[example_gen, statistics_gen, schema_gen, transform],
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            METADATA_PATH
        ),
    )
    LocalDagRunner().run(transform_pipeline)

    transformed_examples_uri = latest_artifact_uri(
        "Transform", "transformed_examples"
    )
    transform_graph_uri = latest_artifact_uri("Transform", "transform_graph")
    transformed_schema_uri = latest_artifact_uri(
        "Transform", "post_transform_schema"
    )

    print("\n===== TRANSFORM RESULTS =====")
    print("Preprocessing module:", PREPROCESSING_MODULE)
    print("Transform graph URI:", transform_graph_uri)
    print("Transformed schema URI:", transformed_schema_uri)
    print("Transformed examples URI:", transformed_examples_uri)
    print(
        "Train URI:",
        os.path.join(transformed_examples_uri, "Split-train"),
    )
    print(
        "Eval URI:",
        os.path.join(transformed_examples_uri, "Split-eval"),
    )
    print("TRANSFORM COMPLETED SUCCESSFULLY: True")


if __name__ == "__main__":
    main()
