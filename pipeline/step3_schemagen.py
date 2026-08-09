"""Step 3: infer, display, and summarize the dataset schema.

Run this script from the project root with the course Conda environment:

    conda run -n tfx-env python pipeline/step3_schemagen.py

The pipeline runs ExampleGen and StatisticsGen because SchemaGen consumes the
statistics artifact produced by the preceding stage.
"""

import glob
import os

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx.components import CsvExampleGen, SchemaGen, StatisticsGen
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.proto import example_gen_pb2


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PIPELINE_ROOT = os.path.join(PROJECT_DIR, "pipeline_output", "step3_schemagen")
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")
SUMMARY_PATH = os.path.join(PIPELINE_ROOT, "schema_summary.txt")


def load_schema(schema_path):
    """Load SchemaGen's text-format protobuf artifact."""
    schema = schema_pb2.Schema()
    with open(schema_path, "r", encoding="utf-8") as schema_file:
        text_format.Parse(schema_file.read(), schema)
    return schema


def feature_type_name(feature):
    """Return the readable protobuf type name for a schema feature."""
    return schema_pb2.FeatureType.Name(feature.type)


def create_schema_summary(schema):
    """Create the short schema review required by the assignment."""
    numeric_features = []
    categorical_features = []

    for feature in schema.feature:
        if feature.type in (schema_pb2.INT, schema_pb2.FLOAT):
            numeric_features.append(feature.name)
        else:
            categorical_features.append(feature.name)

    lines = [
        "===== SCHEMA REVIEW SUMMARY =====",
        "Total features: {}".format(len(schema.feature)),
        "Numeric features ({}): {}".format(
            len(numeric_features), ", ".join(sorted(numeric_features))
        ),
        "Categorical/string features ({}): {}".format(
            len(categorical_features), ", ".join(sorted(categorical_features))
        ),
        "",
        "Feature details:",
    ]

    domains = {domain.name: list(domain.value) for domain in schema.string_domain}
    for feature in sorted(schema.feature, key=lambda item: item.name):
        details = [feature_type_name(feature)]
        if feature.HasField("presence"):
            details.append(
                "required fraction={:.2f}".format(feature.presence.min_fraction)
            )
        if feature.HasField("value_count"):
            details.append(
                "values/example={}-{}".format(
                    feature.value_count.min, feature.value_count.max
                )
            )
        if feature.domain and feature.domain in domains:
            details.append("vocabulary={}".format(domains[feature.domain]))
        lines.append("- {}: {}".format(feature.name, "; ".join(details)))

    lines.extend([
        "",
        "Interpretation:",
        "The inferred schema contains one scalar value per feature and marks all "
        "features as present in the observed training data. Numeric columns were "
        "inferred as integers, while categorical columns and the target 'y' were "
        "inferred as byte strings with explicit vocabularies. This inferred schema "
        "is a useful starting point, but production validation should curate it "
        "with intentional numeric ranges and domain rules instead of relying only "
        "on constraints learned from the same dataset.",
    ])
    return "\n".join(lines)


def main():
    """Run ExampleGen, StatisticsGen, and SchemaGen and display the schema."""
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=2),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=1),
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

    schema_pipeline = pipeline.Pipeline(
        pipeline_name="step3_schemagen",
        pipeline_root=PIPELINE_ROOT,
        components=[example_gen, statistics_gen, schema_gen],
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            METADATA_PATH
        ),
    )
    LocalDagRunner().run(schema_pipeline)

    artifact_dirs = glob.glob(
        os.path.join(PIPELINE_ROOT, "SchemaGen", "schema", "*")
    )
    if not artifact_dirs:
        raise RuntimeError("SchemaGen did not create a schema artifact")

    artifact_uri = max(artifact_dirs, key=os.path.getmtime)
    schema_path = os.path.join(artifact_uri, "schema.pbtxt")
    if not os.path.isfile(schema_path):
        raise RuntimeError("Missing schema artifact: " + schema_path)

    schema = load_schema(schema_path)
    summary = create_schema_summary(schema)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as summary_file:
        summary_file.write(summary + "\n")

    print("\n===== SCHEMAGEN RESULTS =====")
    print("Schema artifact URI:", artifact_uri)
    print("Schema file:", schema_path)
    print("\n===== GENERATED SCHEMA =====")
    with open(schema_path, "r", encoding="utf-8") as schema_file:
        print(schema_file.read())
    print(summary)
    print("\nSchema summary saved to:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
