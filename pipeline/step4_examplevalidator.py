"""Step 4: validate train and eval examples against the inferred schema.

Run from the project root with the course Conda environment:

    conda run -n tfx-env python pipeline/step4_examplevalidator.py
"""

import glob
import os

from tensorflow_metadata.proto.v0 import anomalies_pb2
from tfx.components import CsvExampleGen, ExampleValidator, SchemaGen, StatisticsGen
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.proto import example_gen_pb2


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PIPELINE_ROOT = os.path.join(PROJECT_DIR, "pipeline_output", "step4_validator")
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")


def load_anomalies(anomalies_path):
    """Load the serialized SchemaDiff.pb artifact produced by TFX 1.12."""
    anomalies = anomalies_pb2.Anomalies()
    with open(anomalies_path, "rb") as anomalies_file:
        anomalies.ParseFromString(anomalies_file.read())
    return anomalies


def format_split_result(split_name, anomalies_path, anomalies):
    """Return readable report lines for one split's anomaly artifact."""
    lines = [
        "===== {} VALIDATION =====".format(split_name.upper()),
        "Artifact: {}".format(anomalies_path),
    ]

    if not anomalies.anomaly_info:
        lines.append("Result: No anomalies detected.")
        return lines

    lines.append("Result: {} anomaly/anomalies detected.".format(
        len(anomalies.anomaly_info)
    ))
    for feature_name, anomaly in sorted(anomalies.anomaly_info.items()):
        lines.append("Feature: {}".format(feature_name))
        lines.append("  {}".format(anomaly.short_description))
        if anomaly.description:
            lines.append("  Details: {}".format(anomaly.description))
    return lines


def main():
    """Run ExampleGen, StatisticsGen, SchemaGen, and ExampleValidator."""
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

    # ExampleValidator compares the observed statistics with the inferred schema.
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    validation_pipeline = pipeline.Pipeline(
        pipeline_name="step4_examplevalidator",
        pipeline_root=PIPELINE_ROOT,
        components=[example_gen, statistics_gen, schema_gen, example_validator],
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            METADATA_PATH
        ),
    )
    LocalDagRunner().run(validation_pipeline)

    artifact_dirs = glob.glob(
        os.path.join(PIPELINE_ROOT, "ExampleValidator", "anomalies", "*")
    )
    if not artifact_dirs:
        raise RuntimeError("ExampleValidator did not create an anomaly artifact")

    artifact_uri = max(artifact_dirs, key=os.path.getmtime)
    report_lines = ["ExampleValidator artifact URI: {}".format(artifact_uri)]
    pipeline_healthy = True

    for split_name in ("train", "eval"):
        split_uri = os.path.join(artifact_uri, "Split-" + split_name)
        anomalies_path = os.path.join(split_uri, "SchemaDiff.pb")
        if not os.path.isfile(anomalies_path):
            raise RuntimeError("Missing anomaly artifact: " + anomalies_path)

        anomalies = load_anomalies(anomalies_path)
        if anomalies.anomaly_info:
            pipeline_healthy = False
        report_lines.append("")
        report_lines.extend(
            format_split_result(split_name, anomalies_path, anomalies)
        )

    report_lines.extend([
        "",
        "DATA PIPELINE HEALTHY: {}".format(pipeline_healthy),
    ])
    report = "\n".join(report_lines)
    print("\n" + report)


if __name__ == "__main__":
    main()
