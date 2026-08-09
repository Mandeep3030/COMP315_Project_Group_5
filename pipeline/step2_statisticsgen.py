"""Run ExampleGen and StatisticsGen, then display both split statistics.

Run this file from the project's ``tfx-env`` Conda environment:

    python pipeline/step2_statisticsgen.py
"""

import glob
import os

import tensorflow_data_validation as tfdv
from tfx.components import CsvExampleGen, StatisticsGen
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.proto import example_gen_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


# Resolve all paths from this file so the script works after moving the project.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PIPELINE_ROOT = os.path.join(PROJECT_DIR, "pipeline_output", "step2_statisticsgen_test")
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")
REPORT_PATH = os.path.join(PIPELINE_ROOT, "statistics_report.html")


def _load_statistics(statistics_path):
    """Load a TFX 1.12 FeatureStats.pb artifact."""
    # TFX 1.12 writes FeatureStats.pb as a serialized protobuf (not a
    # TFRecord), so parse the artifact in its native format.
    statistics = statistics_pb2.DatasetFeatureStatisticsList()
    with open(statistics_path, "rb") as statistics_file:
        statistics.ParseFromString(statistics_file.read())
    return statistics


def _write_statistics_report(train_statistics, eval_statistics):
    """Create an interactive Facets report comparing train and eval data."""
    report_html = tfdv.get_statistics_html(
        lhs_statistics=train_statistics,
        rhs_statistics=eval_statistics,
        lhs_name="Train",
        rhs_name="Eval",
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
        report_file.write(report_html)


def main():
    """Execute the two-component test pipeline and display its artifacts."""
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(
                    name="train",
                    hash_buckets=2,
                ),
                example_gen_pb2.SplitConfig.Split(
                    name="eval",
                    hash_buckets=1,
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

    statistics_pipeline = pipeline.Pipeline(
        pipeline_name="statisticsgen_test",
        pipeline_root=PIPELINE_ROOT,
        components=[example_gen, statistics_gen],
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            METADATA_PATH
        ),
    )
    LocalDagRunner().run(statistics_pipeline)

    artifact_dirs = glob.glob(
        os.path.join(PIPELINE_ROOT, "StatisticsGen", "statistics", "*")
    )
    if not artifact_dirs:
        raise RuntimeError("StatisticsGen completed without creating an artifact")

    artifact_uri = max(artifact_dirs, key=os.path.getmtime)
    print("\n===== STATISTICSGEN RESULTS =====")
    print("Artifact URI:", artifact_uri)

    split_statistics = {}
    for split_name in ("train", "eval"):
        split_uri = os.path.join(artifact_uri, "Split-" + split_name)
        statistics_path = os.path.join(split_uri, "FeatureStats.pb")
        if not os.path.isfile(statistics_path):
            raise RuntimeError(
                "Missing statistics for {} split: {}".format(
                    split_name, statistics_path
                )
            )
        print("{} URI:".format(split_name.capitalize()), split_uri)
        split_statistics[split_name] = _load_statistics(statistics_path)

    _write_statistics_report(
        train_statistics=split_statistics["train"],
        eval_statistics=split_statistics["eval"],
    )
    print("\nInteractive train/eval statistics report:", REPORT_PATH)
    print("Open this HTML file in a browser to explore feature distributions.")


if __name__ == "__main__":
    main()
