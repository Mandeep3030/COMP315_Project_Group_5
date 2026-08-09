"""Step 3: generate and display statistics for the train and eval splits.

Run this script from the project root with the course Conda environment:

    conda run -n tfx-env python pipeline/step3_statisticsgen.py
"""

import glob
import os

import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import statistics_pb2
from tfx.components import CsvExampleGen, StatisticsGen
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.proto import example_gen_pb2


# Derive paths from this file so the project can be moved to another computer.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PIPELINE_ROOT = os.path.join(PROJECT_DIR, "pipeline_output", "step3_statisticsgen")
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")
REPORT_PATH = os.path.join(PIPELINE_ROOT, "statistics_report.html")


def load_statistics(statistics_path):
    """Read a StatisticsGen FeatureStats.pb artifact."""
    statistics = statistics_pb2.DatasetFeatureStatisticsList()
    with open(statistics_path, "rb") as statistics_file:
        statistics.ParseFromString(statistics_file.read())
    return statistics


def print_feature_summary(split_name, statistics):
    """Print the main values stored in one split's statistics artifact."""
    print("\n===== {} FEATURE STATISTICS =====".format(split_name.upper()))

    for dataset in statistics.datasets:
        print("Examples:", dataset.num_examples)
        print(
            "{:<12} {:<12} {:>10} {:>10} {:>12} {:>12} {:>12}".format(
                "Feature", "Type", "Missing", "Present", "Min", "Max", "Mean"
            )
        )

        for feature in dataset.features:
            feature_name = feature.path.step[0] if feature.path.step else "unknown"
            common = feature.num_stats.common_stats if feature.HasField("num_stats") else (
                feature.string_stats.common_stats
            )
            missing = dataset.num_examples - common.num_non_missing
            minimum = maximum = mean = "-"
            feature_type = "categorical"

            if feature.HasField("num_stats"):
                feature_type = "numeric"
                minimum = "{:.3f}".format(feature.num_stats.min)
                maximum = "{:.3f}".format(feature.num_stats.max)
                mean = "{:.3f}".format(feature.num_stats.mean)

            print(
                "{:<12} {:<12} {:>10} {:>10} {:>12} {:>12} {:>12}".format(
                    feature_name,
                    feature_type,
                    missing,
                    common.num_non_missing,
                    minimum,
                    maximum,
                    mean,
                )
            )


def create_statistics_report(train_statistics, eval_statistics):
    """Write an interactive Facets view of both stored artifacts."""
    report_html = tfdv.get_statistics_html(
        lhs_statistics=train_statistics,
        rhs_statistics=eval_statistics,
        lhs_name="Train",
        rhs_name="Eval",
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
        report_file.write(report_html)


def main():
    """Run ExampleGen followed by StatisticsGen and display the results."""
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

    # This channel connects StatisticsGen directly to ExampleGen's output.
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"],
    )

    statistics_pipeline = pipeline.Pipeline(
        pipeline_name="step3_statisticsgen",
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
        raise RuntimeError("StatisticsGen did not create a statistics artifact")

    artifact_uri = max(artifact_dirs, key=os.path.getmtime)
    print("\nStatistics artifact URI:", artifact_uri)

    split_statistics = {}
    for split_name in ("train", "eval"):
        split_uri = os.path.join(artifact_uri, "Split-" + split_name)
        statistics_path = os.path.join(split_uri, "FeatureStats.pb")
        if not os.path.isfile(statistics_path):
            raise RuntimeError("Missing artifact: " + statistics_path)

        print("{} artifact:".format(split_name.capitalize()), statistics_path)
        split_statistics[split_name] = load_statistics(statistics_path)
        print_feature_summary(split_name, split_statistics[split_name])

    create_statistics_report(
        train_statistics=split_statistics["train"],
        eval_statistics=split_statistics["eval"],
    )
    print("\nInteractive statistics display:", REPORT_PATH)
    print("Open the HTML report to view distributions and histograms.")


if __name__ == "__main__":
    main()
