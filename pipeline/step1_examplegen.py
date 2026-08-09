import os
import glob
from tfx.components import CsvExampleGen
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.proto import example_gen_pb2

# Resolve paths relative to this file so renaming or moving the project is safe.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PIPELINE_ROOT = os.path.join(PROJECT_DIR, "pipeline_output", "step1_examplegen_test")
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")

output_config = example_gen_pb2.Output(
    split_config=example_gen_pb2.SplitConfig(
        splits=[
            example_gen_pb2.SplitConfig.Split(
                name="train",
                hash_buckets=2
            ),
            example_gen_pb2.SplitConfig.Split(
                name="eval",
                hash_buckets=1
            )
        ]
    )
)

example_gen = CsvExampleGen(
    input_base=DATA_DIR,
    output_config=output_config
)

examplegen_pipeline = pipeline.Pipeline(
    pipeline_name="examplegen_test",
    pipeline_root=PIPELINE_ROOT,
    components=[example_gen],
    metadata_connection_config=metadata.sqlite_metadata_connection_config(
        METADATA_PATH
    ),
)

LocalDagRunner().run(examplegen_pipeline)

print("\n===== EXAMPLEGEN RESULTS =====")

artifact_dirs = glob.glob(
    os.path.join(PIPELINE_ROOT, "CsvExampleGen", "examples", "*")
)
if not artifact_dirs:
    raise RuntimeError("ExampleGen completed without creating an artifact")

artifact_uri = max(artifact_dirs, key=os.path.getmtime)
print("Artifact URI:", artifact_uri)
print("Split names: train, eval")
print("Train URI:", os.path.join(artifact_uri, "Split-train"))
print("Eval URI:", os.path.join(artifact_uri, "Split-eval"))
