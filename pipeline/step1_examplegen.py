import os
from tfx.components import CsvExampleGen
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.proto import example_gen_pb2

PROJECT_DIR = os.path.expanduser("~/COMP315/term_project")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PIPELINE_ROOT = os.path.join(PROJECT_DIR, "pipeline_output", "examplegen_test")

context = InteractiveContext(
    pipeline_root=PIPELINE_ROOT
)

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

context.run(example_gen)

artifacts = example_gen.outputs["examples"].get()

print("\n===== EXAMPLEGEN RESULTS =====")

for artifact in artifacts:
    print("Artifact URI:", artifact.uri)
    print("Split names:", artifact.split_names)
    print("Train URI:", os.path.join(artifact.uri, "Split-train"))
    print("Eval URI:", os.path.join(artifact.uri, "Split-eval"))
