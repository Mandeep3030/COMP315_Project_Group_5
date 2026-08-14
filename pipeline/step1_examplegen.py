# COMP_315 | SEC_401 | GROUP_5
# Mandeep Singh | Dennis Kozevnikoff
# =========

"""Step 1 component: load CSV data and create train/eval splits."""

# ===============
# STEP 1: EXAMPLEGEN
# ===============

from tfx.components import CsvExampleGen
from tfx.proto import example_gen_pb2


def create_example_gen(data_root, train_hash_buckets=2, eval_hash_buckets=1):
    """Create the CSV input component with a reproducible train/eval split."""
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
    return CsvExampleGen(input_base=data_root, output_config=output_config)
