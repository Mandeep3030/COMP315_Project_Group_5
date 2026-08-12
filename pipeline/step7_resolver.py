"""Step 7: resolve the latest model blessed by a previous pipeline run.

Run from the project root with the course Conda environment:

    conda run -n tfx-env python pipeline/step7_resolver.py

The first standalone run normally finds no baseline because Evaluator has not
created a ModelBlessing yet. In the final pipeline, Evaluator will consume the
Resolver outputs and will automatically bless the first model when these
outputs are empty.
"""

import os

from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.types import Channel, standard_artifacts


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_ROOT = os.path.join(PROJECT_DIR, "pipeline_output", "step7_resolver")
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")


def create_resolver():
    """Create a Resolver that returns the latest blessed model and blessing."""
    return resolver.Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=standard_artifacts.Model),
        model_blessing=Channel(type=standard_artifacts.ModelBlessing),
    ).with_id("latest_blessed_model_resolver")


def main():
    """Run a lightweight standalone test of the Resolver component."""
    model_resolver = create_resolver()

    resolver_pipeline = pipeline.Pipeline(
        pipeline_name="step7_resolver",
        pipeline_root=PIPELINE_ROOT,
        components=[model_resolver],
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            METADATA_PATH
        ),
    )
    LocalDagRunner().run(resolver_pipeline)

    print("\n===== RESOLVER RESULTS =====")
    print("Strategy: LatestBlessedModelStrategy")
    print("Metadata database:", METADATA_PATH)
    print("Model output key: model")
    print("Model blessing output key: model_blessing")
    print(
        "No baseline on the first run is expected; Evaluator will then bless "
        "the first candidate model automatically."
    )
    print("RESOLVER COMPLETED SUCCESSFULLY: True")


if __name__ == "__main__":
    main()
