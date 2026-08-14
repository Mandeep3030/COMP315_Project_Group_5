# COMP_315 | SEC_401 | GROUP_5
# Mandeep Singh | Dennis Kozevnikoff
# =========

"""Step 7 component: find the latest model blessed by an earlier run."""

# ===============
# STEP 7: RESOLVER
# ===============

from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)
from tfx.types import Channel, standard_artifacts


def create_resolver():
    """Create a Resolver for the latest blessed model and its blessing."""
    return resolver.Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=standard_artifacts.Model),
        model_blessing=Channel(type=standard_artifacts.ModelBlessing),
    ).with_id("latest_blessed_model_resolver")
