# COMP_315 | SEC_401 | GROUP_5
# Mandeep Singh | Dennis Kozevnikoff
# =========

"""Step 2 component: calculate descriptive statistics for the data."""

# ===============
# STEP 2: STATISTICSGEN
# ===============

from tfx.components import StatisticsGen


def create_statistics_gen(examples):
    """Create StatisticsGen for the ExampleGen output channel."""
    return StatisticsGen(examples=examples)
