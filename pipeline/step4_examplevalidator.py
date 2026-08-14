# COMP_315 | SEC_401 | GROUP_5
# Mandeep Singh | Dennis Kozevnikoff
# =========

"""Step 4 component: detect data anomalies using the inferred schema."""

# ===============
# STEP 4: EXAMPLEVALIDATOR
# ===============

from tfx.components import ExampleValidator


def create_example_validator(statistics, schema):
    """Create ExampleValidator for the statistics and schema channels."""
    return ExampleValidator(statistics=statistics, schema=schema)
