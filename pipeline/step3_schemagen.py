# COMP_315 | SEC_401 | GROUP_5
# Mandeep Singh | Dennis Kozevnikoff
# =========

"""Step 3 component: infer a schema from the generated statistics."""

# ===============
# STEP 3: SCHEMAGEN
# ===============

from tfx.components import SchemaGen


def create_schema_gen(statistics):
    """Create SchemaGen for the StatisticsGen output channel."""
    return SchemaGen(statistics=statistics, infer_feature_shape=False)
