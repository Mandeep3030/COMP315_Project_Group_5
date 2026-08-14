# COMP_315 | SEC_401 | GROUP_5
# Mandeep Singh | Dennis Kozevnikoff
# =========

"""Step 5 component: apply feature engineering with TensorFlow Transform."""

# ===============
# STEP 5: TRANSFORM
# ===============

from tfx.components import Transform


def create_transform(examples, schema, preprocessing_module):
    """Create Transform using the project's preprocessing module."""
    return Transform(
        examples=examples,
        schema=schema,
        module_file=preprocessing_module,
    )
