import tensorflow as tf
import tensorflow_transform as tft

NUMERIC_FEATURES = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

LABEL_KEY = "y"


def transformed_name(key):
    return key + "_xf"


def fill_in_missing(x):
    """Convert SparseTensor features into normal dense tensors."""
    if not isinstance(x, tf.sparse.SparseTensor):
        return x

    default_value = "" if x.dtype == tf.string else 0

    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(
                x.indices,
                x.values,
                [x.dense_shape[0], 1]
            ),
            default_value
        ),
        axis=1
    )


def preprocessing_fn(inputs):
    outputs = {}

    # Scale numerical features using z-score normalization.
    for key in NUMERIC_FEATURES:
        dense_value = fill_in_missing(inputs[key])

        outputs[transformed_name(key)] = tft.scale_to_z_score(
            dense_value
        )

    # Convert categorical strings into integer vocabulary IDs.
    for key in CATEGORICAL_FEATURES:
        dense_value = fill_in_missing(inputs[key])

        outputs[transformed_name(key)] = (
            tft.compute_and_apply_vocabulary(dense_value)
        )

    # Convert target label: no = 0, yes = 1.
    label = fill_in_missing(inputs[LABEL_KEY])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        tf.equal(label, "yes"),
        tf.int64
    )

    return outputs
