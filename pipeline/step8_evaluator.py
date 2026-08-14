# COMP_315 | SEC_401 | GROUP_5
# Mandeep Singh | Dennis Kozevnikoff
# =========

"""Step 8 component: evaluate and bless an acceptable candidate model."""

# ===============
# STEP 8: EVALUATOR
# ===============

import tensorflow_model_analysis as tfma
from google.protobuf import wrappers_pb2
from tfx.components import Evaluator


LABEL_KEY = "y"


def create_eval_config():
    """Define model metrics, data slices, and the accuracy threshold."""
    accuracy_threshold = tfma.MetricThreshold(
        value_threshold=tfma.GenericValueThreshold(
            lower_bound=wrappers_pb2.DoubleValue(value=0.80),
        ),
        change_threshold=tfma.GenericChangeThreshold(
            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
            absolute=wrappers_pb2.DoubleValue(value=0.0001),
        ),
    )
    return tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                label_key=LABEL_KEY,
                signature_name="serving_default",
                preprocessing_function_names=["transformed_labels"],
                prediction_key="probabilities",
            )
        ],
        slicing_specs=[
            tfma.SlicingSpec(),
            tfma.SlicingSpec(feature_keys=["marital"]),
            tfma.SlicingSpec(feature_keys=["job"]),
            tfma.SlicingSpec(feature_keys=["education"]),
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        per_slice_thresholds=[
                            tfma.PerSliceMetricThreshold(
                                slicing_specs=[tfma.SlicingSpec()],
                                threshold=accuracy_threshold,
                            )
                        ],
                    ),
                    tfma.MetricConfig(class_name="AUC"),
                ]
            )
        ],
    )


def create_evaluator(examples, model, baseline_model, eval_config=None):
    """Create Evaluator for the candidate and resolved baseline models."""
    if eval_config is None:
        eval_config = create_eval_config()
    return Evaluator(
        examples=examples,
        model=model,
        baseline_model=baseline_model,
        eval_config=eval_config,
    )
