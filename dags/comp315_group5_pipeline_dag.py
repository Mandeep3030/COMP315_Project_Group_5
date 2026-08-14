# COMP_315 | SEC_401 | GROUP_5
# Mandeep Singh | Dennis Kozevnikoff
# =========

"""Apache Airflow DAG for the complete COMP315 Group 5 TFX pipeline."""

# ===============
# AIRFLOW DAG CONFIGURATION
# ===============

import datetime
import os
import sys

from tfx.orchestration.airflow.airflow_dag_runner import (
    AirflowDagRunner,
    AirflowPipelineConfig,
)


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from pipeline.pipeline_definition import create_pipeline


PIPELINE_NAME = "comp315_group5_pipeline"
PIPELINE_ROOT = os.path.join(PROJECT_DIR, "pipeline_output", "full_pipeline")
METADATA_PATH = os.path.join(PROJECT_DIR, "metadata", "metadata.sqlite")

AIRFLOW_CONFIG = {
    "start_date": datetime.datetime(2026, 1, 1),
    "schedule_interval": None,
    "catchup": False,
    "max_active_runs": 1,
}

LOGICAL_PIPELINE = create_pipeline(
    pipeline_name=PIPELINE_NAME,
    pipeline_root=PIPELINE_ROOT,
    data_root=os.path.join(PROJECT_DIR, "data"),
    preprocessing_module=os.path.join(
        PROJECT_DIR, "pipeline", "preprocessing.py"
    ),
    trainer_module=os.path.join(PROJECT_DIR, "pipeline", "step6_trainer.py"),
    serving_model_dir=os.path.join(PROJECT_DIR, "serving_model"),
    metadata_path=METADATA_PATH,
    train_steps=500,
    eval_steps=100,
    enable_cache=False,
)

# Airflow discovers this module-level DAG object.
DAG = AirflowDagRunner(
    AirflowPipelineConfig(AIRFLOW_CONFIG)
).run(LOGICAL_PIPELINE)
