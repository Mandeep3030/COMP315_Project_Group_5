# COMP315 Group 5 — TFX and Apache Airflow Pipeline

This project builds an automated machine-learning pipeline for the Bank
Marketing dataset. It uses TensorFlow Extended (TFX) for data validation,
feature engineering, training, evaluation, and model deployment. Apache
Airflow provides the browser interface used to trigger and monitor the
pipeline.

## Pipeline overview

The Airflow DAG is named `comp315_group5_pipeline` and contains these nine
executable TFX components:

1. `CsvExampleGen` reads `data/bank_tfx.csv` and creates train/eval splits.
2. `StatisticsGen` calculates feature statistics.
3. `SchemaGen` infers the dataset schema.
4. `ExampleValidator` checks the data for anomalies.
5. `Transform` applies the transformations in `pipeline/preprocessing.py`.
6. `Trainer` trains a Keras binary-classification model and writes TensorBoard
   logs.
7. `Resolver` locates the latest previously blessed model, when one exists.
8. `Evaluator` uses TFMA to calculate Binary Accuracy and AUC overall and for
   configured data slices, then decides whether to bless the model.
9. `Pusher` deploys a blessed model into `serving_model/`.

The assignment's tenth step is the reusable `create_pipeline()` function in
`pipeline/pipeline_definition.py`. It defines the workflow but is not a
separate Airflow task.

## Expected software versions

The project was developed and tested with:

- Linux or WSL Ubuntu
- Python 3.7.16
- TensorFlow 2.11.0
- TFX 1.12.0
- Apache Airflow 2.2.5

These packages have strict compatibility requirements. Use the supplied
course Conda environment if possible, and do not upgrade them individually.
The commands below assume that the environment is named `tfx-env`.

## First-time Airflow setup

Extract the ZIP, open a terminal in the extracted project folder, and activate
the environment:

```bash
cd /path/to/COMP315_Project_Group_5
conda activate tfx-env
```

Set Airflow's local directories. Using `$PWD` keeps these commands portable,
regardless of where the teammate extracted the project:

```bash
export AIRFLOW_HOME="$PWD/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
```

These `export` commands must be repeated in every new terminal used to run
Airflow. `AIRFLOW_HOME` stores Airflow's configuration, run database, and logs.
`AIRFLOW__CORE__DAGS_FOLDER` tells Airflow where to find this project's DAG.

Initialize Airflow's local database:

```bash
airflow db init
```

Create a browser login. Choose a private password when prompted:

```bash
airflow users create \
  --username admin \
  --firstname COMP315 \
  --lastname Student \
  --role Admin \
  --email admin@example.com
```

This initialization and user creation are normally required only once for an
extracted copy of the project.

Confirm that Airflow can discover the project DAG:

```bash
airflow dags list | grep comp315_group5_pipeline
```

The output should contain `comp315_group5_pipeline`. Initial DAG loading may
take a little time because TFX imports TensorFlow and packages component code.

## Run Airflow in the browser

Airflow needs two processes. Keep both terminals open while using it.

### Terminal 1: webserver

From the project folder:

```bash
conda activate tfx-env
export AIRFLOW_HOME="$PWD/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow webserver --port 8080
```

The webserver supplies the browser interface. Open this address and sign in
with the account created above:

<http://localhost:8080>

If port 8080 is already occupied, use another port, for example
`airflow webserver --port 8081`, and open `http://localhost:8081`.

### Terminal 2: scheduler

Open another terminal in the same project folder:

```bash
conda activate tfx-env
export AIRFLOW_HOME="$PWD/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow scheduler
```

The scheduler decides when each task is ready and executes the tasks in
dependency order. The browser may open without it, but pipeline tasks will not
run.

## Trigger the pipeline

1. Open the Airflow **DAGs** page.
2. Find `comp315_group5_pipeline`.
3. Turn its toggle on to unpause it.
4. Click the DAG name and open **Graph View**.
5. Click the **Trigger DAG** play button once and confirm.
6. Watch the components progress through queued, running, and success states.

The DAG uses `schedule_interval=None`, so it runs only when manually triggered.
It also uses `max_active_runs=1`, so only one complete pipeline run is active at
a time. The local SQLite installation uses `SequentialExecutor`, which runs
tasks one at a time.

Common task colors are:

- green: succeeded
- red: failed
- light blue: running
- gray: not yet run or waiting for an upstream task

Click a task in Graph View and select **Log** to inspect its output. Do not
trigger another run until the current run finishes.

## Outputs

Generated runtime files are intentionally kept outside the source modules:

- `pipeline_output/full_pipeline/`: TFX component artifacts
- `metadata/metadata.sqlite`: TFX ML Metadata database
- `serving_model/`: model copied by Pusher after blessing
- `.airflow/`: Airflow database, configuration, and logs
- `screenshots/`: project execution evidence

Airflow's database and the TFX metadata database are different. Airflow tracks
DAG/task execution, while TFX metadata tracks datasets, models, evaluations,
and model blessings.

## Stop and restart Airflow

Press `Ctrl+C` in both the scheduler and webserver terminals to stop them.
This does not remove pipeline results. To restart later, repeat the environment
variables in two terminals and run `airflow webserver --port 8080` and
`airflow scheduler`; do not repeat `airflow db init` or create the user again.

## Common messages and problems

- Python 3.7, `distutils`, Setuptools, Blinker, CUDA, and TensorRT warnings are
  expected in this course environment. Missing CUDA/TensorRT libraries mean
  TensorFlow will use the CPU; they are not pipeline failures.
- `No data found` from `airflow users list` means an Airflow user still needs
  to be created.
- `no such table` usually means `airflow db init` was not run with the same
  `AIRFLOW_HOME` value.
- If the project DAG is absent, confirm `pwd`, repeat the three `export`
  commands, and run `airflow dags list` again.
- If a task turns red, open that task's log in Graph View and inspect the first
  traceback and the final error message.

## Important source files

- `dags/comp315_group5_pipeline_dag.py`: Airflow entry point and run settings
- `pipeline/pipeline_definition.py`: reusable complete TFX pipeline
- `pipeline/preprocessing.py`: TensorFlow Transform feature engineering
- `pipeline/step6_trainer.py`: Keras model, training, TensorBoard, and export
- `pipeline/step8_evaluator.py`: TFMA metrics, slices, and blessing thresholds
- `notebooks/tfma_analysis.ipynb`: model and fairness analysis
