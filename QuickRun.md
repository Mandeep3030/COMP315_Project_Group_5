### Terminal 1: webserver

From the project folder:

cd COMP315_Project_Group_5

```bash
cd COMP315_Project_Group_5
conda activate tfx-env
export AIRFLOW_HOME="$PWD/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow webserver --port 8080

```

<http://localhost:8080>


### Terminal 2: scheduler

Open another terminal in the same project folder:

```bash
cd COMP315_Project_Group_5
conda activate tfx-env
export AIRFLOW_HOME="$PWD/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow scheduler

```