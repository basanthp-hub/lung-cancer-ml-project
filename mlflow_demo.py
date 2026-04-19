import mlflow

mlflow.start_run()

mlflow.log_param("model", "lung_cancer_demo")
mlflow.log_metric("accuracy", 0.92)

mlflow.end_run()

print("MLflow run completed")