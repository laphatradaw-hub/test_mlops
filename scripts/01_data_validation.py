import mlflow
from sklearn.datasets import load_breast_cancer

mlflow.set_tracking_uri("sqlite:///mlflow.db")


def validate_data():
    """
    Loads the wine dataset, performs basic validation checks,
    and logs the results to MLflow.
    """
    # Set the experiment name for this step
    mlflow.set_experiment("breast_cancer - Data Validation")

    with mlflow.start_run():
        print("Starting data validation run...")
        mlflow.set_tag("ml.step", "data_validation")

        # 1. Load data as a Pandas DataFrame
        d = load_breast_cancer(as_frame=True)
        df = d.frame
        print(d.frame.shape)
        print(d.target_names)
        print(d.frame["target"].value_counts(normalize=True))
        print("Data loaded successfully.")

        # 2. Perform simple validation checks
        num_rows, num_cols = df.shape
        num_classes = df["target"].nunique()
        missing_values = df.isnull().sum().sum()
        balance = df["target"].value_counts(normalize=True).min()

        print(f"Dataset shape: {num_rows} rows, {num_cols} columns")
        print(f"Number of classes: {num_classes}")
        print(f"Missing values: {missing_values}")
        print(f"class balance(min): {balance:.4f}")

        # 3. Log validation results to MLflow
        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("num_cols", num_cols)
        mlflow.log_metric("missing_values", missing_values)
        mlflow.log_param("num_classes", num_classes)

        # Check if the data passes our defined criteria
        validation_status = "Success"
        if missing_values > 0 or num_classes != 2 or balance < 0.20:
            validation_status = "Failed"

        mlflow.log_param("validation_status", validation_status)
        print(f"Validation status: {validation_status}")

        # 4. ทำให้ CI จับได้จริง — ต้องคืน exit code ที่ไม่ใช่ 0 เมื่อข้อมูลไม่ผ่าน
        #    ถ้าแค่ print ว่า Failed แล้วจบปกติ step ใน GitHub Actions จะยังขึ้นเขียว
        if validation_status == "Failed":
            raise SystemExit("Data validation failed — หยุด pipeline ไม่ให้ไปขั้นถัดไป")

        print("Data validation run finished.")


if __name__ == "__main__":
    validate_data()
