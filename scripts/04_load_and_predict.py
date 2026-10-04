
import mlflow

from sklearn.datasets import load_breast_cancer

mlflow.set_tracking_uri("sqlite:///mlflow.db")
def load_and_predict():
    """
    Load the model from MLflow Model Registry using the staging alias
    and predict one sample from each class.
    """

    MODEL_NAME = "cancer-classifier-prod"
    MODEL_ALIAS = "staging"

    # Mapping label number to class name
    CLASS_NAMES = {
        0: "malignant",
        1: "benign"
    }

    print(f"Loading model '{MODEL_NAME}' with alias '@{MODEL_ALIAS}'...")

    # Load model from MLflow Model Registry using Alias
    try:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        )
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(
            f"Please make sure a model version has "
            f"the alias '@{MODEL_ALIAS}' in the MLflow UI."
        )
        return

    # Load Breast Cancer dataset
    X, y = load_breast_cancer(
        return_X_y=True,
        as_frame=True
    )

    # Find the first sample of each class
    malignant_index = y[y == 0].index[0]
    benign_index = y[y == 1].index[0]

    samples = [
        ("malignant", malignant_index),
        ("benign", benign_index)
    ]

    print("-" * 50)

    for expected_class, index in samples:

        sample_data = X.loc[[index]]
        actual_label = y.loc[index]

        # Predict
        prediction = model.predict(sample_data)
        predicted_label = int(prediction[0])

        # Convert number to class name
        actual_name = CLASS_NAMES[actual_label]
        predicted_name = CLASS_NAMES[predicted_label]

        # Check whether prediction is correct
        is_correct = actual_name == predicted_name

        print(f"Actual Class    : {actual_name}")
        print(f"Predicted Class : {predicted_name}")
        print(f"Correct?        : {'Yes' if is_correct else 'No'}")
        print("-" * 50)


if __name__ == "__main__":
    load_and_predict()
