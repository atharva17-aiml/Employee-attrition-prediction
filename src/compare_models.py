import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from data_preprocessing import (
    load_data,
    prepare_data
)


DATA_PATH = "data/employee_attrition.csv"
MODEL_PATH = "models/employee_attrition_model.joblib"


def evaluate_model(name, model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    # Probability of Attrition = Yes
    probabilities = model.predict_proba(X_test)[:, 1]

    # Use threshold 0.40
    threshold = 0.40

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


def main():

    print("Loading dataset...")

    df = load_data(DATA_PATH)

    X, y, preprocessor = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ----------------------------------------
    # BASELINE MODEL
    # ----------------------------------------

    baseline_model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                C=1,
                solver="lbfgs",
                class_weight=None,
                max_iter=1000
            )
        )
    ])

    # ----------------------------------------
    # CLASS-BALANCED MODEL
    # ----------------------------------------

    balanced_model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                C=1,
                solver="lbfgs",
                class_weight="balanced",
                max_iter=1000
            )
        )
    ])

    # Evaluate both
    baseline_results = evaluate_model(
        "BASELINE LOGISTIC REGRESSION",
        baseline_model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    balanced_results = evaluate_model(
        "CLASS-BALANCED LOGISTIC REGRESSION",
        balanced_model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ----------------------------------------
    # COMPARISON
    # ----------------------------------------

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        f"{'Model':<35}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'ROC-AUC':<12}"
    )

    for result in [
        baseline_results,
        balanced_results
    ]:

        print(
            f"{result['model']:<35}"
            f"{result['accuracy']:<12.4f}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<12.4f}"
            f"{result['f1']:<12.4f}"
            f"{result['roc_auc']:<12.4f}"
        )

    print("\nComparison completed.")


if __name__ == "__main__":
    main()