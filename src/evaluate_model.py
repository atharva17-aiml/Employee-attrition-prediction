import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    precision_recall_curve
)

from data_preprocessing import (
    load_data,
    prepare_data
)


DATA_PATH = "data/employee_attrition.csv"
MODEL_PATH = "models/employee_attrition_model.joblib"


def evaluate():

    print("========================================")
    print("MODEL EVALUATION")
    print("========================================")

    # Load dataset
    df = load_data(DATA_PATH)

    # Prepare data
    X, y, _ = prepare_data(df)

    # Same split used during training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Load trained model
    print("\nLoading trained model...")

    model = joblib.load(MODEL_PATH)

    # Probability of Attrition = Yes
    y_probability = model.predict_proba(X_test)[:, 1]

    # ========================================
    # ROC-AUC
    # ========================================

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    print("\n========================================")
    print("ROC-AUC")
    print("========================================")

    print(f"ROC-AUC: {roc_auc:.4f}")

    # ========================================
    # THRESHOLD ANALYSIS
    # ========================================

    print("\n========================================")
    print("THRESHOLD ANALYSIS")
    print("========================================")

    thresholds = [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50
    ]

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1 Score':<12}"
        f"{'Accuracy':<12}"
    )

    for threshold in thresholds:

        y_pred_threshold = (
            y_probability >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            y_pred_threshold,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred_threshold,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred_threshold,
            zero_division=0
        )

        accuracy = accuracy_score(
            y_test,
            y_pred_threshold
        )

        print(
            f"{threshold:<12.2f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
            f"{accuracy:<12.4f}"
        )

    # ========================================
    # FINAL THRESHOLD
    # ========================================

    # Lower threshold to improve attrition detection
    threshold = 0.40

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    print("\n========================================")
    print("FINAL MODEL PERFORMANCE")
    print("========================================")

    print(f"Decision Threshold: {threshold}")

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    # ========================================
    # CLASSIFICATION REPORT
    # ========================================

    print("\n========================================")
    print("CLASSIFICATION REPORT")
    print("========================================")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # ========================================
    # CONFUSION MATRIX
    # ========================================

    print("\n========================================")
    print("CONFUSION MATRIX")
    print("========================================")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Attrition", "Attrition"]
    ).plot()

    plt.title(
        f"Employee Attrition - Threshold {threshold}"
    )

    plt.tight_layout()
    plt.show()

    # ========================================
    # ROC CURVE
    # ========================================

    RocCurveDisplay.from_predictions(
        y_test,
        y_probability
    )

    plt.title(
        "Employee Attrition - ROC Curve"
    )

    plt.tight_layout()
    plt.show()

    # ========================================
    # PRECISION-RECALL CURVE
    # ========================================

    precision_values, recall_values, _ = (
        precision_recall_curve(
            y_test,
            y_probability
        )
    )

    plt.plot(
        recall_values,
        precision_values
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        "Employee Attrition - Precision-Recall Curve"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate()