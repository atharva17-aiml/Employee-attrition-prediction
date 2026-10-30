import os
import sys
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Allow Python to find the src package
sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

from src.data_preprocessing import (
    load_data,
    prepare_data
)


DATA_PATH = "data/employee_attrition.csv"
MODEL_PATH = "models/employee_attrition_model.joblib"


def train():

    print("========================================")
    print("EMPLOYEE ATTRITION PREDICTION")
    print("========================================")

    # Load dataset
    print("\nLoading dataset...")

    df = load_data(DATA_PATH)

    print("Dataset shape:", df.shape)

    # Prepare data
    print("\nPreparing data...")

    X, y, preprocessor = prepare_data(df)

    print("Features:", X.shape)
    print("Target:", y.shape)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining data:", X_train.shape)
    print("Testing data:", X_test.shape)

    # Create ML pipeline
    model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ])

    # Hyperparameter grid
    parameter_grid = {

        "classifier__C": [
            0.01,
            0.1,
            1,
            10,
            100
        ],

        "classifier__solver": [
            "liblinear",
            "lbfgs"
        ],

        "classifier__class_weight": [
            None,
            "balanced"
        ]
    }

    # Stratified 5-fold cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1
    )

    # Train
    print("\n========================================")
    print("TRAINING MODEL")
    print("========================================")

    grid_search.fit(
        X_train,
        y_train
    )

    # Results
    print("\n========================================")
    print("BEST MODEL")
    print("========================================")

    print(
        "Best Parameters:"
    )

    print(
        grid_search.best_params_
    )

    print(
        "\nBest CV ROC-AUC:",
        round(
            grid_search.best_score_,
            4
        )
    )

    # Save model
    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        grid_search.best_estimator_,
        MODEL_PATH
    )

    print(
        "\nModel saved to:",
        MODEL_PATH
    )

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    train()