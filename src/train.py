import os
import json
import time
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from preprocessing import create_feature_pipeline


RANDOM_STATE = 42

DATA_PATH = "data/onlinefraud.csv"

MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path=DATA_PATH):

    df = pd.read_csv(path)

    print(
        "Dataset shape:",
        df.shape
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    raw_features = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "nameDest"
    ]

    X = df[raw_features].copy()

    y = df["isFraud"].astype(int)

    return train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )


# ============================================================
# MODELS
# ============================================================

def create_models():

    return {

        "Logistic Regression":
            LogisticRegression(
                max_iter=300,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=12,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE
            )
    }


# ============================================================
# TRAIN BASELINE MODELS
# ============================================================

def train_models(
    X_train,
    y_train,
    X_test,
    y_test
):

    models = create_models()

    trained_models = {}

    results = []

    for name, model in models.items():

        print(
            f"\nTraining {name}..."
        )

        start_time = time.time()

        pipeline = create_feature_pipeline(
            model
        )

        pipeline.fit(
            X_train,
            y_train
        )

        elapsed = (
            time.time()
            - start_time
        )

        y_pred = pipeline.predict(
            X_test
        )

        y_prob = pipeline.predict_proba(
            X_test
        )[:, 1]

        result = {

            "Model": name,

            "Accuracy":
                accuracy_score(
                    y_test,
                    y_pred
                ),

            "Precision":
                precision_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "Recall":
                recall_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "F1":
                f1_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "ROC_AUC":
                roc_auc_score(
                    y_test,
                    y_prob
                ),

            "PR_AUC":
                average_precision_score(
                    y_test,
                    y_prob
                ),

            "Training_Time":
                elapsed
        }

        results.append(result)

        trained_models[name] = pipeline

        print(result)

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="F1",
        ascending=False
    ).reset_index(drop=True)

    return (
        trained_models,
        results_df
    )


# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

def tune_random_forest(
    X_train,
    y_train
):

    pipeline = create_feature_pipeline(
        RandomForestClassifier(
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
    )

    param_distributions = {

        "model__n_estimators": [
            100,
            200
        ],

        "model__max_depth": [
            8,
            12,
            16,
            None
        ],

        "model__min_samples_leaf": [
            2,
            5,
            10
        ],

        "model__max_features": [
            "sqrt",
            "log2"
        ]
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=5,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        random_state=RANDOM_STATE
    )

    search.fit(
        X_train,
        y_train
    )

    print(
        "\nBest parameters:"
    )

    print(
        search.best_params_
    )

    print(
        "\nBest CV PR-AUC:",
        search.best_score_
    )

    return search


# ============================================================
# SAVE FINAL MODEL
# ============================================================

def save_model(
    model,
    model_name="fraud_detection_model"
):

    model_path = os.path.join(
        MODEL_DIR,
        f"{model_name}.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "Model saved:",
        model_path
    )

    return model_path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df = load_data()

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = prepare_data(df)

    (
        trained_models,
        results_df
    ) = train_models(
        X_train,
        y_train,
        X_test,
        y_test
    )

    print(
        "\nModel Comparison:"
    )

    print(
        results_df
    )

    results_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "model_comparison.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Select best baseline model
    # --------------------------------------------------------

    best_name = results_df.iloc[0]["Model"]

    best_model = (
        trained_models[best_name]
    )

    print(
        "\nBest baseline model:",
        best_name
    )

    # --------------------------------------------------------
    # Tune Random Forest
    # --------------------------------------------------------

    print(
        "\nStarting Random Forest tuning..."
    )

    search = tune_random_forest(
        X_train,
        y_train
    )

    tuned_model = (
        search.best_estimator_
    )

    # --------------------------------------------------------
    # Save tuned model
    # --------------------------------------------------------

    final_model_path = save_model(
        tuned_model
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {

        "best_baseline_model":
            best_name,

        "tuned_model":
            "Random Forest",

        "best_parameters":
            search.best_params_,

        "cv_pr_auc":
            float(search.best_score_),

        "model_path":
            final_model_path
    }

    with open(
        os.path.join(
            RESULTS_DIR,
            "model_metadata.json"
        ),
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            default=str
        )

    print(
        "\nTraining completed."
    )
