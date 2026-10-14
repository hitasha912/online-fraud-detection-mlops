import os
import json
import time
import joblib
import pandas as pd
import argparse
from pathlib import Path

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

from src.preprocessing import create_feature_pipeline
from src.mlflow_tracking import log_registered_model


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

DATA_PATH = "data/onlinefraud.csv"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    type=str,
    default=DATA_PATH
)
args = parser.parse_args()
 
DATA_PATH = args.data


MODEL_DIR = "models"
RESULTS_DIR = "results"

# Maximum number of negative samples used for training
MAX_NEGATIVE_SAMPLES = 1_000_000

# Maximum number of samples used for hyperparameter tuning
MAX_TUNING_SAMPLES = 250_000

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)



# ============================================================
# LOAD DATA
# ============================================================

def load_data(path=DATA_PATH):
    """
    Load the fraud transaction dataset.
    """

    print("\nLoading dataset...")

    df = pd.read_csv(path)

    print("Dataset shape:", df.shape)

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):
    """
    Select required features and target.
    Remove raw transaction identifiers.
    """

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

    print("\nOriginal class distribution:")
    print(y.value_counts())

    print("\nOriginal fraud rate:")
    print(y.mean())

    return train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )


# ============================================================
# CREATE TRAINING SUBSET
# ============================================================

def create_training_subset(
    X_train,
    y_train,
    max_negative_samples=MAX_NEGATIVE_SAMPLES
):
    """
    Keep all fraud transactions and sample a manageable
    number of non-fraud transactions.

    This avoids training Random Forest on the entire
    6.3M-row dataset in local development.
    """

    train_data = X_train.copy()
    train_data["target"] = y_train.values

    fraud = train_data[
        train_data["target"] == 1
    ]

    non_fraud = train_data[
        train_data["target"] == 0
    ]

    print("\nFraud samples available:", len(fraud))
    print("Non-fraud samples available:", len(non_fraud))

    sample_size = min(
        max_negative_samples,
        len(non_fraud)
    )

    non_fraud_sample = non_fraud.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    )

    subset = pd.concat(
        [
            fraud,
            non_fraud_sample
        ],
        axis=0
    )

    subset = subset.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    y_subset = subset.pop("target")

    X_subset = subset

    print("\nTraining subset shape:")
    print(X_subset.shape)

    print("\nTraining subset class distribution:")
    print(y_subset.value_counts())

    print("\nTraining subset fraud rate:")
    print(y_subset.mean())

    return X_subset, y_subset


# ============================================================
# CREATE MODELS
# ============================================================

def create_models():
    """
    Create the three models required by the assignment.
    """

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
# EVALUATE MODEL
# ============================================================

def calculate_metrics(
    model,
    X_test,
    y_test
):
    """
    Calculate all important fraud-detection metrics.
    """

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    return {

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
            )
    }


# ============================================================
# TRAIN ALL BASELINE MODELS
# ============================================================

def train_models(
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Train Logistic Regression, Decision Tree
    and Random Forest.
    """

    models = create_models()

    trained_models = []

    results = []

    for name, model in models.items():

        print("\n" + "=" * 60)
        print(f"Training {name}")
        print("=" * 60)

        start_time = time.time()

        pipeline = create_feature_pipeline(
            model
        )

        pipeline.fit(
            X_train,
            y_train
        )

        training_time = (
            time.time() - start_time
        )

        metrics = calculate_metrics(
            pipeline,
            X_test,
            y_test
        )

        result = {
            "Model": name,
            **metrics,
            "Training_Time_sec": training_time
        }

        results.append(result)

        trained_models.append(
            (name, pipeline)
        )

        print("\nResults:")

        for metric, value in metrics.items():
            print(
                f"{metric}: {value:.6f}"
            )

        print(
            f"Training Time: "
            f"{training_time:.2f} seconds"
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="F1",
        ascending=False
    ).reset_index(drop=True)

    return trained_models, results_df


# ============================================================
# CREATE TUNING DATASET
# ============================================================

def create_tuning_subset(
    X_train,
    y_train,
    max_samples=MAX_TUNING_SAMPLES
):
    """
    Create a smaller stratified dataset for
    RandomizedSearchCV.
    """

    if len(X_train) <= max_samples:
        return X_train, y_train

    X_tune, _, y_tune, _ = train_test_split(
        X_train,
        y_train,
        train_size=max_samples,
        stratify=y_train,
        random_state=RANDOM_STATE
    )

    print("\nTuning dataset shape:")
    print(X_tune.shape)

    print("\nTuning class distribution:")
    print(y_tune.value_counts())

    return X_tune, y_tune


# ============================================================
# TUNE LOGISTIC REGRESSION
# ============================================================

def tune_logistic_regression(
    X_train,
    y_train
):

    print(
        "\nStarting Logistic Regression tuning..."
    )

    pipeline = create_feature_pipeline(
        LogisticRegression(
            class_weight="balanced",
            max_iter=300,
            random_state=RANDOM_STATE
        )
    )

    param_distributions = {

        "model__C": [
            0.01,
            0.1,
            1,
            10,
            100
        ],

        "model__solver": [
            "liblinear",
            "lbfgs"
        ]
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=5,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )

    search.fit(
        X_train,
        y_train
    )

    print(
        "\nBest Logistic Regression parameters:"
    )
    print(search.best_params_)

    print(
        "\nBest CV PR-AUC:",
        search.best_score_
    )

    return search


# ============================================================
# TUNE DECISION TREE
# ============================================================

def tune_decision_tree(
    X_train,
    y_train
):

    print(
        "\nStarting Decision Tree tuning..."
    )

    pipeline = create_feature_pipeline(
        DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    )

    param_distributions = {

        "model__max_depth": [
            6,
            8,
            10,
            12,
            16,
            20,
            None
        ],

        "model__min_samples_leaf": [
            1,
            2,
            5,
            10,
            20
        ],

        "model__min_samples_split": [
            2,
            5,
            10
        ]
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=5,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )

    search.fit(
        X_train,
        y_train
    )

    print(
        "\nBest Decision Tree parameters:"
    )
    print(search.best_params_)

    print(
        "\nBest CV PR-AUC:",
        search.best_score_
    )

    return search


# ============================================================
# TUNE RANDOM FOREST
# ============================================================

def tune_random_forest(
    X_train,
    y_train
):

    print(
        "\nStarting Random Forest tuning..."
    )

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
            150,
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
        param_distributions,
        n_iter=5,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )

    search.fit(
        X_train,
        y_train
    )

    print(
        "\nBest Random Forest parameters:"
    )
    print(search.best_params_)

    print(
        "\nBest CV PR-AUC:",
        search.best_score_
    )

    return search


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    model,
    model_name
):
    """
    Extract feature importance from the final model.
    """

    print(
        f"\nGenerating feature importance "
        f"for {model_name}..."
    )

    try:

        preprocessor = (
            model.named_steps[
                "preprocessor"
            ]
        )

        estimator = (
            model.named_steps[
                "model"
            ]
        )

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        if hasattr(
            estimator,
            "feature_importances_"
        ):

            importance = (
                estimator
                .feature_importances_
            )

        elif hasattr(
            estimator,
            "coef_"
        ):

            importance = abs(
                estimator.coef_[0]
            )

        else:

            print(
                "Feature importance "
                "not available."
            )

            return None

        importance_df = pd.DataFrame({

            "Feature": feature_names,

            "Importance": importance

        })

        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .reset_index(drop=True)
        )

        output_path = os.path.join(
            RESULTS_DIR,
            "feature_importance.csv"
        )

        importance_df.to_csv(
            output_path,
            index=False
        )

        print(
            "\nTop 20 important features:"
        )

        print(
            importance_df.head(20)
        )

        print(
            "\nFeature importance saved:",
            output_path
        )

        return importance_df

    except Exception as e:

        print(
            "Could not calculate "
            f"feature importance: {e}"
        )

        return None


# ============================================================
# SAVE MODEL
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
        "\nModel saved:",
        model_path
    )

    return model_path


# ============================================================
# VERIFY SAVED MODEL
# ============================================================

def verify_saved_model(
    model_path,
    X_test,
    y_test
):
    """
    Reload the saved model and verify
    that it can make predictions.
    """

    print(
        "\nVerifying saved model..."
    )

    loaded_model = joblib.load(
        model_path
    )

    predictions = (
        loaded_model.predict(
            X_test.head(10)
        )
    )

    print(
        "Reload successful."
    )

    print(
        "Sample predictions:",
        predictions
    )

    test_probabilities = (
        loaded_model
        .predict_proba(
            X_test.head(10)
        )[:, 1]
    )

    print(
        "Sample fraud probabilities:",
        test_probabilities
    )

    return loaded_model


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    df = load_data()


    # --------------------------------------------------------
    # 2. TRAIN / TEST SPLIT
    # --------------------------------------------------------

    (
        X_train_full,
        X_test,
        y_train_full,
        y_test
    ) = prepare_data(df)


    # --------------------------------------------------------
    # 3. CREATE MANAGEABLE TRAINING SUBSET
    # --------------------------------------------------------

    (
        X_train,
        y_train
    ) = create_training_subset(
        X_train_full,
        y_train_full
    )


    # --------------------------------------------------------
    # 4. TRAIN BASELINE MODELS
    # --------------------------------------------------------

    (
        trained_models,
        baseline_results
    ) = train_models(
        X_train,
        y_train,
        X_test,
        y_test
    )


    # --------------------------------------------------------
    # 5. SAVE BASELINE RESULTS
    # --------------------------------------------------------

    baseline_path = os.path.join(
        RESULTS_DIR,
        "model_comparison.csv"
    )

    baseline_results.to_csv(
        baseline_path,
        index=False
    )

    print(
        "\nBaseline model comparison:"
    )

    print(
        baseline_results
    )

    print(
        "\nBaseline results saved:",
        baseline_path
    )


    # --------------------------------------------------------
    # 6. SELECT BEST BASELINE MODEL
    # --------------------------------------------------------

    best_baseline_model = (
        baseline_results.iloc[0]["Model"]
    )

    print(
        "\nBest baseline model:",
        best_baseline_model
    )


    # --------------------------------------------------------
    # 7. CREATE SMALLER TUNING DATASET
    # --------------------------------------------------------

    (
        X_tune,
        y_tune
    ) = create_tuning_subset(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # 8. HYPERPARAMETER TUNING
    # --------------------------------------------------------

    tuning_results = {}


    # Logistic Regression
    logistic_search = (
        tune_logistic_regression(
            X_tune,
            y_tune
        )
    )

    tuning_results[
        "Logistic Regression"
    ] = logistic_search


    # Decision Tree
    tree_search = (
        tune_decision_tree(
            X_tune,
            y_tune
        )
    )

    tuning_results[
        "Decision Tree"
    ] = tree_search


    # Random Forest
    rf_search = (
        tune_random_forest(
            X_tune,
            y_tune
        )
    )

    tuning_results[
        "Random Forest"
    ] = rf_search


    # --------------------------------------------------------
    # 9. EVALUATE TUNED MODELS
    # --------------------------------------------------------

    tuned_results = []

    tuned_models = {}

    print(
        "\n" + "=" * 60
    )

    print(
        "TUNED MODEL RESULTS"
    )

    print(
        "=" * 60
    )

    for name, search in tuning_results.items():

        tuned_model = (
            search.best_estimator_
        )

        tuned_models[name] = (
            tuned_model
        )

        metrics = calculate_metrics(
            tuned_model,
            X_test,
            y_test
        )

        result = {
            "Model": name,
            **metrics,
            "Best_CV_PR_AUC":
                search.best_score_
        }

        tuned_results.append(
            result
        )

        print(
            f"\n{name}"
        )

        for metric, value in metrics.items():

            print(
                f"{metric}: {value:.6f}"
            )


    tuned_results_df = pd.DataFrame(
        tuned_results
    )

    tuned_results_df = (
        tuned_results_df
        .sort_values(
            by="F1",
            ascending=False
        )
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # 10. SAVE TUNED RESULTS
    # --------------------------------------------------------

    tuned_results_path = os.path.join(
        RESULTS_DIR,
        "tuned_model_comparison.csv"
    )

    tuned_results_df.to_csv(
        tuned_results_path,
        index=False
    )

    print(
        "\nTuned results:"
    )

    print(
        tuned_results_df
    )

    print(
        "\nTuned results saved:",
        tuned_results_path
    )


    # --------------------------------------------------------
    # 11. SELECT FINAL MODEL
    # --------------------------------------------------------

    final_model_name = (
        tuned_results_df.iloc[0]["Model"]
    )

    final_model = (
        tuned_models[
            final_model_name
        ]
    )

    print(
        "\nFinal selected model:",
        final_model_name
    )


    # --------------------------------------------------------
    # 12. SAVE FINAL MODEL
    # --------------------------------------------------------

    final_model_path = save_model(
        final_model,
        "fraud_detection_model"
    )


    # --------------------------------------------------------
    # 13. FEATURE IMPORTANCE
    # --------------------------------------------------------

    feature_importance = (
        save_feature_importance(
            final_model,
            final_model_name
        )
    )


    # --------------------------------------------------------
    # 14. SAVE MODEL METADATA
    # --------------------------------------------------------

    final_search = (
        tuning_results[
            final_model_name
        ]
    )

    registered_model_name = os.getenv(
        "MLFLOW_REGISTERED_MODEL_NAME",
        "fraud_detection_model"
    )
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{Path(__file__).resolve().parent.parent.joinpath('mlflow.db').as_posix()}"
    )
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "online-fraud-detection"
    )
    final_metrics = {
        key: float(value)
        for key, value in (
            tuned_results_df.iloc[0]
            .drop("Model")
            .to_dict()
            .items()
        )
    }

    model_info = log_registered_model(
        model=final_model,
        model_name=final_model_name,
        registered_model_name=registered_model_name,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        input_example=X_test.head(5),
        metrics=final_metrics,
        parameters={
            **final_search.best_params_,
            "random_state": RANDOM_STATE,
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
    )

    metadata = {

        "best_baseline_model":
            best_baseline_model,

        "final_model":
            final_model_name,

        "best_parameters":
            final_search.best_params_,

        "best_cv_pr_auc":
            float(
                final_search.best_score_
            ),

        "test_metrics":
            tuned_results_df.iloc[0]
            .drop("Model")
            .to_dict(),

        "training_samples":
            len(X_train),

        "test_samples":
            len(X_test),

        "model_path":
            final_model_path,

        "mlflow_run_id":
            model_info.run_id,

        "mlflow_model_uri":
            model_info.model_uri,

        "mlflow_registered_model_name":
            registered_model_name,

        "mlflow_registered_model_version":
            model_info.registered_model_version,

        "random_state":
            RANDOM_STATE
    }


    metadata_path = os.path.join(
        RESULTS_DIR,
        "model_metadata.json"
    )

    with open(
        metadata_path,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            default=str
        )

    print(
        "\nMetadata saved:",
        metadata_path
    )


    # --------------------------------------------------------
    # 15. VERIFY MODEL RELOADING
    # --------------------------------------------------------

    verify_saved_model(
        final_model_path,
        X_test,
        y_test
    )


    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "TRAINING PIPELINE COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 60
    )
