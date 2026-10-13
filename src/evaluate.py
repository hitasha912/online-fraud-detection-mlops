import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)


def calculate_metrics(y_true, y_pred, y_prob):

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "ROC_AUC": roc_auc_score(
            y_true,
            y_prob
        ),

        "PR_AUC": average_precision_score(
            y_true,
            y_prob
        )
    }


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name,
    output_dir="results"
):

    os.makedirs(output_dir, exist_ok=True)

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_prob
    )

    print(
        f"\n===== {model_name} ====="
    )

    for metric, value in metrics.items():
        print(
            f"{metric}: {value:.4f}"
        )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            digits=4
        )
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            f"{model_name}_confusion_matrix.png"
        )
    )

    plt.show()
    plt.close()

    return metrics


def plot_roc_curves(
    trained_models,
    X_test,
    y_test,
    output_path="results/roc_curves.png"
):

    plt.figure(figsize=(7, 5))

    for name, model in trained_models.items():

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        fpr, tpr, _ = roc_curve(
            y_test,
            y_prob
        )

        auc = roc_auc_score(
            y_test,
            y_prob
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={auc:.4f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        "--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curves"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(output_path)

    plt.show()
    plt.close()


def plot_pr_curves(
    trained_models,
    X_test,
    y_test,
    output_path="results/pr_curves.png"
):

    plt.figure(figsize=(7, 5))

    for name, model in trained_models.items():

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        precision, recall, _ = (
            precision_recall_curve(
                y_test,
                y_prob
            )
        )

        pr_auc = average_precision_score(
            y_test,
            y_prob
        )

        plt.plot(
            recall,
            precision,
            label=(
                f"{name} "
                f"(PR-AUC={pr_auc:.4f})"
            )
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title(
        "Precision-Recall Curves"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(output_path)

    plt.show()
    plt.close()
