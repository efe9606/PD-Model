from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using classification metrics and ROC curve.
    """

    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_preds))

    auc = roc_auc_score(y_test, y_probs)
    print(f"AUC Score: {auc:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for PD Model")
    plt.legend()
    plt.grid(True)
    plt.show()