from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class EvaluationMetrics:
    """
    Model tahminlerini gerçek etiketlerle karşılaştırır.
    """

    @staticmethod
    def decision_to_label(decision: str) -> int:
        if decision == "anomaly":
            return 1
        return 0

    @staticmethod
    def evaluate(y_true, predictions):
        y_pred = [
            EvaluationMetrics.decision_to_label(prediction["decision"])
            for prediction in predictions
        ]

        min_len = min(len(y_true), len(y_pred))

        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }