import json
import os

def load_training_history(history_path="data/history.json"):
    """
    Load the training history metrics.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, history_path)
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            return json.load(f)
    return None

def load_test_metrics(metrics_path="data/metrics.json"):
    """
    Load test set evaluation metrics and confusion matrix.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, metrics_path)
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            return json.load(f)
    return None
