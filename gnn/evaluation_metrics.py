"""
Evaluation metrics implementation module.
Contains common regression and ranking evaluation metrics.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def rmse(y_true, y_pred):
    """
    Root Mean Square Error

    Args:
        y_true: ground-truth values
        y_pred: predicted values

    Returns:
        float: RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mse(y_true, y_pred):
    """
    Mean Square Error

    Args:
        y_true: ground-truth values
        y_pred: predicted values

    Returns:
        float: MSE value
    """
    return mean_squared_error(y_true, y_pred)


def rmsle(y_true, y_pred):
    """
    Root Mean Square Logarithmic Error

    Args:
        y_true: ground-truth values
        y_pred: predicted values

    Returns:
        float: RMSLE value
    """
    # Ensure all values are non-negative
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)

    # Compute logarithms
    log_true = np.log1p(y_true)  # log1p(x) = log(1+x)
    log_pred = np.log1p(y_pred)

    # Compute the root mean square logarithmic error
    return np.sqrt(np.mean((log_true - log_pred) ** 2))


def male(y_true, y_pred):
    """
    Mean Absolute Logarithmic Error

    Args:
        y_true: ground-truth values
        y_pred: predicted values

    Returns:
        float: MALE value
    """
    # Ensure all values are non-negative
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)

    # Compute logarithms
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    # Compute the mean absolute logarithmic error
    return np.mean(np.abs(log_true - log_pred))


def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error

    Args:
        y_true: ground-truth values
        y_pred: predicted values

    Returns:
        float: MAPE value (as a percentage)
    """
    # Avoid division-by-zero errors
    mask = y_true != 0
    if not np.any(mask):
        return np.inf

    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    return np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100


def ndcg(y_true, y_pred, k=20):
    """
    Normalized Discounted Cumulative Gain

    Args:
        y_true: ground-truth relevance scores
        y_pred: predicted scores
        k: compute over the top k results; if None, compute over all

    Returns:
        float: NDCG value
    """
    if k is None:
        k = len(y_true)

    # Sort by predicted score
    sorted_indices = np.argsort(y_pred)[::-1]

    # Compute DCG
    dcg = 0
    for i in range(min(k, len(y_true))):
        idx = sorted_indices[i]
        dcg += y_true[idx] / np.log2(i + 2)

    # Compute IDCG (ideal DCG)
    ideal_sorted_indices = np.argsort(y_true)[::-1]
    idcg = 0
    for i in range(min(k, len(y_true))):
        idx = ideal_sorted_indices[i]
        idcg += y_true[idx] / np.log2(i + 2)

    # Avoid division by zero
    if idcg == 0:
        return 0

    return dcg / idcg


def hit_at_k(y_true, y_pred, k=10):
    """
    Hit@K metric

    Args:
        y_true: ground-truth labels (0 or 1)
        y_pred: predicted scores
        k: compute over the top k results

    Returns:
        float: Hit@K value
    """
    # Sort by predicted score and take the top k
    top_k_indices = np.argsort(y_pred)[::-1][:k]

    # Check whether there is a positive sample among the top k
    return 1 if np.any(y_true[top_k_indices] == 1) else 0


def spearman_correlation(y_true, y_pred):
    """
    Spearman rank correlation coefficient

    Args:
        y_true: ground-truth values
        y_pred: predicted values

    Returns:
        float: Spearman correlation coefficient
    """
    correlation, _ = spearmanr(y_true, y_pred)
    return correlation


def r2_score_metric(y_true, y_pred):
    """
    Coefficient of Determination (R²)

    Args:
        y_true: ground-truth values
        y_pred: predicted values

    Returns:
        float: R² value
    """
    return r2_score(y_true, y_pred)


def calculate_all_metrics(y_true, y_pred, k=10):
    """
    Compute all evaluation metrics

    Args:
        y_true: ground-truth values
        y_pred: predicted values
        k: the k value used for Hit@K and NDCG@K

    Returns:
        dict: a dictionary containing all metrics
    """
    metrics = {}

    # Regression metrics
    metrics['RMSE'] = rmse(y_true, y_pred)
    metrics['MSE'] = mse(y_true, y_pred)
    metrics['RMSLE'] = rmsle(y_true, y_pred)
    metrics['MALE'] = male(y_true, y_pred)
    metrics['MAPE'] = mape(y_true, y_pred)
    metrics['R2'] = r2_score_metric(y_true, y_pred)
    metrics['Spearman'] = spearman_correlation(y_true, y_pred)

    # Ranking metrics
    metrics[f'NDCG@{k}'] = ndcg(y_true, y_pred, k)
    metrics[f'Hit@{k}'] = hit_at_k(y_true, y_pred, k)

    return metrics


def print_metrics(metrics):
    """
    Print the evaluation metrics

    Args:
        metrics: metrics dictionary
    """
    print("=" * 50)
    print("Evaluation metric results:")
    print("=" * 50)

    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name:15}: {value:.6f}")
        else:
            print(f"{metric_name:15}: {value}")

    print("=" * 50)


# Usage example
if __name__ == "__main__":
    # Generate example data
    np.random.seed(42)
    n_samples = 100

    # Regression task example
    y_true_reg = np.random.randn(n_samples) * 10 + 50
    y_pred_reg = y_true_reg + np.random.randn(n_samples) * 2

    print("Regression task evaluation metrics:")
    reg_metrics = calculate_all_metrics(y_true_reg, y_pred_reg)
    print_metrics(reg_metrics)

    # Ranking task example
    y_true_rank = np.random.randint(0, 2, n_samples)  # labels of 0 or 1
    y_pred_rank = np.random.rand(n_samples)  # predicted scores

    print("\nRanking task evaluation metrics:")
    rank_metrics = calculate_all_metrics(y_true_rank, y_pred_rank)
    print_metrics(rank_metrics)
