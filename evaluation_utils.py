"""Utility functions for model evaluation and metrics calculation."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any
import ast
import warnings
warnings.filterwarnings('ignore')


def parse_labels_string(labels_str: str) -> np.ndarray:
    """Parse string representation of numpy array back to numpy array.
    
    Args:
        labels_str: String representation of labels array
        
    Returns:
        numpy array of label probabilities
    """
    try:
        # Remove extra spaces and parse the string as a numpy array
        labels_str = labels_str.strip()
        if labels_str.startswith('[') and labels_str.endswith(']'):
            # Parse as literal array
            return np.array(ast.literal_eval(labels_str))
        else:
            # Handle numpy array string format
            return np.fromstring(labels_str.strip('[]'), sep=' ')
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing labels: {labels_str}")
        raise e


def extract_predicted_labels(df: pd.DataFrame) -> np.ndarray:
    """Extract predicted class labels from probability distributions.
    
    Args:
        df: DataFrame with 'labels' column containing probability arrays
        
    Returns:
        Array of predicted class indices (0: negative, 1: neutral, 2: positive)
    """
    predicted_labels = []
    
    for idx, labels_str in enumerate(df['labels']):
        try:
            probs = parse_labels_string(labels_str)
            predicted_class = np.argmax(probs)
            predicted_labels.append(predicted_class)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            # Default to neutral class (1) if parsing fails
            predicted_labels.append(1)
    
    return np.array(predicted_labels)


def create_train_val_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into training and validation sets.
    
    Args:
        df: Input dataframe
        test_size: Proportion of data for validation (default: 0.2)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df)
    """
    # Ensure reproducible results
    np.random.seed(random_state)
    
    # Get indices for random split
    n_samples = len(df)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Calculate split point
    split_point = int(n_samples * (1 - test_size))
    
    # Split indices
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]
    
    # Create train and validation sets
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    
    return train_df, val_df


def calculate_metrics_manual(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> Dict[str, Any]:
    """Calculate classification metrics manually without sklearn.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        num_classes: Number of classes (default: 3 for negative, neutral, positive)
        
    Returns:
        Dictionary containing calculated metrics
    """
    # Calculate confusion matrix manually
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix[true_label, pred_label] += 1
    
    # Calculate metrics for each class
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    
    for i in range(num_classes):
        # True positives, false positives, false negatives
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precision_per_class.append(precision)
        
        # Recall = TP / (TP + FN)  
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recall_per_class.append(recall)
        
        # F1 = 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_per_class.append(f1)
    
    # Calculate macro averages
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)
    
    # Calculate weighted averages (weighted by support)
    support_per_class = [confusion_matrix[i, :].sum() for i in range(num_classes)]
    total_support = sum(support_per_class)
    
    weighted_precision = sum(p * s for p, s in zip(precision_per_class, support_per_class)) / total_support
    weighted_recall = sum(r * s for r, s in zip(recall_per_class, support_per_class)) / total_support
    weighted_f1 = sum(f * s for f, s in zip(f1_per_class, support_per_class)) / total_support
    
    # Overall accuracy
    accuracy = np.trace(confusion_matrix) / confusion_matrix.sum()
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'support_per_class': support_per_class
    }


def print_evaluation_report(metrics: Dict[str, Any], class_names: List[str] = None) -> None:
    """Print a detailed evaluation report.
    
    Args:
        metrics: Dictionary containing calculated metrics
        class_names: Optional list of class names (default: ['Negative', 'Neutral', 'Positive'])
    """
    if class_names is None:
        class_names = ['Negative', 'Neutral', 'Positive']
    
    print("=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    
    print("\n" + "-" * 60)
    print("Per-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    print("-" * 60)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<12} {metrics['precision_per_class'][i]:<12.4f} "
              f"{metrics['recall_per_class'][i]:<12.4f} {metrics['f1_per_class'][i]:<12.4f} "
              f"{metrics['support_per_class'][i]:<12}")
    
    print("-" * 60)
    print(f"{'Macro Avg':<12} {metrics['macro_precision']:<12.4f} "
          f"{metrics['macro_recall']:<12.4f} {metrics['macro_f1']:<12.4f} "
          f"{sum(metrics['support_per_class']):<12}")
    print(f"{'Weighted Avg':<12} {metrics['weighted_precision']:<12.4f} "
          f"{metrics['weighted_recall']:<12.4f} {metrics['weighted_f1']:<12.4f} "
          f"{sum(metrics['support_per_class']):<12}")
    
    print("\n" + "-" * 60)
    print("Confusion Matrix:")
    print("-" * 60)
    print("Predicted ->")
    header = "Actual    " + "".join([f"{name[:8]:<10}" for name in class_names])
    print(header)
    
    for i, class_name in enumerate(class_names):
        row = f"{class_name[:8]:<10}"
        for j in range(len(class_names)):
            row += f"{metrics['confusion_matrix'][i, j]:<10}"
        print(row)
    
    print("=" * 60)


def analyze_class_distribution(y: np.ndarray, class_names: List[str] = None) -> Dict[str, Any]:
    """Analyze the distribution of classes in the dataset.
    
    Args:
        y: Array of class labels
        class_names: Optional list of class names
        
    Returns:
        Dictionary with class distribution information
    """
    if class_names is None:
        class_names = ['Negative', 'Neutral', 'Positive']
    
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    distribution = {}
    for class_idx, count in zip(unique, counts):
        if class_idx < len(class_names):
            distribution[class_names[class_idx]] = {
                'count': count,
                'percentage': (count / total) * 100
            }
    
    return {
        'total_samples': total,
        'distribution': distribution,
        'class_counts': dict(zip(unique, counts))
    }


def print_class_distribution(distribution: Dict[str, Any]) -> None:
    """Print class distribution analysis.
    
    Args:
        distribution: Dictionary from analyze_class_distribution
    """
    print("\n" + "=" * 40)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 40)
    print(f"Total Samples: {distribution['total_samples']}")
    print("-" * 40)
    print(f"{'Class':<12} {'Count':<8} {'Percentage':<12}")
    print("-" * 40)
    
    for class_name, info in distribution['distribution'].items():
        print(f"{class_name:<12} {info['count']:<8} {info['percentage']:<12.2f}%")
    print("=" * 40)