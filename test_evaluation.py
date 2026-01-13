"""Test script to validate evaluation functionality without external dependencies."""

import sys
import os
import csv
import numpy as np

# Simple CSV reader without pandas
def read_csv_simple(filepath):
    """Read CSV file without pandas dependency."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

# Test parsing of labels
def test_label_parsing():
    """Test the label parsing functionality."""
    print("Testing label parsing...")
    
    # Sample label strings from the dataset
    test_labels = [
        "[0.07599603 0.72843176 0.19557217]",
        "[0.5045957  0.36809996 0.12730436]",
        "[0.45802507 0.31449795 0.22747697]"
    ]
    
    def parse_labels_simple(labels_str):
        """Simple label parser."""
        # Remove brackets and split by spaces
        labels_str = labels_str.strip('[]')
        values = [float(x) for x in labels_str.split()]
        return np.array(values)
    
    for i, label_str in enumerate(test_labels):
        try:
            parsed = parse_labels_simple(label_str)
            predicted_class = np.argmax(parsed)
            class_names = ['Negative', 'Neutral', 'Positive']
            print(f"  Sample {i+1}: {parsed} -> {class_names[predicted_class]}")
        except Exception as e:
            print(f"  Error parsing sample {i+1}: {e}")
    
    print("Label parsing test completed.\n")

# Test metrics calculation
def test_metrics_calculation():
    """Test the manual metrics calculation."""
    print("Testing metrics calculation...")
    
    # Sample predictions and ground truth
    y_true = np.array([0, 1, 2, 0, 1, 2, 1, 1, 0, 2])  # True labels
    y_pred = np.array([0, 1, 1, 0, 1, 2, 1, 2, 0, 2])  # Predicted labels
    
    def calculate_accuracy(y_true, y_pred):
        """Calculate accuracy manually."""
        return np.sum(y_true == y_pred) / len(y_true)
    
    def calculate_confusion_matrix(y_true, y_pred, num_classes=3):
        """Calculate confusion matrix manually."""
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for true_label, pred_label in zip(y_true, y_pred):
            cm[true_label, pred_label] += 1
        return cm
    
    # Test calculations
    accuracy = calculate_accuracy(y_true, y_pred)
    confusion_matrix = calculate_confusion_matrix(y_true, y_pred)
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    {confusion_matrix}")
    
    # Calculate per-class metrics
    class_names = ['Negative', 'Neutral', 'Positive']
    print(f"  Per-class analysis:")
    for i in range(3):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"    {class_names[i]}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    print("Metrics calculation test completed.\n")

# Test data loading
def test_data_loading():
    """Test loading the actual dataset."""
    print("Testing data loading...")
    
    data_path = "data/labeled_tweets.csv"
    if not os.path.exists(data_path):
        print(f"  Error: Data file not found at {data_path}")
        return False
    
    try:
        data = read_csv_simple(data_path)
        print(f"  Successfully loaded {len(data)} samples")
        
        if len(data) > 0:
            print(f"  Columns: {list(data[0].keys())}")
            
            # Show first few samples
            print(f"  Sample data:")
            for i in range(min(3, len(data))):
                labels_str = data[i].get('labels', 'N/A')
                text = data[i].get('lemmatized_text', 'N/A')[:50] + "..."
                print(f"    {i+1}. Labels: {labels_str}")
                print(f"       Text: {text}")
        
        print("Data loading test completed.\n")
        return True
        
    except Exception as e:
        print(f"  Error loading data: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("TESTING EVALUATION FUNCTIONALITY")
    print("="*60)
    
    # Test 1: Label parsing
    test_label_parsing()
    
    # Test 2: Metrics calculation
    test_metrics_calculation()
    
    # Test 3: Data loading
    success = test_data_loading()
    
    print("="*60)
    if success:
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("The evaluation system should work once pandas is installed.")
    else:
        print("SOME TESTS FAILED")
        print("Please check the errors above.")
    print("="*60)

if __name__ == "__main__":
    main()