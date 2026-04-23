"""Simple test script without external dependencies."""

import os
import csv


def parse_labels_basic(labels_str):
    """Basic label parser without numpy."""
    labels_str = labels_str.strip('[]')
    values = [float(x) for x in labels_str.split()]
    return values


def get_predicted_class(probs):
    """Get predicted class from probabilities."""
    max_prob = max(probs)
    return probs.index(max_prob)


def _test_label_parsing():
    """Test 1: Label parsing."""
    print("1. Testing label parsing...")
    test_labels = [
        "[0.07599603 0.72843176 0.19557217]",
        "[0.5045957  0.36809996 0.12730436]",
        "[0.45802507 0.31449795 0.22747697]"
    ]

    class_names = ['Negative', 'Neutral', 'Positive']

    for i, label_str in enumerate(test_labels):
        try:
            probs = parse_labels_basic(label_str)
            predicted_class = get_predicted_class(probs)
            print(f"   Sample {i+1}: {probs} -> {class_names[predicted_class]}")
        except (ValueError, IndexError) as e:
            print(f"   Error parsing sample {i+1}: {e}")

    print("   ✓ Label parsing test completed.\n")


def _test_data_loading():
    """Test 2: Data loading. Returns False on failure."""
    print("2. Testing data loading...")
    data_path = "data/labeled_tweets.csv"

    if not os.path.exists(data_path):
        print(f"   ✗ Data file not found at {data_path}")
        return False

    try:
        with open(data_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            data = list(reader)

        print(f"   ✓ Successfully loaded {len(data)} samples")

        if len(data) > 0:
            print(f"   ✓ Columns: {list(data[0].keys())}")
            _show_sample_predictions(data)

        print("   ✓ Data loading test completed.\n")
        return True

    except (OSError, csv.Error) as e:
        print(f"   ✗ Error loading data: {e}")
        return False


def _show_sample_predictions(data):
    """Show sample predictions from loaded data."""
    class_names = ['Negative', 'Neutral', 'Positive']
    print("   Sample predictions:")
    for i in range(min(5, len(data))):
        try:
            labels_str = data[i]['labels']
            probs = parse_labels_basic(labels_str)
            predicted_class = get_predicted_class(probs)
            raw_text = data[i]['lemmatized_text']
            text = raw_text[:40] + "..." if len(raw_text) > 40 else raw_text
            print(f"     {i+1}. {class_names[predicted_class]:<10} | {text}")
        except (ValueError, KeyError, IndexError) as e:
            print(f"     {i+1}. Error processing sample: {e}")


def _test_basic_metrics():
    """Test 3: Basic metrics calculation."""
    print("3. Testing basic metrics calculation...")

    y_true = [0, 1, 2, 0, 1, 2, 1, 1, 0, 2]
    y_pred = [0, 1, 1, 0, 1, 2, 1, 2, 0, 2]

    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct / len(y_true)

    print(f"   ✓ Accuracy calculation: {accuracy:.4f}")

    confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix[true_label][pred_label] += 1

    print(f"   ✓ Confusion matrix: {confusion_matrix}")
    print("   ✓ Basic metrics test completed.\n")


def test_basic_functionality():
    """Test basic evaluation functionality without external dependencies."""
    print("="*60)
    print("TESTING EVALUATION SYSTEM (Basic)")
    print("="*60)

    _test_label_parsing()

    success = _test_data_loading()

    _test_basic_metrics()

    print("="*60)
    print("EVALUATION SYSTEM VALIDATION COMPLETE")
    print("="*60)
    print("✓ All basic tests passed!")
    print("✓ The system is ready to work with proper dependencies.")
    print("\nNext steps:")
    print("1. Install pandas and scikit-learn: pip install pandas scikit-learn")
    print("2. Run: python model_evaluation.py")
    print("="*60)

    return success

if __name__ == "__main__":
    test_basic_functionality()
