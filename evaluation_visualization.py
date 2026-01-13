"""Simple text-based visualization for evaluation results."""

def create_text_bar_chart(data, title="Bar Chart", max_width=50):
    """Create a simple text-based bar chart.
    
    Args:
        data: Dictionary with labels as keys and values as data
        title: Chart title
        max_width: Maximum width of bars in characters
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    if not data:
        print("No data to display")
        return
    
    # Find max value for scaling
    max_value = max(data.values())
    
    for label, value in data.items():
        # Calculate bar length
        bar_length = int((value / max_value) * max_width) if max_value > 0 else 0
        bar = "█" * bar_length
        
        # Format the display
        print(f"{label:<15} |{bar:<{max_width}} {value:.2f}")
    print()


def create_confusion_matrix_text(confusion_matrix, class_names):
    """Create a text representation of confusion matrix.
    
    Args:
        confusion_matrix: 2D list/array with confusion matrix values
        class_names: List of class names
    """
    print("\nConfusion Matrix (Text Format)")
    print("=" * 40)
    print("Rows = Actual, Columns = Predicted")
    print()
    
    # Header
    header = "Actual\\Pred  "
    for name in class_names:
        header += f"{name[:8]:<10}"
    print(header)
    print("-" * len(header))
    
    # Matrix rows
    for i, actual_class in enumerate(class_names):
        row = f"{actual_class[:8]:<12} "
        for j in range(len(class_names)):
            row += f"{confusion_matrix[i][j]:<10}"
        print(row)
    print()


def visualize_evaluation_results(metrics, class_names=None):
    """Create text-based visualizations for evaluation results.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        class_names: List of class names
    """
    if class_names is None:
        class_names = ['Negative', 'Neutral', 'Positive']
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS VISUALIZATION")
    print("=" * 60)
    
    # Overall metrics
    print(f"\nOverall Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"Macro F1 Score: {metrics.get('macro_f1', 0):.4f}")
    print(f"Weighted F1 Score: {metrics.get('weighted_f1', 0):.4f}")
    
    # Per-class precision chart
    if 'precision_per_class' in metrics:
        precision_data = {
            class_names[i]: metrics['precision_per_class'][i] 
            for i in range(len(class_names))
        }
        create_text_bar_chart(precision_data, "Precision by Class")
    
    # Per-class recall chart  
    if 'recall_per_class' in metrics:
        recall_data = {
            class_names[i]: metrics['recall_per_class'][i] 
            for i in range(len(class_names))
        }
        create_text_bar_chart(recall_data, "Recall by Class")
    
    # Per-class F1 chart
    if 'f1_per_class' in metrics:
        f1_data = {
            class_names[i]: metrics['f1_per_class'][i] 
            for i in range(len(class_names))
        }
        create_text_bar_chart(f1_data, "F1 Score by Class")
    
    # Support (sample count) chart
    if 'support_per_class' in metrics:
        support_data = {
            class_names[i]: metrics['support_per_class'][i] 
            for i in range(len(class_names))
        }
        create_text_bar_chart(support_data, "Sample Count by Class")
    
    # Confusion matrix
    if 'confusion_matrix' in metrics:
        create_confusion_matrix_text(metrics['confusion_matrix'], class_names)
    
    print("=" * 60)


def create_performance_summary(metrics, threshold_good=0.7, threshold_excellent=0.85):
    """Create a performance summary with recommendations.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        threshold_good: Threshold for good performance
        threshold_excellent: Threshold for excellent performance
    """
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    
    accuracy = metrics.get('accuracy', 0)
    macro_f1 = metrics.get('macro_f1', 0)
    
    # Overall assessment
    print("\nOverall Assessment:")
    if accuracy >= threshold_excellent and macro_f1 >= threshold_excellent:
        print("🎉 EXCELLENT: Model performance is excellent!")
        grade = "A"
    elif accuracy >= threshold_good and macro_f1 >= threshold_good:
        print("✅ GOOD: Model performance is good with room for improvement.")
        grade = "B"
    elif accuracy >= 0.5 and macro_f1 >= 0.5:
        print("⚠️  MODERATE: Model performance is moderate, needs improvement.")
        grade = "C"
    else:
        print("❌ POOR: Model performance is poor, significant improvement needed.")
        grade = "D"
    
    print(f"Performance Grade: {grade}")
    
    # Detailed analysis
    print(f"\nDetailed Analysis:")
    print(f"• Accuracy: {accuracy:.4f} ({'Excellent' if accuracy >= threshold_excellent else 'Good' if accuracy >= threshold_good else 'Needs Improvement'})")
    print(f"• Macro F1: {macro_f1:.4f} ({'Excellent' if macro_f1 >= threshold_excellent else 'Good' if macro_f1 >= threshold_good else 'Needs Improvement'})")
    
    # Class balance analysis
    if 'f1_per_class' in metrics:
        f1_scores = metrics['f1_per_class']
        f1_min, f1_max = min(f1_scores), max(f1_scores)
        f1_range = f1_max - f1_min
        
        print(f"• Class Balance: {f1_range:.4f} F1 range ({'Balanced' if f1_range < 0.15 else 'Moderate Imbalance' if f1_range < 0.3 else 'Significant Imbalance'})")
    
    # Recommendations
    print(f"\nRecommendations:")
    
    if accuracy < threshold_good:
        print("📈 Improve overall accuracy:")
        print("   - Collect more training data")
        print("   - Try different model architectures")
        print("   - Improve data preprocessing")
    
    if macro_f1 < threshold_good:
        print("⚖️  Improve class balance:")
        print("   - Balance the training dataset") 
        print("   - Use class weights in training")
        print("   - Apply oversampling/undersampling")
    
    if 'f1_per_class' in metrics:
        f1_scores = metrics['f1_per_class']
        class_names = ['Negative', 'Neutral', 'Positive']
        
        poor_classes = [class_names[i] for i, f1 in enumerate(f1_scores) if f1 < 0.6]
        if poor_classes:
            print(f"🎯 Focus on improving: {', '.join(poor_classes)}")
            print("   - Collect more examples for these classes")
            print("   - Review labeling quality for these classes")
    
    print("✅ Always consider:")
    print("   - Cross-validation for robust evaluation")
    print("   - Human evaluation of a sample of predictions") 
    print("   - Regular model updates with new data")
    
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    sample_metrics = {
        'accuracy': 0.75,
        'macro_f1': 0.72,
        'weighted_f1': 0.74,
        'precision_per_class': [0.8, 0.7, 0.65],
        'recall_per_class': [0.75, 0.75, 0.68],
        'f1_per_class': [0.77, 0.72, 0.66],
        'support_per_class': [150, 120, 80],
        'confusion_matrix': [[113, 25, 12], [18, 90, 12], [15, 20, 45]]
    }
    
    visualize_evaluation_results(sample_metrics)
    create_performance_summary(sample_metrics)