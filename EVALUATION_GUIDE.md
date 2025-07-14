# Model Evaluation Guide

This guide explains how to use the model testing and evaluation system for the Maandamano sentiment analysis project.

## Overview

The evaluation system provides comprehensive testing and assessment of the sentiment analysis model's performance using various metrics including:

- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / (True positives + False positives) per class
- **Recall**: True positives / (True positives + False negatives) per class  
- **F1 Score**: Harmonic mean of precision and recall per class
- **Confusion Matrix**: Detailed breakdown of predictions vs actual labels

## Files

- `model_evaluation.py` - Main evaluation script
- `evaluation_utils.py` - Utility functions for metrics calculation
- `test_basic_evaluation.py` - Basic functionality test without dependencies
- `test_evaluation.py` - Full test suite with numpy/pandas

## Requirements

Before running the evaluation, ensure you have the required dependencies:

```bash
pip install pandas scikit-learn numpy
```

## Usage

### Quick Start

Run the complete evaluation pipeline:

```bash
python model_evaluation.py
```

This will:
1. Load the labeled tweets dataset
2. Analyze the full dataset predictions
3. Split data into training/validation sets (80/20)
4. Evaluate model performance on validation set
5. Generate comprehensive metrics report

### Custom Evaluation

You can also use the evaluation system programmatically:

```python
from model_evaluation import SentimentModelEvaluator

# Initialize evaluator
evaluator = SentimentModelEvaluator(data_path="data/labeled_tweets.csv")

# Load and prepare data
evaluator.load_data()
evaluator.prepare_data(test_size=0.3, random_state=123)

# Run evaluation
metrics = evaluator.evaluate_on_validation()
```

### Testing the System

To test the evaluation system without installing all dependencies:

```bash
python test_basic_evaluation.py
```

## Output

The evaluation generates several types of output:

### 1. Class Distribution Analysis
```
CLASS DISTRIBUTION ANALYSIS
========================================
Total Samples: 200
----------------------------------------
Class        Count    Percentage  
----------------------------------------
Negative     85       42.50%
Neutral      75       37.50%
Positive     40       20.00%
========================================
```

### 2. Performance Metrics Report
```
MODEL EVALUATION REPORT
============================================================

Overall Accuracy: 0.7500

------------------------------------------------------------
Per-Class Metrics:
------------------------------------------------------------
Class        Precision    Recall       F1-Score     Support     
------------------------------------------------------------
Negative     0.8000       0.7500       0.7742       85          
Neutral      0.7200       0.7600       0.7396       75          
Positive     0.6800       0.7000       0.6899       40          
------------------------------------------------------------
Macro Avg    0.7333       0.7367       0.7346       200         
Weighted Avg 0.7520       0.7500       0.7508       200         
```

### 3. Confusion Matrix
```
Confusion Matrix:
------------------------------------------------------------
Predicted ->
Actual    Negative  Neutral   Positive  
Negative  64        15        6         
Neutral   8         57        10        
Positive  6         6         28        
```

### 4. Sample Predictions
```
SAMPLE PREDICTIONS:
------------------------------------------------------------
 1. Negative   | UDA fan solution demo madness Ruto must...
 2. Neutral    | Whatever hold mind consistent basis exact...
 3. Negative   | Uko Kisumu na unafungua duka middle Ich...
 4. Positive   | grateful Residents Pipeline selflessly...
 5. Neutral    | Guys peaceful today demonstrate picket...
```

## Evaluation Methodology

### Current Implementation

Since the dataset contains pre-labeled sentiment predictions from the cardiffnlp/twitter-xlm-roberta-base-sentiment model, the current evaluation:

1. **Extracts predicted labels** from probability distributions
2. **Creates synthetic ground truth** for demonstration (with controlled noise)
3. **Calculates standard metrics** using manual implementations
4. **Provides comprehensive reporting** of model performance

### For Real-World Usage

In a production environment, you should:

1. **Collect human-annotated ground truth labels** for a subset of data
2. **Replace synthetic ground truth** with actual annotations
3. **Implement cross-validation** for more robust evaluation
4. **Add additional metrics** like ROC curves and AUC scores

## Customization

### Changing Evaluation Parameters

```python
# Different train/validation split
evaluator.prepare_data(test_size=0.3, random_state=42)

# Different data path
evaluator = SentimentModelEvaluator(data_path="data/my_tweets.csv")
```

### Adding New Metrics

You can extend `evaluation_utils.py` to add new metrics:

```python
def calculate_macro_averaged_metrics(metrics):
    """Calculate additional macro-averaged metrics."""
    # Your implementation here
    pass
```

## Interpretation Guide

### Accuracy
- **> 0.8**: Excellent performance
- **0.6-0.8**: Good performance  
- **0.4-0.6**: Moderate performance
- **< 0.4**: Poor performance, needs improvement

### F1 Scores
- **Macro F1**: Unweighted average across all classes
- **Weighted F1**: Weighted by class frequency
- Look for balanced F1 scores across classes to avoid bias

### Class Imbalance
- Check if F1 scores vary significantly between classes
- Consider data balancing techniques if needed
- Monitor precision/recall trade-offs per class

## Troubleshooting

### Common Issues

1. **"No module named pandas"**
   ```bash
   pip install pandas scikit-learn numpy
   ```

2. **"Data file not found"**
   - Ensure `data/labeled_tweets.csv` exists
   - Run data collection and preprocessing scripts first

3. **Label parsing errors**
   - Check data format in CSV file
   - Ensure labels column contains valid probability arrays

### Getting Help

If you encounter issues:
1. Run the basic test: `python test_basic_evaluation.py`
2. Check the data format and file paths
3. Ensure all dependencies are installed
4. Review error messages for specific issues

## Next Steps

1. **Collect ground truth annotations** for more accurate evaluation
2. **Implement cross-validation** for robust performance assessment  
3. **Add visualization components** for better result interpretation
4. **Integrate with model training pipeline** for continuous evaluation
5. **Add automated testing** for the evaluation system itself