# Manual Cell Classification Tool

This tool provides an interactive GUI for manually reviewing and correcting cell classification labels. Instead of just evaluating model performance, it allows you to directly fix labeling errors in your training dataset.

## Problem Statement

When working with cell classification datasets, you might encounter:
- Ambiguous cell images that are difficult to classify automatically
- Inconsistent labeling in the original dataset
- Cases where the model's prediction is actually more accurate than the original label
- Need to improve dataset quality for better model training

This tool helps you systematically review disputed cases and correct the training data.

## Features

- 🖼️ **Interactive GUI**: View cell images in large, clear format (400x400 pixels)
- 🔵🟠 **Direct Classification**: Simply choose "Normal" or "Budding" for each cell
- 🔄 **Auto-advance**: Automatically moves to next cell after each selection
- 💾 **Live Data Updates**: Automatically updates training data label files
- 📊 **Performance Tracking**: Shows corrected metrics and improvement statistics
- 📁 **Backup & Logging**: Saves all corrections with timestamps

## Quick Start

### Option 1: Using the helper script (Recommended)

```bash
python run_manual_verification.py
```

This will use default paths and start the verification process.

### Option 2: Using the main script directly

```bash
python visualize_test_results.py --manual_verify --data_dir classification_data_final_fixed --model_path classification_models/best_f1_model.pth --output_dir verification_results
```

## How It Works

1. **Load Model & Data**: The tool loads your trained model and test dataset
2. **Find Disputed Cases**: Identifies all predictions that don't match original labels
3. **Show GUI**: Opens an interactive window for each disputed case
4. **Direct Classification**: You look at the cell and choose what type it actually is
5. **Update Training Data**: Automatically updates the label files in your dataset
6. **Recalculate Metrics**: Shows improved performance metrics

## GUI Interface

The classification GUI shows:

- **Cell Image**: Large, clear view of the cell (400x400 pixels)
- **Information Panel**: Original label, model prediction, and confidence score
- **Classification Question**: "What type of cell is this?"
- **Choice Buttons**: 
  - 🔵 **NORMAL CELL**: Select if this is a normal cell
  - 🟠 **BUDDING CELL**: Select if this is a budding cell
- **Auto-advance**: Automatically moves to next cell after selection
- **Navigation**: Previous button to review previous choices if needed

## Command Line Options

```bash
python visualize_test_results.py --help
```

Key options:
- `--manual_verify`: Enable manual verification (required)
- `--data_dir`: Directory with your classification data
- `--model_path`: Path to your trained model (.pth file)
- `--output_dir`: Where to save results
- `--num_samples`: Number of samples for visualization plots

## Output Files

After verification, you'll get:

### 📄 `performance_report.json`
Complete performance metrics including:
- Accuracy, precision, recall, F1 scores
- Confusion matrix
- Original vs. corrected performance comparison
- Summary of manual corrections

### 📄 `manual_corrections.json`
Details of all corrections made:
- Which samples were corrected
- Original vs. corrected labels
- Correction statistics

### 📊 Visualization Files
- `confusion_matrix.png`: Updated confusion matrix
- `probability_distribution.png`: Model confidence analysis
- `sample_predictions_*.png`: Visual examples of predictions

## Example Workflow

1. **Train your model** (if not already done)
2. **Run the tool**:
   ```bash
   python run_manual_verification.py --data_dir your_data --model_path your_model.pth
   ```
3. **For each disputed case**:
   - Look at the cell image carefully
   - Decide: is this a Normal cell or Budding cell?
   - Click 🔵 NORMAL CELL or 🟠 BUDDING CELL
   - Tool automatically advances to next case
4. **Results**:
   - Training data labels updated automatically
   - Original accuracy: 0.8500
   - Corrected accuracy: 0.9200 (+0.0700 improvement!)
   - 15 labels corrected, 3 confirmed as originally correct

## Tips for Manual Verification

### What to Look For in Cell Images:

**Normal Cells:**
- Single, round cell body
- Uniform shape
- No visible daughter cell formation

**Budding Cells:**
- Visible bud formation (small protrusion)
- Mother cell with attached smaller cell
- Non-uniform shape indicating division

### Best Practices:

1. **Take Your Time**: Look carefully at each image
2. **Be Consistent**: Use the same criteria throughout
3. **When in Doubt**: Consider the model's confidence score
4. **Save Progress**: The tool saves your corrections as you go
5. **Review Summary**: Check the final statistics make sense

## Troubleshooting

### Common Issues:

**"No incorrect predictions found"**
- Your model might already be very accurate
- Check if you're using the right test data split

**GUI doesn't appear**
- Make sure you're in the correct Python environment
- Check that tkinter is installed: `python -c "import tkinter"`

**Image display issues**
- Ensure Pillow/PIL is installed: `pip install pillow`
- Check image data format

**Performance doesn't improve**
- This is normal if your original labels were already accurate
- Manual verification helps confirm model reliability

## Advanced Usage

### Custom Data Splits
If you want to use a specific train/test split:

```python
# In your code, modify the random_split parameters
train_size = 800  # Your specific size
test_size = 200   # Your specific size
```

### Batch Processing
For large datasets, consider processing in batches:

1. Run verification on a subset first
2. Analyze patterns in corrections
3. Apply systematic corrections if patterns emerge

## Integration with Training Pipeline

After manual verification, you can:

1. **Update your dataset** with corrected labels
2. **Retrain the model** if significant corrections were made
3. **Use corrected metrics** for model evaluation and comparison
4. **Document the verification process** for reproducibility

## Example Results

```
📊 Manual Labeling Results:
  🔄 Labels corrected: 15  
  ✅ Labels confirmed: 3   
  ⏭️ Not labeled: 0

💾 Training data updated: 15 files relabeled
📈 Accuracy improvement: 0.8500 → 0.9200 (+0.0700)
```

This shows that 15 out of 18 disputed labels were corrected, and the training dataset files were automatically updated with the new labels.

## Conclusion

Manual verification is a powerful way to:
- Get more accurate performance metrics
- Identify systematic labeling issues
- Build confidence in your model's performance
- Improve the quality of your training data

The few minutes spent on manual verification can provide much more reliable assessment of your model's true capabilities. 