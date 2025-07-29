# Two-Stage Cell Classification System

A complete machine learning pipeline for binary cell classification and segmentation that distinguishes between **normal cells** and **budding cells**, followed by mother/daughter segmentation for budding cells.

## 🎯 System Overview

This system implements a two-stage approach:

1. **Stage 1: Binary Classification** - Determines if a cell is "Normal" or "Budding"
2. **Stage 2: Segmentation** - For budding cells, segments them into mother and daughter cells

### Architecture

```
Input Cell Mask → [Classification Model] → Normal Cell ✓
                                        ↓
                                     Budding Cell → [Segmentation Model] → Mother/Daughter Masks
```

## 📁 Project Structure

```
├── cell_classification_processor.py    # Data processing for classification
├── cell_classification_model.py        # Lightweight CNN for binary classification
├── two_stage_inference.py             # Complete inference pipeline
├── run_classification_pipeline.py     # Automated pipeline runner
├── cell_division_unet.py              # Original segmentation model
├── data_processor.py                  # Original data processor
├── train_unet.py                      # Original U-Net training
├── data/
│   ├── dic_masks/                      # All cell masks (normal + budding)
│   ├── divided_masks/                  # Only budding cell masks
│   └── divided_outlines/               # Cell annotations (.txt files)
├── classification_data/               # Generated classification training data
├── classification_models/             # Trained classification models
├── models/                           # Original segmentation models
└── inference_results/                # Inference outputs
```

## 🚀 Quick Start

### Option 1: Automated Pipeline (Recommended)

Run the complete pipeline with a single command:

```bash
python run_classification_pipeline.py
```

This will automatically:
1. Process your data for classification training
2. Train the binary classification model
3. Run inference demonstration

### Option 2: Manual Step-by-Step

#### Step 1: Process Data for Classification

```bash
python cell_classification_processor.py --data_root data --output_dir classification_data
```

This extracts individual cells and creates binary labels (0=Normal, 1=Budding).

#### Step 2: Train Classification Model

```bash
python cell_classification_model.py
```

Trains a lightweight CNN for binary classification.

#### Step 3: Run Two-Stage Inference

```bash
python two_stage_inference.py path/to/your/cell_mask.tif --visualize
```

## 📊 Data Requirements

### Input Data Structure

Your data should be organized as follows:

```
data/
├── dic_masks/              # All cell masks (contains both normal and budding cells)
│   ├── image1.tif
│   ├── image2.tif
│   └── ...
├── divided_masks/          # Only budding cell masks (for training segmentation)
│   ├── image1.tif
│   ├── image2.tif
│   └── ...
└── divided_outlines/       # Cell annotations
    ├── image1.txt
    ├── image2.txt
    └── ...
```

### Cell ID Naming Convention

The system uses the following cell ID convention:
- **Mother cells**: ID ending with `2` (e.g., 102, 202, 302)
- **Daughter cells**: ID ending with `1` (e.g., 101, 201, 301)
- **Normal cells**: All other IDs (e.g., 100, 103, 105)

## 🧠 Model Architectures

### Classification Model (LightweightCNN)

- **Input**: 128×128 binary cell masks
- **Output**: Binary classification (Normal=0, Budding=1)
- **Architecture**: 4 convolutional blocks + 3 fully connected layers
- **Parameters**: ~500K (lightweight and fast)

### Segmentation Model (U-Net)

- **Input**: 256×256 binary cell masks
- **Output**: 3-class segmentation (Background=0, Daughter=1, Mother=2)
- **Architecture**: Standard U-Net with skip connections
- **Parameters**: ~31M (detailed segmentation)

## 📈 Usage Examples

### Basic Classification

```python
from two_stage_inference import TwoStageInferencePipeline

# Initialize pipeline
pipeline = TwoStageInferencePipeline()

# Process a single cell
cell_type, result_mask, info = pipeline.process_cell(cell_mask)
print(f"Cell type: {cell_type}")
print(f"Confidence: {info['classification_confidence']:.3f}")
```

### Batch Processing

```python
# Process multiple cells in an image
results = pipeline.process_image_with_multiple_cells(labeled_mask)

for cell_id, result in results.items():
    print(f"Cell {cell_id}: {result['cell_type']}")
```

### Command Line Usage

```bash
# Single cell processing
python two_stage_inference.py cell_mask.tif --visualize

# Batch processing with custom threshold
python two_stage_inference.py multi_cell_mask.tif --threshold 0.7 --output_dir results/

# Use custom models
python two_stage_inference.py cell_mask.tif \
    --classification_model my_classifier.pth \
    --segmentation_model my_segmenter.pth
```

## ⚙️ Configuration Options

### Classification Threshold

Adjust the confidence threshold for classification decisions:

```python
# Conservative (fewer false positives)
cell_type, result_mask, info = pipeline.process_cell(cell_mask, classification_threshold=0.8)

# Liberal (more sensitive to budding cells)
cell_type, result_mask, info = pipeline.process_cell(cell_mask, classification_threshold=0.3)
```

### Model Parameters

Customize model architectures:

```python
# Smaller, faster classification model
model = LightweightCNN(input_size=64, num_classes=2, dropout=0.5)

# Larger, more accurate classification model
model = LightweightCNN(input_size=256, num_classes=2, dropout=0.2)
```

## 📊 Performance Metrics

The system tracks several key metrics:

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Normal/Budding precision scores
- **Recall**: Normal/Budding recall scores
- **F1 Score**: Balanced performance measure

### Segmentation Metrics (for budding cells)
- **IoU (Intersection over Union)**: Overlap accuracy for mother/daughter regions
- **Pixel Accuracy**: Per-pixel classification accuracy

## 🔧 Advanced Usage

### Custom Data Processing

```python
from cell_classification_processor import CellClassificationProcessor

# Custom data processor
processor = CellClassificationProcessor(
    data_root="custom_data",
    output_dir="custom_output"
)

# Process with custom target size
processor.target_size = (64, 64)  # Faster processing
num_samples = processor.run_full_pipeline()
```

### Training with Custom Parameters

```python
from cell_classification_model import main as train_main

# Modify training parameters in the script:
# - num_epochs: Number of training epochs
# - batch_size: Training batch size  
# - learning_rate: Learning rate
# - dropout: Dropout rate for regularization
```

## 🐛 Troubleshooting

### Common Issues

1. **"No classification data found"**
   - Run the data processor first: `python cell_classification_processor.py`

2. **"Model not found"**
   - Train the classification model: `python cell_classification_model.py`

3. **Low classification accuracy**
   - Check data quality and annotations
   - Increase training epochs
   - Adjust class weights for imbalanced data

4. **Memory errors during training**
   - Reduce batch size
   - Use smaller input image size
   - Enable gradient checkpointing

### Performance Optimization

```python
# For faster inference (less accurate)
pipeline = TwoStageInferencePipeline(device='cpu')  # Use CPU
processor.target_size = (64, 64)  # Smaller images

# For better accuracy (slower)
pipeline = TwoStageInferencePipeline(device='cuda')  # Use GPU
processor.target_size = (256, 256)  # Larger images
```

## 📚 Dependencies

```python
# Core dependencies
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
tifffile>=2021.7.2
tqdm>=4.62.0
pandas>=1.3.0
```

Install all dependencies:

```bash
pip install torch torchvision numpy opencv-python scikit-learn matplotlib seaborn tifffile tqdm pandas
```

## 🎨 Visualization Features

The system includes comprehensive visualization tools:

### Training Curves
- Loss curves (training/validation)
- Accuracy progression
- F1 score evolution
- Learning rate scheduling

### Inference Results
- Original cell mask
- Classification result
- Segmentation overlay (for budding cells)
- Confidence scores

### Confusion Matrix
- Classification performance breakdown
- Per-class accuracy analysis

## 🔬 Research Applications

This system is particularly useful for:

1. **Cell Biology Research**: Automated analysis of cell division cycles
2. **Drug Discovery**: Screening compounds that affect cell division
3. **Medical Diagnostics**: Identifying abnormal cell behavior
4. **Time-lapse Analysis**: Tracking cell division over time

## 📝 Citation

If you use this system in your research, please consider citing:

```
@software{two_stage_cell_classification,
  title={Two-Stage Cell Classification System},
  author={Your Name},
  year={2024},
  description={Binary classification and segmentation pipeline for cell analysis}
}
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Model Architectures**: More efficient or accurate models
2. **Data Augmentation**: Advanced augmentation techniques
3. **Visualization**: Enhanced plotting and analysis tools
4. **Performance**: Optimization for large-scale processing

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

---

## 🎉 Success Stories

After running the complete pipeline, you should see:

```
✅ All steps completed successfully!

🎉 Your two-stage cell classification system is ready!

Next steps:
1. Check the results in 'inference_results/' directory
2. Use 'two_stage_inference.py' to process new cell images  
3. Adjust classification threshold if needed
```

Your system can now automatically distinguish between normal and budding cells, and segment budding cells into mother/daughter components with high accuracy! 