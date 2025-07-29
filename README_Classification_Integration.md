# Mask Editor with Cell Classification Integration

Version 1.1.0

## Overview

The Mask Editor has been enhanced with integrated cell classification capabilities, allowing users to classify cell mask instances as Normal or Budding cells using a trained PyTorch model.

## New Features

### 1. Three-Panel Layout

When PyTorch is available, the interface uses a three-panel layout:
- **Left Panel**: Original editing tools (selector, eraser, divider, drag) and object management
- **Middle Panel**: Enlarged visualization area for better image viewing
- **Right Panel**: Cell classification controls and results

### 2. Cell Classification

#### Model Loading
- Load trained PyTorch models (.pth, .pt files)
- Supports the LightweightCNN architecture for binary classification
- Model status indicator shows load success/failure

#### Classification Operations
- **Classify Selected Cells**: Classify only currently selected cells
- **Classify All Cells**: Classify all cells in the mask
- Progress bar shows classification progress
- Background processing to avoid UI freezing

#### Manual Classification
- Override model predictions with manual classification
- Select cells and apply Normal/Budding labels manually
- Manual classifications are marked with full confidence

### 3. Visual Enhancements

#### Color-Coded Display
- **Green**: Normal cells (brightness indicates confidence)
- **Red**: Budding cells (brightness indicates confidence)
- **Yellow**: Selected cells (overrides classification colors)
- **Default colors**: Unclassified cells

#### Enhanced Cell Labels
When "Show Cell Numbers" is enabled, labels include:
- Cell ID number
- Classification (N=Normal, B=Budding)
- Confidence score

### 4. Batch Operations

#### Selective Deletion
- **Delete Normal**: Remove all cells classified as Normal
- **Delete Budding**: Remove all cells classified as Budding
- Confirmation dialogs prevent accidental deletion
- Undo functionality available

#### Results Management
- Detailed results display with confidence indicators:
  - 🟢 High confidence (>90%)
  - 🟡 Medium confidence (70-90%)
  - 🔴 Low confidence (<70%)
- Export classification results to JSON
- Auto-save classification results during batch processing

## Usage Workflow

### Single File Mode
1. Load a mask file
2. Load a trained classification model
3. Select cells or classify all cells
4. Review and manually correct classifications if needed
5. Use batch deletion to remove unwanted cell types
6. Save mask and export classification results

### Batch Processing Mode
1. Select input and output folders
2. Load a classification model
3. Process files one by one:
   - Classification results auto-save with each file
   - Manual corrections can be applied per file
   - Navigate between files with previous/next buttons

## File Outputs

### Mask Files
- Original .tif mask files with edited cells
- Maintains 16-bit precision for object labels

### Classification Results
- JSON files with format: `{filename}_classifications.json`
- Contains metadata, confidence scores, and predictions
- Includes both automatic and manual classifications

### Outline Files (Optional)
- FISH-QUANT format .txt files for downstream analysis
- Generated alongside mask files if enabled

## Technical Requirements

### Required Dependencies
- PyTorch (for classification features)
- torchvision
- scikit-image
- numpy
- matplotlib
- PyQt5

### Model Compatibility
- LightweightCNN architecture
- Input size: 256x256 pixels
- Binary output: Normal (class 0) vs Budding (class 1)
- Supports models trained with the provided training pipeline

## Performance Features

- Background threading for classification to maintain UI responsiveness
- Adaptive pooling handles different input sizes
- GPU support when available
- Memory-efficient cell cropping and resizing

## Backward Compatibility

- Full compatibility with existing mask files
- Works without PyTorch (classification features disabled)
- Original two-panel layout when classification unavailable
- All original editing features preserved

## Keyboard Shortcuts

- **Cmd/Ctrl + Z**: Undo last action
- **Cmd/Ctrl + Click**: Multi-select cells
- **Delete/D**: Delete selected cells
- **R**: Auto-renumber cells
- **+/-**: Zoom in/out
- **0**: Reset zoom

## Tips for Best Results

1. **Model Selection**: Use models trained on similar cell types and imaging conditions
2. **Manual Review**: Always review low-confidence predictions manually
3. **Batch Processing**: For consistent results, classify all cells before manual corrections
4. **Quality Control**: Use the visualization to verify classification accuracy
5. **Data Backup**: Original masks are preserved; use undo for mistake recovery 