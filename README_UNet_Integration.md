# UNet Integration in Mask Editor

## Overview

The mask editor now includes UNet cell division segmentation functionality, allowing you to automatically segment budding cells into mother and daughter cells.

## Features Added

### 1. UNet Model Loading
- **Location**: Right sidebar → "UNet Cell Division Segmentation" section
- **Default Model Path**: `/Volumes/ExFAT/Archive/models/best_iou_model.pth`
- **Functionality**: Automatically loads the default model if it exists, otherwise opens file dialog

### 2. Cell Segmentation Options
- **Segment Selected Cells**: Process only the currently selected cells
- **Segment Budding Cells**: Process only cells classified as budding (requires classification first)
- **Replace Original Cells**: Replace the original cell with segmented mother/daughter cells
- **Add as New Cells**: Keep original cells and add segmented cells as new objects

### 3. UNet Model Architecture
- **Input**: 1-channel grayscale image (256x256)
- **Output**: 3-class segmentation (0=background, 1=daughter, 2=mother)
- **Architecture**: Encoder-decoder U-Net with skip connections

## Usage Instructions

### Step 1: Environment Setup
```bash
# Activate the cell_div conda environment
conda activate cell_div
```

### Step 2: Launch the Application
```bash
python mask_editor.py
```

### Step 3: Load Data
1. Load a mask file (File Management section)
2. Load a classification model for cell classification (required for budding cell detection)
3. Load UNet model (will try default path first)

### Step 4: Classify and Segment Cells
1. **Classify cells** to identify budding cells (required for budding segmentation)
2. **Select cells** you want to segment (optional - for selected cells only)
3. **Choose segmentation mode**:
   - Replace Original Cells: Original cells will be replaced with mother/daughter
   - Add as New Cells: Original cells kept, new cells added
4. **Click segmentation button**:
   - "Segment Selected Cells" for selected cells only
   - "Segment Budding Cells" for all budding cells (enabled only when budding cells are found)

### Step 5: Review Results
- Check the "Segmentation Results" panel for processing summary
- View segmented cells in the main visualization
- Segmented cells will be numbered automatically

## Technical Details

### Model Input Processing
1. Extract individual cell masks
2. Find bounding box and add padding (ensuring image boundary constraints)
3. Resize to 256x256 pixels
4. Normalize to [0,1] range

### Model Output Processing
1. Apply argmax to get class predictions
2. Resize back to exact original cropped size
3. Place segmented result at precise original coordinates
4. Apply mask to ensure segmentation stays within original cell boundaries
5. Assign new cell IDs based on chosen mode

### Coordinate Precision Features
- **Exact coordinate preservation**: Segmented cells maintain their original position
- **Boundary enforcement**: Segmentation is masked to original cell boundaries
- **Edge case handling**: Proper padding for cells near image edges
- **Pixel conservation**: Total pixel count is preserved during segmentation

### Cell ID Assignment
- **Replace Mode**: Original cell removed, new IDs assigned sequentially
- **Add Mode**: Original cell kept, new IDs added after highest existing ID

## Integration with Classification
- UNet segmentation works independently of cell classification
- Both models can be loaded and used simultaneously
- Classification results are preserved during segmentation

## File Output
When saving masks, the following files are generated:
- **Mask file** (`.tif`): Updated mask with segmented cells
- **Outline file** (`.txt`): FISH-QUANT format outline file (if enabled)
- **Classification results** (`.json`): Cell classification data (if available)

## Troubleshooting

### Common Issues
1. **"No UNet model loaded"**: Load the UNet model first
2. **"No cells found"**: Ensure the mask contains labeled cells
3. **"Segmentation failed"**: Check if cells are large enough (>100 pixels)
4. **"Segment Budding Cells" disabled**: First classify cells to identify budding cells

### Fixed Issues
- **✅ Coordinate precision**: Segmented cells now maintain exact original positions
- **✅ Pixel conservation**: No pixel loss or gain during segmentation
- **✅ Boundary preservation**: Segmentation strictly within original cell boundaries

### Performance Tips
- Larger cells segment better than very small cells
- Process in batches for large numbers of cells
- Use "Replace Original Cells" mode to keep cell count manageable

## Model Requirements
- **PyTorch**: Required for UNet functionality
- **Model format**: `.pth` or `.pt` files with state_dict
- **Expected architecture**: UNet with 1 input channel, 3 output classes 