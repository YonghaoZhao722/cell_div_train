import sys
import os
import glob
import numpy as np
from scipy import ndimage
from skimage import measure, segmentation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Polygon
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QPushButton, 
                            QFileDialog, QSlider, QGroupBox, QGridLayout, 
                            QSpinBox, QCheckBox, QLineEdit, QMessageBox, 
                            QRadioButton, QButtonGroup, QListWidget, QListWidgetItem,
                            QTabWidget, QStackedWidget, QProgressBar, QTextEdit)
from PyQt5.QtCore import Qt, QPoint, QSize, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap
# Try to import SVG support
try:
    from PyQt5.QtSvg import QSvgRenderer
    SVG_SUPPORT = True
except ImportError:
    SVG_SUPPORT = False
    print("Warning: SVG support not available")

import cv2
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
import tifffile as tiff
from datetime import datetime

# Classification model imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    TORCH_AVAILABLE = True
    print("PyTorch available - Classification features enabled")
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available - Classification features disabled")

# UNet model for segmentation
class UNet(nn.Module):
    """U-Net model for cell division segmentation"""
    
    def __init__(self, in_channels=1, num_classes=3):
        super(UNet, self).__init__()
        
        # Encoder (downsampling path)
        self.enc1 = self._double_conv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._double_conv(512, 1024)
        
        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._double_conv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._double_conv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._double_conv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._double_conv(128, 64)
        
        # Final output layer
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def _double_conv(self, in_channels, out_channels):
        """Double convolution block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        
        enc4 = self.enc4(pool3)
        pool4 = self.pool4(enc4)
        
        # Bottleneck
        bottleneck = self.bottleneck(pool4)
        
        # Decoder path
        upconv4 = self.upconv4(bottleneck)
        concat4 = torch.cat([upconv4, enc4], dim=1)
        dec4 = self.dec4(concat4)
        
        upconv3 = self.upconv3(dec4)
        concat3 = torch.cat([upconv3, enc3], dim=1)
        dec3 = self.dec3(concat3)
        
        upconv2 = self.upconv2(dec3)
        concat2 = torch.cat([upconv2, enc2], dim=1)
        dec2 = self.dec2(concat2)
        
        upconv1 = self.upconv1(dec2)
        concat1 = torch.cat([upconv1, enc1], dim=1)
        dec1 = self.dec1(concat1)
        
        # Output
        out = self.out_conv(dec1)
        
        return out

# Version identifier
VERSION = "1.1.0"

def generate_fishquant_outline(mask_img, cy3_filename, output_file):
    """
    Generate FISH-QUANT format outline file from mask image.
    Similar to Mask2Outline.py functionality but integrated into mask editor.
    """
    try:
        # Get unique cell labels (excluding background 0)
        cell_labels = np.unique(mask_img)
        cell_labels = cell_labels[cell_labels != 0]
        
        if len(cell_labels) == 0:
            print("No cells found in mask")
            return False
        
        # Find contours for each cell and get their top-most point for sorting
        cell_contours = []
        for label in cell_labels:
            binary_mask = (mask_img == label).astype(np.uint8)
            contours = measure.find_contours(binary_mask, level=0.5)
            if contours:
                contour = max(contours, key=len)
                min_y = np.min(contour[:, 0])  # Top-most Y coordinate  
                cell_contours.append((label, contour, min_y))
        
        # Sort by Y coordinate (top to bottom) to match Mask2Outline.py behavior
        cell_contours = sorted(cell_contours, key=lambda x: x[2])
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        # Write FISH-QUANT format outline file
        with open(output_file, 'w') as fid:
            fid.write('FISH-QUANT\tv2a\n')
            from datetime import datetime
            current_date = datetime.now().strftime("%d-%b-%Y")
            fid.write(f'RESULTS OF SPOT DETECTION PERFORMED ON {current_date}\n')
            fid.write('COMMENT\t\n')
            fid.write(f'IMG_Raw\t{cy3_filename}\n')
            fid.write('IMG_Filtered\t\n')
            fid.write('IMG_DAPI\t\n')
            fid.write('IMG_TS_label\t\n')
            fid.write('FILE_settings\t\n')
            fid.write('PARAMETERS\n')
            fid.write('Pix-XY\tPix-Z\tRI\tEx\tEm\tNA\tType\n')
            fid.write('160\t300\t1.518\t583\t547\t1.4\twidefield\n')
            
            for i, (label, contour, min_y) in enumerate(cell_contours, 1):
                y_coords = contour[:, 0] 
                x_coords = contour[:, 1]
                
                # Use the original cell label number instead of renumbering
                cell_name = f'Cell_{label}'
                fid.write(f'CELL\t{cell_name}\n')
                fid.write('X_POS')
                for x in x_coords:
                    fid.write(f'\t{int(x)}')
                fid.write('\tEND\n')
                fid.write('Y_POS')
                for y in y_coords:
                    fid.write(f'\t{int(y)}')
                fid.write('\tEND\n')
                fid.write('Z_POS\t\n')
        
        print(f"Generated outline file: {output_file} with {len(cell_contours)} cells")
        return True
        
    except Exception as e:
        print(f"Error generating outline file: {e}")
        return False

class LightweightCNN(nn.Module):
    """
    Lightweight CNN for binary cell classification (Normal vs Budding)
    """
    
    def __init__(self, input_size=128, num_classes=2, dropout=0.3):
        super(LightweightCNN, self).__init__()
        
        self.input_size = input_size
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: 128 -> 64
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout * 0.5),
            
            # Block 2: 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout * 0.7),
            
            # Block 3: 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout),
            
            # Block 4: 16 -> 8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout),
        )
        
        # Add adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Fixed feature size after adaptive pooling: 256 channels * 8 * 8 = 16384
        self.feature_size = 256 * 8 * 8
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        
        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)
        
        # Flatten for classification
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x

class ClassificationWorker(QThread):
    """Background worker for cell classification"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, mask_img, model, device, cell_labels=None):
        super().__init__()
        self.mask_img = mask_img
        self.model = model
        self.device = device
        self.cell_labels = cell_labels
        
    def run(self):
        try:
            if self.cell_labels is None:
                # Get all unique cell labels
                unique_labels = np.unique(self.mask_img)
                self.cell_labels = unique_labels[unique_labels != 0]
            
            results = {}
            total_cells = len(self.cell_labels)
            
            for i, cell_id in enumerate(self.cell_labels):
                # Extract cell mask
                cell_mask = (self.mask_img == cell_id).astype(np.float32)
                
                # Get bounding box to crop the cell
                y_coords, x_coords = np.where(cell_mask > 0)
                if len(y_coords) == 0:
                    continue
                    
                y_min, y_max = y_coords.min(), y_coords.max()
                x_min, x_max = x_coords.min(), x_coords.max()
                
                # Add padding
                padding = 10
                y_min = max(0, y_min - padding)
                y_max = min(cell_mask.shape[0], y_max + padding + 1)
                x_min = max(0, x_min - padding)
                x_max = min(cell_mask.shape[1], x_max + padding + 1)
                
                # Crop the cell
                cropped_cell = cell_mask[y_min:y_max, x_min:x_max]
                
                # Resize to model input size (256x256)
                from skimage.transform import resize
                resized_cell = resize(cropped_cell, (256, 256), preserve_range=True)
                
                # Convert to tensor and add batch dimension
                cell_tensor = torch.from_numpy(resized_cell).float()
                cell_tensor = cell_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                cell_tensor = cell_tensor.to(self.device)
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(cell_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities.max().item()
                
                # Store results
                results[int(cell_id)] = {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'class_name': 'Normal' if predicted_class == 0 else 'Budding',
                    'probabilities': probabilities.cpu().numpy().tolist()[0]
                }
                
                # Emit progress
                progress_percent = int((i + 1) / total_cells * 100)
                self.progress.emit(progress_percent)
            
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))

class MaskEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mask_img = None
        self.original_mask = None
        self.mask_file_path = None
        self.current_tool = "selector"
        self.selected_object_ids = set()  # Changed from single ID to set for multi-selection
        self.brush_size = 5
        self.is_drawing = False
        self.last_pos = None
        self.divide_points = []
        self.cell_objects = {}  # Dictionary to store cell objects and their properties
        
        # Undo functionality
        self.mask_history = []
        self.max_history = 50  # Maximum undo steps
        
        # Folder processing
        self.input_folder = None
        self.output_folder = None
        self.file_list = []
        self.current_file_index = 0
        
        # Zoom functionality
        self.zoom_level = 1.0
        self.zoom_center_x = 0.5  # Center as fraction of image width
        self.zoom_center_y = 0.5  # Center as fraction of image height
        
        # Drag functionality
        self.is_dragging = False
        self.drag_start_pos = None
        
        # Color map for different objects
        self.object_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        # Classification functionality
        self.classification_model = None
        self.device = None
        self.model_path = None
        self.cell_classifications = {}  # Store classification results
        self.classification_worker = None
        
        # UNet segmentation functionality
        self.unet_model = None
        self.unet_model_path = None
        self.target_size = (256, 256)  # UNet input size
        
        # Initialize PyTorch device if available
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Classification device: {self.device}")
        
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.initUI()
    
    def initUI(self):
        title = 'Mask Editor with Cell Classification & UNet Segmentation' if TORCH_AVAILABLE else 'Mask Editor'
        self.setWindowTitle(title)
        
        # Set larger window size for three-panel layout when classification is available
        if TORCH_AVAILABLE:
            self.setGeometry(50, 50, 1600, 900)  # Wider for three panels
            self.setMinimumSize(1400, 750)  # Larger minimum for three panels
        else:
            self.setGeometry(100, 100, 1200, 800)  # Original size for two panels
            self.setMinimumSize(800, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(2, 2, 2, 2)  # Minimal margins for main layout
        main_layout.setSpacing(2)  # Minimal spacing between panels
        
        # Left sidebar for tools and controls
        left_sidebar = QWidget()
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setSpacing(8)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)
        
        # File Management group with mode selection
        self.file_group = QGroupBox("File Management")
        file_layout = QVBoxLayout()
        file_layout.setSpacing(3)  # Reduce spacing
        file_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        
        # Mode selection - more compact
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(10)
        self.single_mode_radio = QRadioButton("Single")
        self.batch_mode_radio = QRadioButton("Batch")
        self.single_mode_radio.setChecked(True)
        self.single_mode_radio.toggled.connect(self.toggle_file_mode)
        self.batch_mode_radio.toggled.connect(self.toggle_file_mode)
        
        mode_layout.addWidget(self.single_mode_radio)
        mode_layout.addWidget(self.batch_mode_radio)
        mode_layout.addStretch()
        file_layout.addLayout(mode_layout)
        
        # Stacked widget for different modes
        self.file_stack = QStackedWidget()
        
        # Single file mode widget
        single_widget = QWidget()
        single_widget.setMaximumHeight(80)  # Compact for single mode
        single_layout = QVBoxLayout()
        single_layout.setSpacing(2)
        single_layout.setContentsMargins(0, 0, 0, 0)
        
        # Load/Save buttons in one row
        buttons_layout = QHBoxLayout()
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self.load_mask)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_mask)
        buttons_layout.addWidget(load_btn)
        buttons_layout.addWidget(save_btn)
        single_layout.addLayout(buttons_layout)
        
        self.mask_file_label = QLabel("No mask loaded")
        self.mask_file_label.setWordWrap(True)
        self.mask_file_label.setStyleSheet("font-size: 11px; color: gray;")
        single_layout.addWidget(self.mask_file_label)
        
        single_widget.setLayout(single_layout)
        self.file_stack.addWidget(single_widget)
        
        # Batch processing mode widget
        batch_widget = QWidget()
        batch_widget.setMaximumHeight(160)  # More space for batch controls
        batch_layout = QVBoxLayout()
        batch_layout.setSpacing(2)
        batch_layout.setContentsMargins(0, 0, 0, 0)
        
        # Input folder
        input_folder_layout = QHBoxLayout()
        input_folder_layout.addWidget(QLabel("In:"))
        input_folder_btn = QPushButton("Browse")
        input_folder_btn.clicked.connect(self.select_input_folder)
        input_folder_layout.addWidget(input_folder_btn)
        batch_layout.addLayout(input_folder_layout)
        
        self.input_folder_label = QLabel("No folder selected")
        self.input_folder_label.setWordWrap(True)
        self.input_folder_label.setStyleSheet("font-size: 10px; color: gray;")
        batch_layout.addWidget(self.input_folder_label)
        
        # Output folder
        output_folder_layout = QHBoxLayout()
        output_folder_layout.addWidget(QLabel("Out:"))
        output_folder_btn = QPushButton("Browse")
        output_folder_btn.clicked.connect(self.select_output_folder)
        output_folder_layout.addWidget(output_folder_btn)
        batch_layout.addLayout(output_folder_layout)
        
        self.output_folder_label = QLabel("No folder selected")
        self.output_folder_label.setWordWrap(True)
        self.output_folder_label.setStyleSheet("font-size: 10px; color: gray;")
        batch_layout.addWidget(self.output_folder_label)
        
        # File navigation
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀")
        self.prev_btn.clicked.connect(self.previous_file)
        self.prev_btn.setEnabled(False)
        self.prev_btn.setFixedWidth(30)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("▶")
        self.next_btn.clicked.connect(self.next_file)
        self.next_btn.setEnabled(False)
        self.next_btn.setFixedWidth(30)
        nav_layout.addWidget(self.next_btn)
        
        batch_layout.addLayout(nav_layout)
        
        self.file_progress_label = QLabel("No files loaded")
        self.file_progress_label.setStyleSheet("font-size: 10px; color: gray;")
        batch_layout.addWidget(self.file_progress_label)
        
        batch_widget.setLayout(batch_layout)
        self.file_stack.addWidget(batch_widget)
        
        file_layout.addWidget(self.file_stack)
        
        # Set adaptive height for the file management group
        self.file_group.setMinimumHeight(120)
        self.file_group.setMaximumHeight(140)  # Start with single mode height
        
        # Undo button with icon (universal for both modes)
        undo_layout = QHBoxLayout()
        undo_layout.setContentsMargins(0, 2, 0, 0)
        self.undo_btn = QPushButton()
        self.create_undo_icon()
        self.undo_btn.setToolTip("Undo (Cmd/Ctrl + Z)")
        self.undo_btn.clicked.connect(self.undo_action)
        undo_layout.addWidget(self.undo_btn)
        undo_layout.addStretch()
        file_layout.addLayout(undo_layout)
        
        self.file_group.setLayout(file_layout)
        sidebar_layout.addWidget(self.file_group)
        
        # Add vertical spacing
        sidebar_layout.addSpacing(10)
        
        # Tools group
        tools_group = QGroupBox("Editing Tools")
        tools_layout = QVBoxLayout()
        tools_layout.setSpacing(5)
        tools_layout.setContentsMargins(5, 5, 5, 5)
        
        # Tool selection buttons - horizontal layout with icons
        tools_button_layout = QHBoxLayout()
        tools_button_layout.setSpacing(3)
        self.tool_group = QButtonGroup()
        
        # Create tool buttons with icons
        self.selector_btn = QPushButton()
        self.eraser_btn = QPushButton()
        self.divider_btn = QPushButton()
        self.drag_btn = QPushButton()
        
        # Set up tool buttons
        self.setup_tool_buttons()
        
        # Add buttons to layout and group
        tools_button_layout.addWidget(self.selector_btn)
        tools_button_layout.addWidget(self.eraser_btn)
        tools_button_layout.addWidget(self.divider_btn)
        tools_button_layout.addWidget(self.drag_btn)
        
        self.tool_group.addButton(self.selector_btn, 0)
        self.tool_group.addButton(self.eraser_btn, 1)
        self.tool_group.addButton(self.divider_btn, 2)
        self.tool_group.addButton(self.drag_btn, 3)
        
        self.tool_group.buttonClicked.connect(self.on_tool_button_clicked)
        self.selector_btn.setChecked(True)  # Default selection
        
        tools_layout.addLayout(tools_button_layout)
        
        # Brush size control - more compact
        brush_layout = QHBoxLayout()
        brush_layout.setSpacing(5)
        brush_layout.addWidget(QLabel("Size:"))
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(20)
        self.brush_slider.setValue(5)
        self.brush_slider.valueChanged.connect(self.update_brush_size)
        brush_layout.addWidget(self.brush_slider)
        
        self.brush_size_label = QLabel("5")
        self.brush_size_label.setFixedWidth(20)
        brush_layout.addWidget(self.brush_size_label)
        tools_layout.addLayout(brush_layout)
        
        tools_group.setLayout(tools_layout)
        sidebar_layout.addWidget(tools_group)
        
        # Add vertical spacing
        sidebar_layout.addSpacing(8)
        
        # Object management group
        object_group = QGroupBox("Object Management")
        object_layout = QVBoxLayout()
        object_layout.setSpacing(5)
        object_layout.setContentsMargins(5, 5, 5, 5)
        
        # Selected object display
        self.selected_label = QLabel("Selected: None")
        self.selected_label.setToolTip("Hold Cmd/Ctrl and click cells to select multiple for deletion")
        object_layout.addWidget(self.selected_label)
        
        # Cell numbering controls
        renumber_layout = QHBoxLayout()
        renumber_btn = QPushButton("Auto Renumber Cells")
        renumber_btn.clicked.connect(self.auto_renumber_cells)
        renumber_layout.addWidget(renumber_btn)
        object_layout.addLayout(renumber_layout)
        
        # Manual cell number editor
        editor_layout = QHBoxLayout()
        editor_layout.addWidget(QLabel("Edit Cell #:"))
        self.cell_number_input = QLineEdit()
        self.cell_number_input.setPlaceholderText("Enter number")
        self.cell_number_input.returnPressed.connect(self.apply_cell_number_edit)
        editor_layout.addWidget(self.cell_number_input)
        
        apply_number_btn = QPushButton("Apply")
        apply_number_btn.clicked.connect(self.apply_cell_number_edit)
        editor_layout.addWidget(apply_number_btn)
        object_layout.addLayout(editor_layout)
        
        # Object list - responsive height
        self.object_list = QListWidget()
        self.object_list.itemClicked.connect(self.select_object_from_list)
        self.object_list.setMinimumHeight(80)
        self.object_list.setMaximumHeight(200)
        self.object_list.setSelectionMode(QListWidget.ExtendedSelection)  # Allow multi-selection
        self.object_list.setToolTip("Click to select cells. Hold Cmd/Ctrl for multi-selection.")
        object_layout.addWidget(self.object_list)
        
        # Delete selected objects
        delete_btn = QPushButton("Delete Selected Objects")
        delete_btn.clicked.connect(self.delete_selected_object)
        delete_btn.setToolTip("Delete all currently selected cells. Use Cmd/Ctrl+click to select multiple cells.")
        object_layout.addWidget(delete_btn)
        
        object_group.setLayout(object_layout)
        sidebar_layout.addWidget(object_group)
        
        # Add vertical spacing
        sidebar_layout.addSpacing(8)
        
        # View controls group
        view_group = QGroupBox("View Controls")
        view_layout = QVBoxLayout()
        view_layout.setSpacing(3)
        view_layout.setContentsMargins(5, 5, 5, 5)
        
        # Show object borders
        self.show_borders_cb = QCheckBox("Show Object Borders")
        self.show_borders_cb.setChecked(True)
        self.show_borders_cb.toggled.connect(self.update_display)
        view_layout.addWidget(self.show_borders_cb)
        
        # Show cell numbers
        self.show_numbers_cb = QCheckBox("Show Cell Numbers")
        self.show_numbers_cb.setChecked(True)
        self.show_numbers_cb.toggled.connect(self.update_display)
        view_layout.addWidget(self.show_numbers_cb)
        
        # Auto renumber option
        self.auto_renumber_cb = QCheckBox("Auto Renumber After Division")
        self.auto_renumber_cb.setChecked(True)
        view_layout.addWidget(self.auto_renumber_cb)
        
        # Generate outline files option
        self.generate_outlines_cb = QCheckBox("Generate Outline Files (.txt)")
        self.generate_outlines_cb.setChecked(True)
        self.generate_outlines_cb.setToolTip("Generate FISH-QUANT format outline files when saving masks")
        view_layout.addWidget(self.generate_outlines_cb)
        
        # Compact labels option
        self.compact_labels_cb = QCheckBox("Compact Labels")
        self.compact_labels_cb.setChecked(False)
        self.compact_labels_cb.setToolTip("Use smaller, more compact cell labels (automatic for >20 cells)")
        self.compact_labels_cb.toggled.connect(self.update_display)
        view_layout.addWidget(self.compact_labels_cb)
        
        # Add spacing
        view_layout.addSpacing(8)
        
        # Zoom control
        zoom_layout = QHBoxLayout()
        zoom_layout.setSpacing(5)
        zoom_layout.addWidget(QLabel("Zoom:"))
        
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(25)  # 0.25x zoom
        self.zoom_slider.setMaximum(500)  # 5.0x zoom
        self.zoom_slider.setValue(100)  # 1.0x zoom (100%)
        self.zoom_slider.valueChanged.connect(self.update_zoom)
        zoom_layout.addWidget(self.zoom_slider)
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(40)
        zoom_layout.addWidget(self.zoom_label)
        view_layout.addLayout(zoom_layout)
        
        # Zoom reset button
        zoom_reset_btn = QPushButton("Reset Zoom")
        zoom_reset_btn.clicked.connect(self.reset_zoom)
        view_layout.addWidget(zoom_reset_btn)
        
        view_group.setLayout(view_layout)
        sidebar_layout.addWidget(view_group)
        
        # Add vertical spacing
        sidebar_layout.addSpacing(8)
        
        # Responsive stretch - takes remaining space
        sidebar_layout.addStretch(1)
        
        left_sidebar.setLayout(sidebar_layout)
        left_sidebar.setMinimumWidth(250)
        left_sidebar.setMaximumWidth(280)
        main_layout.addWidget(left_sidebar)
        
        # Middle area for visualization
        viz_widget = QWidget()
        viz_layout = QVBoxLayout()
        viz_layout.setContentsMargins(2, 2, 2, 2)  # Minimal margins
        viz_layout.setSpacing(2)  # Minimal spacing
        
        # Status bar
        self.status_label = QLabel("Single file mode - Load a mask to begin editing")
        self.status_label.setMaximumHeight(25)  # Compact status bar
        viz_layout.addWidget(self.status_label)
        
        # Visualization area with minimal margins
        self.figure = plt.figure(figsize=(12, 10))
        self.figure.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)  # Minimal margins
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()
        viz_layout.addWidget(self.canvas)
        
        viz_widget.setLayout(viz_layout)
        main_layout.addWidget(viz_widget, stretch=3)  # Give more space to visualization
        
        # Right sidebar for classification (only show if PyTorch is available)
        if TORCH_AVAILABLE:
            right_sidebar = QWidget()
            right_sidebar_layout = QVBoxLayout()
            right_sidebar_layout.setSpacing(8)
            right_sidebar_layout.setContentsMargins(5, 5, 5, 5)
            
            # Classification group
            classification_group = QGroupBox("Cell Classification")
            classification_layout = QVBoxLayout()
            classification_layout.setSpacing(5)
            classification_layout.setContentsMargins(5, 5, 5, 5)
            
            # Model loading section
            model_section = QGroupBox("Model")
            model_section_layout = QVBoxLayout()
            model_section_layout.setSpacing(3)
            
            # Model loading button
            self.load_model_btn = QPushButton("Load Model")
            self.load_model_btn.clicked.connect(self.load_classification_model)
            model_section_layout.addWidget(self.load_model_btn)
            
            # Model status
            self.model_status_label = QLabel("No model loaded")
            self.model_status_label.setWordWrap(True)
            self.model_status_label.setStyleSheet("font-size: 11px; color: gray;")
            model_section_layout.addWidget(self.model_status_label)
            
            model_section.setLayout(model_section_layout)
            classification_layout.addWidget(model_section)
            
            # Classification controls section
            classify_section = QGroupBox("Classification")
            classify_section_layout = QVBoxLayout()
            classify_section_layout.setSpacing(3)
            
            # Classification buttons
            classify_buttons_layout = QVBoxLayout()
            classify_buttons_layout.setSpacing(2)
            
            self.classify_selected_btn = QPushButton("Classify Selected Cells")
            self.classify_selected_btn.clicked.connect(self.classify_selected_cells)
            self.classify_selected_btn.setEnabled(False)
            classify_buttons_layout.addWidget(self.classify_selected_btn)
            
            self.classify_all_btn = QPushButton("Classify All Cells")
            self.classify_all_btn.clicked.connect(self.classify_all_cells)
            self.classify_all_btn.setEnabled(False)
            classify_buttons_layout.addWidget(self.classify_all_btn)
            
            classify_section_layout.addLayout(classify_buttons_layout)
            
            # Progress bar
            self.classification_progress = QProgressBar()
            self.classification_progress.setVisible(False)
            classify_section_layout.addWidget(self.classification_progress)
            
            classify_section.setLayout(classify_section_layout)
            classification_layout.addWidget(classify_section)
            
            # Manual correction section
            manual_section = QGroupBox("Manual Classification")
            manual_section_layout = QVBoxLayout()
            manual_section_layout.setSpacing(3)
            
            # Manual classification controls
            correction_layout = QHBoxLayout()
            correction_layout.addWidget(QLabel("Class:"))
            self.manual_class_combo = QComboBox()
            self.manual_class_combo.addItems(["Normal", "Budding"])
            correction_layout.addWidget(self.manual_class_combo)
            manual_section_layout.addLayout(correction_layout)
            
            self.apply_manual_btn = QPushButton("Apply to Selected")
            self.apply_manual_btn.clicked.connect(self.apply_manual_classification)
            self.apply_manual_btn.setEnabled(False)
            manual_section_layout.addWidget(self.apply_manual_btn)
            
            manual_section.setLayout(manual_section_layout)
            classification_layout.addWidget(manual_section)
            
            # Results section
            results_section = QGroupBox("Results")
            results_section_layout = QVBoxLayout()
            results_section_layout.setSpacing(3)
            
            # Classification results button
            self.view_results_btn = QPushButton("View Classification Results")
            self.view_results_btn.clicked.connect(self.show_classification_popup)
            self.view_results_btn.setEnabled(False)
            results_section_layout.addWidget(self.view_results_btn)
            
            # Batch delete buttons
            delete_buttons_layout = QHBoxLayout()
            delete_buttons_layout.setSpacing(2)
            
            self.delete_normal_btn = QPushButton("Delete Normal")
            self.delete_normal_btn.clicked.connect(self.delete_normal_cells)
            self.delete_normal_btn.setEnabled(False)
            self.delete_normal_btn.setStyleSheet("QPushButton { background-color: #ffcccc; }")
            delete_buttons_layout.addWidget(self.delete_normal_btn)
            
            self.delete_budding_btn = QPushButton("Delete Budding")
            self.delete_budding_btn.clicked.connect(self.delete_budding_cells)
            self.delete_budding_btn.setEnabled(False)
            self.delete_budding_btn.setStyleSheet("QPushButton { background-color: #ffcccc; }")
            delete_buttons_layout.addWidget(self.delete_budding_btn)
            
            results_section_layout.addLayout(delete_buttons_layout)
            
            # Export results
            self.export_results_btn = QPushButton("Export Results")
            self.export_results_btn.clicked.connect(self.export_classification_results)
            self.export_results_btn.setEnabled(False)
            results_section_layout.addWidget(self.export_results_btn)
            
            results_section.setLayout(results_section_layout)
            classification_layout.addWidget(results_section)
            
            classification_group.setLayout(classification_layout)
            right_sidebar_layout.addWidget(classification_group)
            
            # UNet Segmentation group
            unet_group = QGroupBox("UNet Cell Division Segmentation")
            unet_layout = QVBoxLayout()
            unet_layout.setSpacing(5)
            unet_layout.setContentsMargins(5, 5, 5, 5)
            
            # UNet model loading section
            unet_model_section = QGroupBox("UNet Model")
            unet_model_section_layout = QVBoxLayout()
            unet_model_section_layout.setSpacing(3)
            
            # UNet model loading button
            self.load_unet_btn = QPushButton("Load UNet Model")
            self.load_unet_btn.clicked.connect(self.load_unet_model)
            unet_model_section_layout.addWidget(self.load_unet_btn)
            
            # UNet model status
            self.unet_status_label = QLabel("No UNet model loaded")
            self.unet_status_label.setWordWrap(True)
            self.unet_status_label.setStyleSheet("font-size: 11px; color: gray;")
            unet_model_section_layout.addWidget(self.unet_status_label)
            
            unet_model_section.setLayout(unet_model_section_layout)
            unet_layout.addWidget(unet_model_section)
            
            # UNet segmentation controls section
            unet_segment_section = QGroupBox("Segmentation")
            unet_segment_section_layout = QVBoxLayout()
            unet_segment_section_layout.setSpacing(3)
            
            # Segmentation buttons
            segment_buttons_layout = QVBoxLayout()
            segment_buttons_layout.setSpacing(2)
            
            self.segment_selected_btn = QPushButton("Segment Selected Cells")
            self.segment_selected_btn.clicked.connect(self.segment_selected_cells)
            self.segment_selected_btn.setEnabled(False)
            segment_buttons_layout.addWidget(self.segment_selected_btn)
            
            self.segment_budding_btn = QPushButton("Segment Budding Cells")
            self.segment_budding_btn.clicked.connect(self.segment_budding_cells)
            self.segment_budding_btn.setEnabled(False)
            segment_buttons_layout.addWidget(self.segment_budding_btn)
            
            unet_segment_section_layout.addLayout(segment_buttons_layout)
            

            
            unet_segment_section.setLayout(unet_segment_section_layout)
            unet_layout.addWidget(unet_segment_section)
            
            # UNet results section
            unet_results_section = QGroupBox("Segmentation Results")
            unet_results_section_layout = QVBoxLayout()
            unet_results_section_layout.setSpacing(3)
            
            # Segmentation results display
            self.segmentation_results = QTextEdit()
            self.segmentation_results.setMaximumHeight(150)
            self.segmentation_results.setReadOnly(True)
            self.segmentation_results.setPlaceholderText("Segmentation results will appear here...")
            self.segmentation_results.setStyleSheet("font-family: monospace; font-size: 11px;")
            unet_results_section_layout.addWidget(self.segmentation_results)
            
            unet_results_section.setLayout(unet_results_section_layout)
            unet_layout.addWidget(unet_results_section)
            
            unet_group.setLayout(unet_layout)
            right_sidebar_layout.addWidget(unet_group)
            
            # Add stretch at bottom
            right_sidebar_layout.addStretch(1)
            
            right_sidebar.setLayout(right_sidebar_layout)
            right_sidebar.setMinimumWidth(260)
            right_sidebar.setMaximumWidth(300)
            main_layout.addWidget(right_sidebar)
        
        # Mouse and keyboard events
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Global keyboard shortcuts
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        
        # Undo shortcut
        self.undo_shortcut = QShortcut(QKeySequence.Undo, self)
        self.undo_shortcut.activated.connect(self.undo_action)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Initialize empty display
        self.update_display()
    
    def create_undo_icon(self):
        """Create undo icon for the button"""
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        undo_icon_path = os.path.join(script_dir, "undo.svg")
        
        # Set button size first - enlarged for better visibility
        self.undo_btn.setFixedSize(40, 35)
        
        # Set rounded rectangle styling for undo button
        self.undo_btn.setStyleSheet("""
            QPushButton {
                border: 2px solid #d0d0d0;
                border-radius: 8px;
                background-color: #f5f5f5;
                padding: 2px;
            }
            QPushButton:hover {
                border-color: #a0a0a0;
                background-color: #e8e8e8;
            }
            QPushButton:pressed {
                background-color: #c0c0c0;
            }
        """)
        
        try:
            # Try to load custom undo SVG icon
            print(f"Attempting to load undo icon from: {undo_icon_path}")
            print(f"File exists: {os.path.exists(undo_icon_path)}")
            
            if os.path.exists(undo_icon_path):
                icon = QIcon(undo_icon_path)
                print(f"Icon loaded: {not icon.isNull()}")
                print(f"Available sizes: {icon.availableSizes()}")
                
                if not icon.isNull():
                    self.undo_btn.setIcon(icon)
                    # Try explicit sizing with QSize - larger for better visibility
                    icon_size = QSize(28, 28)  # Explicit size in pixels
                    self.undo_btn.setIconSize(icon_size)
                    self.undo_btn.setText("")
                    print(f"Successfully set undo icon with size: {icon_size}")
                    return
                else:
                    raise Exception("Icon is null after loading")
            else:
                raise Exception("Undo icon file not found")
        except Exception as e:
            print(f"Warning: Could not load undo icon {undo_icon_path}: {e}")
            # Fallback to Unicode undo symbol
            self.undo_btn.setText("⟲")
            # Update styling to include rounded corners for text fallback
            self.undo_btn.setStyleSheet("""
                QPushButton {
                    border: 2px solid #d0d0d0;
                    border-radius: 8px;
                    background-color: #f5f5f5;
                    padding: 2px;
                    font-size: 18px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    border-color: #a0a0a0;
                    background-color: #e8e8e8;
                }
                QPushButton:pressed {
                    background-color: #c0c0c0;
                }
            """)
    
    def setup_tool_buttons(self):
        """Set up tool buttons with icons and styling"""
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tools_data = [
            (self.selector_btn, "selector", os.path.join(script_dir, "selector.svg"), "Selector"),
            (self.eraser_btn, "eraser", os.path.join(script_dir, "eraser.svg"), "Eraser"), 
            (self.divider_btn, "divider", os.path.join(script_dir, "divider.svg"), "Divider"),
            (self.drag_btn, "drag", os.path.join(script_dir, "drag.svg"), "Drag")
        ]
        
        for btn, tool_name, icon_path, tooltip in tools_data:
            btn.setCheckable(True)
            btn.setFixedSize(50, 35)
            btn.setToolTip(f"{tooltip} Tool")
            
            # Set rounded rectangle styling
            btn.setStyleSheet("""
                QPushButton {
                    border: 2px solid #d0d0d0;
                    border-radius: 8px;
                    background-color: #f5f5f5;
                    padding: 2px;
                }
                QPushButton:hover {
                    border-color: #a0a0a0;
                    background-color: #e8e8e8;
                }
                QPushButton:checked {
                    border-color: #4a90e2;
                    background-color: #ddeeff;
                }
                QPushButton:pressed {
                    background-color: #c0c0c0;
                }
            """)
            
            # Try to load custom icon files
            try:
                if os.path.exists(icon_path):
                    icon = QIcon(icon_path)
                    if not icon.isNull():
                        btn.setIcon(icon)
                        # SVG icons scale better, use slightly larger size
                        btn.setIconSize(btn.size() * 0.7)  # Scale icon to 70% of button size
                        btn.setText("")
                    else:
                        raise Exception("Failed to load icon")
                else:
                    raise Exception("Icon file not found")
            except Exception as e:
                print(f"Warning: Could not load icon {icon_path}: {e}")
                # Fallback to emoji/text
                if tool_name == "selector":
                    btn.setText("🖱")
                elif tool_name == "eraser":
                    btn.setText("⌫")
                elif tool_name == "divider":
                    btn.setText("✂")
                elif tool_name == "drag":
                    btn.setText("✋")
                
                # Update styling to include rounded corners for text fallback
                btn.setStyleSheet("""
                    QPushButton {
                        border: 2px solid #d0d0d0;
                        border-radius: 8px;
                        background-color: #f5f5f5;
                        padding: 2px;
                        font-size: 14px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        border-color: #a0a0a0;
                        background-color: #e8e8e8;
                    }
                    QPushButton:checked {
                        border-color: #4a90e2;
                        background-color: #ddeeff;
                    }
                    QPushButton:pressed {
                        background-color: #c0c0c0;
                    }
                """)
            
            # Store tool name as property
            btn.tool_name = tool_name
    
    def on_tool_button_clicked(self, button):
        """Handle tool button clicks"""
        tool_name = getattr(button, 'tool_name', 'selector')
        self.set_tool(tool_name)
    
    def toggle_file_mode(self):
        """Toggle between single file and batch processing mode"""
        if self.single_mode_radio.isChecked():
            self.file_stack.setCurrentIndex(0)  # Single file mode
            self.status_label.setText("Single file mode - Load a mask to begin editing")
            # Clear batch processing state
            self.file_list = []
            self.current_file_index = 0
            self.update_navigation_buttons()
            # Optimize space for single mode
            self.file_group.setMaximumHeight(140)
        else:
            self.file_stack.setCurrentIndex(1)  # Batch processing mode
            self.status_label.setText("Batch mode - Select input and output folders")
            # Clear current single file
            if hasattr(self, 'mask_file_label'):
                self.mask_file_label.setText("No mask loaded")
            # More space for batch mode
            self.file_group.setMaximumHeight(220)
            # Re-scan files if folders are already selected
            if self.input_folder and self.output_folder:
                self.rescan_files_and_update()
    
    def save_state(self):
        """Save current mask state to history for undo functionality"""
        if self.mask_img is not None:
            # Add current state to history
            self.mask_history.append(self.mask_img.copy())
            
            # Limit history size
            if len(self.mask_history) > self.max_history:
                self.mask_history.pop(0)
    
    def undo_action(self):
        """Undo the last action"""
        if len(self.mask_history) > 0:
            # Restore previous state
            self.mask_img = self.mask_history.pop().copy()
            
            # Clear selection
            self.selected_object_ids.clear()
            self.update_selection_display()
            
            # Update display
            self.update_display()
            self.update_object_list()
            self.status_label.setText("Undo applied")
        else:
            self.status_label.setText("No actions to undo")
    
    def select_input_folder(self):
        """Select input folder for batch processing"""
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_folder = folder
            self.input_folder_label.setText(os.path.basename(folder))
            
            self.rescan_files_and_update()
    
    def select_output_folder(self):
        """Select output folder for batch processing"""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_folder_label.setText(os.path.basename(folder))
            
            # Rescan files if input folder is also selected
            if self.input_folder:
                self.rescan_files_and_update()
    
    def rescan_files_and_update(self):
        """Scan files and filter out already processed ones"""
        if not self.input_folder:
            self.file_list = []
            self.file_progress_label.setText("No input folder selected")
            self.update_navigation_buttons()
            return
        
        # Find all image files in the input folder
        extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.bmp']
        all_files = []
        for ext in extensions:
            all_files.extend(glob.glob(os.path.join(self.input_folder, ext)))
            all_files.extend(glob.glob(os.path.join(self.input_folder, ext.upper())))
        
        all_files.sort()
        
        if self.output_folder:
            # Filter out files that already have corresponding output files (both mask and outline)
            self.file_list = []
            skipped_count = 0
            
            for input_file in all_files:
                filename = os.path.basename(input_file)
                name_without_ext = os.path.splitext(filename)[0]
                output_mask_file = os.path.join(self.output_folder, f"{name_without_ext}.tif")
                output_outline_file = os.path.join(self.output_folder, f"{name_without_ext}.txt")
                
                # Check if required output files exist based on settings
                mask_exists = os.path.exists(output_mask_file)
                outline_exists = os.path.exists(output_outline_file)
                
                # If outline generation is enabled, check for both files
                if self.generate_outlines_cb.isChecked():
                    if mask_exists and outline_exists:
                        skipped_count += 1
                        print(f"Skipping already processed file: {filename} (both mask and outline exist)")
                    else:
                        self.file_list.append(input_file)
                        if mask_exists or outline_exists:
                            print(f"Will reprocess {filename} (missing mask or outline file)")
                else:
                    # If outline generation is disabled, only check for mask file
                    if mask_exists:
                        skipped_count += 1
                        print(f"Skipping already processed file: {filename} (mask exists)")
                    else:
                        self.file_list.append(input_file)
            
            total_files = len(all_files)
            remaining_files = len(self.file_list)
            
            if self.file_list:
                self.file_progress_label.setText(f"{remaining_files}/{total_files} files to process")
                if skipped_count > 0:
                    self.status_label.setText(f"Skipped {skipped_count} already processed files")
            else:
                if total_files > 0:
                    self.file_progress_label.setText(f"All {total_files} files already processed")
                    self.status_label.setText("All files have been processed!")
                else:
                    self.file_progress_label.setText("No image files found")
        else:
            # No output folder selected, show all files
            self.file_list = all_files
            if self.file_list:
                self.file_progress_label.setText(f"Found {len(self.file_list)} files")
            else:
                self.file_progress_label.setText("No image files found")
        
        # Reset to first file
        self.current_file_index = 0
        self.update_navigation_buttons()
        
        # Load first file if both folders are selected and files are available
        if self.output_folder and self.file_list:
            self.load_current_file()
    
    def load_current_file(self):
        """Load the current file in batch processing"""
        if not self.file_list or self.current_file_index >= len(self.file_list):
            return
        
        file_path = self.file_list[self.current_file_index]
        
        try:
            # Load the mask
            self.mask_img = imread(file_path)
            
            # Convert to grayscale if needed
            if len(self.mask_img.shape) == 3:
                self.mask_img = self.mask_img[:,:,0]
            
            # Ensure it's integer type for object labeling
            self.mask_img = self.mask_img.astype(np.uint16)
            
            # Store original and clear history
            self.original_mask = self.mask_img.copy()
            self.mask_history = []
            self.mask_file_path = file_path
            
            # Clear selection and classification results for new file
            self.selected_object_ids.clear()
            self.cell_classifications = {}
            if TORCH_AVAILABLE:
                self.delete_normal_btn.setEnabled(False)
                self.delete_budding_btn.setEnabled(False)
                self.update_budding_segmentation_button_state()
            self.update_selection_display()
            
            # Auto renumber cells
            self.auto_renumber_cells()
            
            # Update display
            self.update_display()
            self.update_object_list()
            self.update_navigation_buttons()
            
            # Update classification display to clear results
            if TORCH_AVAILABLE:
                self.update_classification_display()
            
            filename = os.path.basename(file_path)
            self.mask_file_label.setText(f"Batch: {filename}")
            
            # Show progress with remaining files info
            remaining_after_current = len(self.file_list) - self.current_file_index
            self.file_progress_label.setText(f"File {self.current_file_index + 1}/{len(self.file_list)} ({remaining_after_current} remaining)")
            self.status_label.setText(f"Loaded: {filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load {file_path}: {str(e)}")
    
    def auto_save_current(self):
        """Auto save current file in batch processing"""
        if not self.output_folder or not self.file_list or self.mask_img is None:
            return
        
        input_file = self.file_list[self.current_file_index]
        filename = os.path.basename(input_file)
        
        # Change extension to .tif for output
        name_without_ext = os.path.splitext(filename)[0]
        output_file = os.path.join(self.output_folder, f"{name_without_ext}.tif")
        outline_file = os.path.join(self.output_folder, f"{name_without_ext}.txt")
        
        try:
            # Save as 16-bit TIFF to preserve object labels
            tiff.imwrite(output_file, self.mask_img.astype(np.uint16))
            
            # Save classification results if available
            classification_file = None
            if self.cell_classifications and TORCH_AVAILABLE:
                classification_file = os.path.join(self.output_folder, f"{name_without_ext}_classifications.json")
                export_data = {
                    'metadata': {
                        'export_date': datetime.now().isoformat(),
                        'model_path': self.model_path,
                        'mask_file': filename,
                        'total_cells': len(self.cell_classifications),
                        'normal_count': sum(1 for r in self.cell_classifications.values() if r['predicted_class'] == 0),
                        'budding_count': sum(1 for r in self.cell_classifications.values() if r['predicted_class'] == 1)
                    },
                    'classifications': self.cell_classifications
                }
                
                import json
                with open(classification_file, 'w') as f:
                    json.dump(export_data, f, indent=2)
            
            # Generate outline file if enabled
            if self.generate_outlines_cb.isChecked():
                # Generate outline file with the original filename as CY3 reference
                cy3_filename = filename  # Use original input filename
                outline_success = generate_fishquant_outline(self.mask_img, cy3_filename, outline_file)
                
                if outline_success:
                    status_msg = f"Auto-saved: {filename} (mask + outline"
                    if classification_file:
                        status_msg += " + classifications"
                    status_msg += ")"
                    self.status_label.setText(status_msg)
                else:
                    status_msg = f"Auto-saved: {filename} (mask only - outline failed"
                    if classification_file:
                        status_msg += ", classifications saved"
                    status_msg += ")"
                    self.status_label.setText(status_msg)
            else:
                status_msg = f"Auto-saved: {filename} (mask only"
                if classification_file:
                    status_msg += " + classifications"
                status_msg += ")"
                self.status_label.setText(status_msg)
            
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save {output_file}: {str(e)}")
            return False
    
    def next_file(self):
        """Move to next file in batch processing"""
        if not self.file_list:
            return
        
        # Auto save current file
        if self.mask_img is not None:
            self.auto_save_current()
        
        # Move to next file
        if self.current_file_index < len(self.file_list) - 1:
            self.current_file_index += 1
            self.load_current_file()
        else:
            # Check if there might be more files to process (re-scan)
            old_file_count = len(self.file_list)
            self.rescan_files_and_update()
            
            if len(self.file_list) > 0:
                # New files found or some were un-processed
                self.load_current_file()
            elif old_file_count > 0:
                QMessageBox.information(self, "Batch Complete", "All files have been processed!")
            else:
                self.status_label.setText("No more files to process")
    
    def previous_file(self):
        """Move to previous file in batch processing"""
        if not self.file_list:
            return
        
        # Auto save current file
        if self.mask_img is not None:
            self.auto_save_current()
        
        # Move to previous file
        if self.current_file_index > 0:
            self.current_file_index -= 1
            self.load_current_file()
    
    def update_navigation_buttons(self):
        """Update navigation button states"""
        has_files = bool(self.file_list)
        has_folders = bool(self.input_folder and self.output_folder)
        is_batch_mode = self.batch_mode_radio.isChecked()
        
        # Only enable navigation in batch mode when both folders are selected and files exist
        can_navigate = has_files and has_folders and is_batch_mode
        
        self.prev_btn.setEnabled(can_navigate and self.current_file_index > 0)
        self.next_btn.setEnabled(can_navigate and self.current_file_index < len(self.file_list) - 1)
        
        # Debug info
        print(f"Navigation update: files={has_files}, folders={has_folders}, batch={is_batch_mode}")
        print(f"Current index: {self.current_file_index}, Total files: {len(self.file_list) if self.file_list else 0}")
        print(f"Prev enabled: {self.prev_btn.isEnabled()}, Next enabled: {self.next_btn.isEnabled()}")
    
    def set_tool(self, tool):
        self.current_tool = tool
        self.divide_points = []  # Clear divide points when switching tools
        self.status_label.setText(f"Tool: {tool.title()}")
        if tool == "divider":
            self.status_label.setText("Tool: Divider - Drag to draw division line")
    
    def update_brush_size(self, value):
        self.brush_size = value
        self.brush_size_label.setText(str(value))
    
    def update_zoom(self, value):
        """Update zoom level from slider value"""
        self.zoom_level = value / 100.0  # Convert percentage to decimal
        self.zoom_label.setText(f"{value}%")
        self.update_display()
    
    def reset_zoom(self):
        """Reset zoom to 100%"""
        self.zoom_slider.setValue(100)
        self.zoom_center_x = 0.5
        self.zoom_center_y = 0.5
        self.update_zoom(100)
    
    def set_zoom_center(self, x_frac, y_frac):
        """Set the center point for zooming as fractions of image dimensions"""
        self.zoom_center_x = max(0, min(1, x_frac))
        self.zoom_center_y = max(0, min(1, y_frac))
        self.update_display()
    
    def load_mask(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Mask File", "", 
            "Image Files (*.png *.jpg *.tif *.tiff *.bmp);;All Files (*)"
        )
        if file_path:
            try:
                # Load the mask
                self.mask_img = imread(file_path)
                
                # Convert to grayscale if needed
                if len(self.mask_img.shape) == 3:
                    self.mask_img = self.mask_img[:,:,0]  # Take first channel
                
                # Ensure it's integer type for object labeling
                self.mask_img = self.mask_img.astype(np.uint16)
                
                # Store original for reference
                self.original_mask = self.mask_img.copy()
                self.mask_file_path = file_path
                
                # Clear undo history for new file
                self.mask_history = []
                
                # Clear classification results for new file
                self.cell_classifications = {}
                if TORCH_AVAILABLE:
                    self.delete_normal_btn.setEnabled(False)
                    self.delete_budding_btn.setEnabled(False)
                    self.update_budding_segmentation_button_state()
                
                # Update file label
                self.mask_file_label.setText(os.path.basename(file_path))
                
                # Auto renumber cells using logic similar to Mask2Outline.py
                self.auto_renumber_cells()
                
                # Update display
                self.update_display()
                self.update_object_list()
                
                # Update classification display to clear results
                if TORCH_AVAILABLE:
                    self.update_classification_display()
                
                self.status_label.setText(f"Loaded mask: {os.path.basename(file_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load mask: {str(e)}")
    
    def save_mask(self):
        if self.mask_img is None:
            QMessageBox.warning(self, "Warning", "No mask to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Mask File", "", 
            "TIFF Files (*.tif *.tiff);;PNG Files (*.png);;All Files (*)"
        )
        if file_path:
            try:
                # Save as 16-bit TIFF to preserve object labels
                if file_path.lower().endswith(('.tif', '.tiff')):
                    tiff.imwrite(file_path, self.mask_img.astype(np.uint16))
                else:
                    # For other formats, convert to 8-bit
                    max_val = np.max(self.mask_img)
                    if max_val > 255:
                        scaled = (self.mask_img * 255 / max_val).astype(np.uint8)
                    else:
                        scaled = self.mask_img.astype(np.uint8)
                    imsave(file_path, scaled)
                
                # Save classification results if available
                classification_saved = False
                if self.cell_classifications and TORCH_AVAILABLE:
                    base_name = os.path.splitext(file_path)[0]
                    classification_path = f"{base_name}_classifications.json"
                    
                    try:
                        export_data = {
                            'metadata': {
                                'export_date': datetime.now().isoformat(),
                                'model_path': self.model_path,
                                'mask_file': self.mask_file_path,
                                'total_cells': len(self.cell_classifications),
                                'normal_count': sum(1 for r in self.cell_classifications.values() if r['predicted_class'] == 0),
                                'budding_count': sum(1 for r in self.cell_classifications.values() if r['predicted_class'] == 1)
                            },
                            'classifications': self.cell_classifications
                        }
                        
                        import json
                        with open(classification_path, 'w') as f:
                            json.dump(export_data, f, indent=2)
                        classification_saved = True
                    except Exception as e:
                        print(f"Warning: Could not save classification results: {e}")
                
                # Generate outline file if enabled
                if self.generate_outlines_cb.isChecked():
                    base_name = os.path.splitext(file_path)[0]
                    outline_path = f"{base_name}.txt"
                    
                    # Use the mask filename as CY3 reference (or original filename if available)
                    cy3_filename = os.path.basename(self.mask_file_path) if self.mask_file_path else os.path.basename(file_path)
                    
                    # Generate FISH-QUANT outline file
                    outline_success = generate_fishquant_outline(self.mask_img, cy3_filename, outline_path)
                    
                    # Prepare success message
                    success_msg = f"Mask saved to: {file_path}\n"
                    if outline_success:
                        success_msg += f"Outline saved to: {outline_path}\n"
                    else:
                        success_msg += "Warning: Could not generate outline file\n"
                    
                    if classification_saved:
                        success_msg += f"Classifications saved to: {classification_path}"
                    
                    if outline_success:
                        QMessageBox.information(self, "Success", success_msg)
                    else:
                        QMessageBox.warning(self, "Partial Success", success_msg)
                else:
                    # No outline generation
                    success_msg = f"Mask saved to: {file_path}"
                    if classification_saved:
                        success_msg += f"\nClassifications saved to: {classification_path}"
                    QMessageBox.information(self, "Success", success_msg)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save mask: {str(e)}")
    
    def auto_renumber_cells(self):
        """Auto renumber cells similar to Mask2Outline.py logic"""
        if self.mask_img is None:
            return
        
        # Get unique labels (excluding background 0)
        unique_labels = np.unique(self.mask_img)
        unique_labels = unique_labels[unique_labels != 0]
        
        if len(unique_labels) == 0:
            return
        
        # Find contours for each object and get their top-most point for sorting
        cell_data = []
        for label in unique_labels:
            binary_mask = (self.mask_img == label).astype(np.uint8)
            contours = measure.find_contours(binary_mask, level=0.5)
            if contours:
                contour = max(contours, key=len)
                min_y = np.min(contour[:, 0])  # Top-most Y coordinate
                cell_data.append((label, min_y))
        
        # Sort by Y coordinate (top to bottom)
        cell_data = sorted(cell_data, key=lambda x: x[1])
        
        # Create new mask with renumbered cells
        new_mask = np.zeros_like(self.mask_img)
        self.cell_objects = {}
        
        for new_id, (old_label, min_y) in enumerate(cell_data, 1):
            mask_region = (self.mask_img == old_label)
            new_mask[mask_region] = new_id
            self.cell_objects[new_id] = {
                'label': new_id,
                'original_label': old_label,
                'min_y': min_y
            }
        
        self.mask_img = new_mask
        
        # Clear classification results since cell IDs have changed
        if self.cell_classifications:
            old_count = len(self.cell_classifications)
            self.cell_classifications = {}
            if TORCH_AVAILABLE:
                self.delete_normal_btn.setEnabled(False)
                self.delete_budding_btn.setEnabled(False)
                self.update_budding_segmentation_button_state()
                self.update_classification_display()
            self.status_label.setText(f"Renumbered {len(cell_data)} cells. Cleared {old_count} classification results.")
        else:
            self.status_label.setText(f"Renumbered {len(cell_data)} cells")
        
        # Clear selection since cell IDs have changed
        self.selected_object_ids.clear()
        self.update_selection_display()
        
        # Refresh the display and object list
        self.update_display()
        self.update_object_list()
    
    def update_object_list(self):
        """Update the object list widget"""
        self.object_list.clear()
        if self.mask_img is not None:
            unique_labels = np.unique(self.mask_img)
            unique_labels = unique_labels[unique_labels != 0]
            
            for label in sorted(unique_labels):
                item = QListWidgetItem(f"Cell {label}")
                item.setData(Qt.UserRole, label)
                
                # Highlight selected items
                if label in self.selected_object_ids:
                    item.setSelected(True)
                
                self.object_list.addItem(item)
    
    def select_object_from_list(self, item):
        """Select an object from the list"""
        object_id = item.data(Qt.UserRole)
        
        # Get modifiers from Qt application
        modifiers = QApplication.keyboardModifiers()
        
        # If Command/Ctrl is held, toggle selection
        if modifiers & (Qt.ControlModifier | Qt.MetaModifier):
            if object_id in self.selected_object_ids:
                self.selected_object_ids.remove(object_id)
            else:
                self.selected_object_ids.add(object_id)
        else:
            # Clear previous selection and select only this object
            self.selected_object_ids.clear()
            self.selected_object_ids.add(object_id)
        
        # Update UI
        self.update_selection_display()
        self.update_display()
    

    
    def delete_selected_object(self):
        """Delete the currently selected objects"""
        if self.selected_object_ids:
            self.save_state()  # Save state before making changes
            deleted_count = len(self.selected_object_ids)
            for obj_id in list(self.selected_object_ids): # Iterate over a copy
                self.mask_img[self.mask_img == obj_id] = 0
            self.selected_object_ids.clear()
            self.update_selection_display()
            self.update_display()
            self.update_object_list()
            self.status_label.setText(f"Deleted {deleted_count} selected object{'s' if deleted_count > 1 else ''}")
    

    
    def on_mouse_press(self, event):
        if not event.inaxes or self.mask_img is None:
            return
        
        # Check for zoom centering with Ctrl/Cmd + click
        if hasattr(event, 'key') and event.key in ['ctrl+', 'cmd+']:
            # Center zoom on clicked position
            img_width = self.mask_img.shape[1]
            img_height = self.mask_img.shape[0]
            x_frac = event.xdata / img_width
            y_frac = event.ydata / img_height
            self.set_zoom_center(x_frac, y_frac)
            return
        
        self.is_drawing = True
        self.last_pos = (event.xdata, event.ydata)
        
        if self.current_tool == "selector":
            # Select object at click position
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < self.mask_img.shape[1] and 0 <= y < self.mask_img.shape[0]:
                object_id = self.mask_img[y, x]
                if object_id != 0:
                    # Get modifiers from Qt application
                    modifiers = QApplication.keyboardModifiers()
                    
                    # If Command/Ctrl is held, toggle selection
                    if modifiers & (Qt.ControlModifier | Qt.MetaModifier):
                        if object_id in self.selected_object_ids:
                            self.selected_object_ids.remove(object_id)
                        else:
                            self.selected_object_ids.add(object_id)
                    else:
                        # Clear previous selection and select only this object
                        self.selected_object_ids.clear()
                        self.selected_object_ids.add(object_id)
                    
                    # Update UI
                    self.update_selection_display()
                    self.update_display()
                else:
                    # Clicked on background - clear selection if no modifier key
                    modifiers = QApplication.keyboardModifiers()
                    if not (modifiers & (Qt.ControlModifier | Qt.MetaModifier)):
                        self.selected_object_ids.clear()
                        self.update_selection_display()
                        self.update_display()
        
        elif self.current_tool == "divider":
            # Start division line
            self.divide_points = [(event.xdata, event.ydata)]
        
        elif self.current_tool == "drag":
            # Start dragging
            self.is_dragging = True
            self.drag_start_pos = (event.xdata, event.ydata)
        
        # Reset brush state for eraser
        if self.current_tool == "eraser":
            self._brush_state_saved = False
    
    def on_mouse_release(self, event):
        if self.current_tool == "divider" and len(self.divide_points) == 1 and event.inaxes:
            # Complete division line
            self.divide_points.append((event.xdata, event.ydata))
            self.apply_division()
            self.divide_points = []
        
        # Stop dragging
        if self.current_tool == "drag":
            self.is_dragging = False
            self.drag_start_pos = None
        
        self.is_drawing = False
    
    def on_mouse_move(self, event):
        if not event.inaxes or self.mask_img is None:
            return
        
        # Handle dragging
        if self.current_tool == "drag" and self.is_dragging and self.drag_start_pos is not None:
            # Calculate movement delta
            current_pos = (event.xdata, event.ydata)
            delta_x = current_pos[0] - self.drag_start_pos[0]
            delta_y = current_pos[1] - self.drag_start_pos[1]
            
            # Get image dimensions
            img_width = self.mask_img.shape[1]
            img_height = self.mask_img.shape[0]
            
            # Convert delta to fraction of image
            delta_x_frac = delta_x / img_width
            delta_y_frac = delta_y / img_height
            
            # Update zoom center (subtract to move in opposite direction)
            new_center_x = self.zoom_center_x - delta_x_frac / self.zoom_level
            new_center_y = self.zoom_center_y - delta_y_frac / self.zoom_level
            
            # Clamp to valid range
            self.zoom_center_x = max(0, min(1, new_center_x))
            self.zoom_center_y = max(0, min(1, new_center_y))
            
            # Update display
            self.update_display()
            return
        
        if not self.is_drawing:
            return
            
        if self.current_tool == "eraser":
            self.apply_brush_operation(event.xdata, event.ydata)
    
    def apply_brush_operation(self, x, y):
        """Apply brush operation (erase only)"""
        if x is None or y is None:
            return
        
        # Save state only once at the start of brush stroke
        if not hasattr(self, '_brush_state_saved') or not self._brush_state_saved:
            self.save_state()
            self._brush_state_saved = True
        
        center_x, center_y = int(x), int(y)
        
        # Create circular brush
        y_indices, x_indices = np.ogrid[:self.mask_img.shape[0], :self.mask_img.shape[1]]
        distance = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
        brush_mask = distance <= self.brush_size
        
        # Erase: set to background (0)
        self.mask_img[brush_mask] = 0
        
        self.update_display()
    
    def apply_division(self):
        """Apply division line to split objects"""
        if len(self.divide_points) < 2 or self.mask_img is None:
            return
        
        self.save_state()  # Save state before making changes
        
        # Get line coordinates
        x1, y1 = self.divide_points[0]
        x2, y2 = self.divide_points[1]
        
        # Create line mask
        line_mask = np.zeros_like(self.mask_img, dtype=bool)
        
        # Draw line using Bresenham's algorithm (simplified)
        num_points = int(max(abs(x2-x1), abs(y2-y1)) * 2)
        if num_points == 0:
            return
            
        x_coords = np.linspace(x1, x2, num_points).astype(int)
        y_coords = np.linspace(y1, y2, num_points).astype(int)
        
        # Filter out coordinates outside image bounds
        valid_indices = ((x_coords >= 0) & (x_coords < self.mask_img.shape[1]) & 
                        (y_coords >= 0) & (y_coords < self.mask_img.shape[0]))
        x_coords = x_coords[valid_indices]
        y_coords = y_coords[valid_indices]
        
        if len(x_coords) == 0:
            return
        
        # Set line pixels to 0 (background)
        line_mask[y_coords, x_coords] = True
        
        # Apply line mask to split objects
        original_mask = self.mask_img.copy()
        self.mask_img[line_mask] = 0
        
        # Re-label connected components for objects that were split
        affected_labels = np.unique(original_mask[line_mask])
        affected_labels = affected_labels[affected_labels != 0]
        
        for label in affected_labels:
            object_mask = (self.mask_img == label)
            if np.any(object_mask):
                # Find connected components
                labeled_components, num_components = ndimage.label(object_mask)
                
                if num_components > 1:
                    # Replace with new labels
                    self.mask_img[object_mask] = 0  # Clear old label
                    
                    max_label = np.max(self.mask_img)
                    for comp_id in range(1, num_components + 1):
                        comp_mask = (labeled_components == comp_id)
                        new_label = max_label + comp_id
                        self.mask_img[comp_mask] = new_label
        
        self.update_display()
        self.update_object_list()
        self.status_label.setText("Applied division")
        
        # Auto renumber cell numbers after division if enabled
        if self.auto_renumber_cb.isChecked():
            self.auto_renumber_cells()
    
    def apply_cell_number_edit(self):
        """Apply manual cell number edit with conflict resolution"""
        if not self.selected_object_ids:
            QMessageBox.warning(self, "Warning", "Please select a cell first")
            return
            
        if len(self.selected_object_ids) > 1:
            QMessageBox.warning(self, "Warning", "Cell number editing only works with a single selected cell. Please select only one cell.")
            return
        
        if self.mask_img is None:
            return
        
        try:
            new_number = int(self.cell_number_input.text().strip())
            if new_number <= 0:
                QMessageBox.warning(self, "Warning", "Cell number must be positive")
                return
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter a valid number")
            return
        
        self.save_state()  # Save state before making changes
        
        # Get current unique labels
        unique_labels = np.unique(self.mask_img)
        unique_labels = unique_labels[unique_labels != 0]
        
        # If the new number already exists and it's not in the selected cells
        if new_number in unique_labels and new_number not in self.selected_object_ids:
            # Find all cells that need to be shifted (>= new_number, excluding selected)
            cells_to_shift = unique_labels[unique_labels >= new_number]
            # Filter out selected cells from cells to shift
            cells_to_shift = [cell for cell in cells_to_shift if cell not in self.selected_object_ids]
            
            # Sort in descending order to avoid conflicts during shifting
            cells_to_shift = sorted(cells_to_shift, reverse=True)
            
            # Use temporary high numbers to avoid conflicts during shifting
            max_label = np.max(unique_labels)
            temp_offset = max_label + 1000  # Use high temporary numbers
            
            # First pass: assign temporary numbers
            for i, cell_id in enumerate(cells_to_shift):
                temp_number = temp_offset + i
                self.mask_img[self.mask_img == cell_id] = temp_number
            
            # Second pass: assign final numbers (shift each by +1)
            for i, original_cell_id in enumerate(cells_to_shift):
                temp_number = temp_offset + i
                final_number = original_cell_id + 1
                self.mask_img[self.mask_img == temp_number] = final_number
        
        # Apply the new number to the selected cell (we know there's only one)
        selected_object_id = next(iter(self.selected_object_ids))
        self.mask_img[self.mask_img == selected_object_id] = new_number
        
        # Update selected object IDs
        self.selected_object_ids.clear()
        self.selected_object_ids.add(new_number) # Keep the new number as the selected one
        self.update_selection_display()
        
        # Clear the input field
        self.cell_number_input.clear()
        
        # Update display and object list
        self.update_display()
        self.update_object_list()
        
        self.status_label.setText(f"Cell renumbered to {new_number}")
    
    def on_key_press(self, event):
        """Handle key press events"""
        if event.key == 'delete' or event.key == 'd':
            self.delete_selected_object()
        elif event.key == 'r':
            self.auto_renumber_cells()
            self.update_display()
        elif event.key == '+' or event.key == '=':
            # Zoom in
            current_value = self.zoom_slider.value()
            new_value = min(self.zoom_slider.maximum(), current_value + 25)
            self.zoom_slider.setValue(new_value)
        elif event.key == '-':
            # Zoom out
            current_value = self.zoom_slider.value()
            new_value = max(self.zoom_slider.minimum(), current_value - 25)
            self.zoom_slider.setValue(new_value)
        elif event.key == '0':
            # Reset zoom
            self.reset_zoom()
    
    def load_classification_model(self):
        """Load a trained classification model"""
        if not TORCH_AVAILABLE:
            QMessageBox.warning(self, "Warning", "PyTorch not available. Cannot load classification model.")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Classification Model", "", 
            "PyTorch Models (*.pth *.pt);;All Files (*)"
        )
        
        if file_path:
            try:
                # Initialize model
                self.classification_model = LightweightCNN(input_size=256, num_classes=2, dropout=0.3)
                
                # Load state dict
                state_dict = torch.load(file_path, map_location=self.device)
                self.classification_model.load_state_dict(state_dict)
                self.classification_model.to(self.device)
                self.classification_model.eval()
                
                self.model_path = file_path
                model_name = os.path.basename(file_path)
                self.model_status_label.setText(f"✅ {model_name}")
                
                # Enable classification buttons
                self.classify_selected_btn.setEnabled(True)
                self.classify_all_btn.setEnabled(True)
                self.export_results_btn.setEnabled(True)
                
                self.status_label.setText(f"Model loaded: {model_name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
                self.classification_model = None
                self.model_status_label.setText("❌ Model load failed")
    
    def classify_selected_cells(self):
        """Classify only the selected cells"""
        if not self.selected_object_ids or not self.classification_model or self.mask_img is None:
            QMessageBox.warning(self, "Warning", "Please select cells and load a model first.")
            return
        
        self.start_classification(list(self.selected_object_ids))
    
    def classify_all_cells(self):
        """Classify all cells in the mask"""
        if not self.classification_model or self.mask_img is None:
            QMessageBox.warning(self, "Warning", "Please load a model and mask first.")
            return
        
        # Get all unique cell labels
        unique_labels = np.unique(self.mask_img)
        cell_labels = unique_labels[unique_labels != 0]
        
        if len(cell_labels) == 0:
            QMessageBox.warning(self, "Warning", "No cells found in the mask.")
            return
        
        self.start_classification(list(cell_labels))
    
    def start_classification(self, cell_labels):
        """Start classification process in background thread"""
        if self.classification_worker and self.classification_worker.isRunning():
            QMessageBox.warning(self, "Warning", "Classification already in progress.")
            return
        
        # Disable buttons during classification
        self.classify_selected_btn.setEnabled(False)
        self.classify_all_btn.setEnabled(False)
        
        # Show progress bar
        self.classification_progress.setVisible(True)
        self.classification_progress.setValue(0)
        
        # Start worker thread
        self.classification_worker = ClassificationWorker(
            self.mask_img, self.classification_model, self.device, cell_labels
        )
        self.classification_worker.progress.connect(self.update_classification_progress)
        self.classification_worker.finished.connect(self.on_classification_finished)
        self.classification_worker.error.connect(self.on_classification_error)
        self.classification_worker.start()
        
        self.status_label.setText(f"Classifying {len(cell_labels)} cells...")
    
    def update_classification_progress(self, progress):
        """Update classification progress bar"""
        self.classification_progress.setValue(progress)
    
    def on_classification_finished(self, results):
        """Handle completion of classification"""
        # Update classification results
        self.cell_classifications.update(results)
        
        # Hide progress bar
        self.classification_progress.setVisible(False)
        
        # Re-enable buttons
        self.classify_selected_btn.setEnabled(True)
        self.classify_all_btn.setEnabled(True)
        self.apply_manual_btn.setEnabled(True)
        
        # Enable delete buttons if we have classification results
        if TORCH_AVAILABLE and self.cell_classifications:
            self.delete_normal_btn.setEnabled(True)
            self.delete_budding_btn.setEnabled(True)
        
        # Update display
        self.update_classification_display()
        self.update_display()  # Refresh the main visualization
        
        # Update budding segmentation button state
        self.update_budding_segmentation_button_state()
        
        # Show summary
        normal_count = sum(1 for r in results.values() if r['predicted_class'] == 0)
        budding_count = sum(1 for r in results.values() if r['predicted_class'] == 1)
        
        self.status_label.setText(f"Classification complete: {normal_count} Normal, {budding_count} Budding")
    
    def on_classification_error(self, error_msg):
        """Handle classification error"""
        self.classification_progress.setVisible(False)
        self.classify_selected_btn.setEnabled(True)
        self.classify_all_btn.setEnabled(True)
        
        QMessageBox.critical(self, "Classification Error", f"Classification failed: {error_msg}")
        self.status_label.setText("Classification failed")
    
    def update_classification_display(self):
        """Update the classification results button"""
        if not self.cell_classifications:
            self.view_results_btn.setText("View Classification Results")
            self.view_results_btn.setEnabled(False)
            return
        
        # Update button text with summary
        normal_count = sum(1 for r in self.cell_classifications.values() if r['predicted_class'] == 0)
        budding_count = sum(1 for r in self.cell_classifications.values() if r['predicted_class'] == 1)
        total = len(self.cell_classifications)
        
        button_text = f"View Results ({total} cells: {normal_count}N, {budding_count}B)"
        self.view_results_btn.setText(button_text)
        self.view_results_btn.setEnabled(True)
    
    def show_classification_popup(self):
        """Show detailed classification results in a popup dialog"""
        if not self.cell_classifications:
            QMessageBox.information(self, "Classification Results", "No classification results available.")
            return
        
        # Create formatted results text
        results_text = []
        results_text.append("Classification Results:")
        results_text.append("-" * 30)
        
        # Sort by cell ID
        sorted_cells = sorted(self.cell_classifications.items())
        
        for cell_id, result in sorted_cells:
            class_name = result['class_name']
            confidence = result['confidence']
            confidence_percent = confidence * 100
            
            # Add confidence indicator
            if confidence > 0.9:
                conf_indicator = "🟢"  # High confidence
            elif confidence > 0.7:
                conf_indicator = "🟡"  # Medium confidence
            else:
                conf_indicator = "🔴"  # Low confidence
            
            results_text.append(f"Cell {cell_id}: {class_name} {conf_indicator} ({confidence_percent:.1f}%)")
        
        # Add summary
        normal_count = sum(1 for r in self.cell_classifications.values() if r['predicted_class'] == 0)
        budding_count = sum(1 for r in self.cell_classifications.values() if r['predicted_class'] == 1)
        total = len(self.cell_classifications)
        
        results_text.append("-" * 30)
        results_text.append(f"Summary: {normal_count} Normal ({normal_count/total*100:.1f}%), "
                          f"{budding_count} Budding ({budding_count/total*100:.1f}%)")
        
        # Create custom dialog for better display
        from PyQt5.QtWidgets import QDialog
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Classification Results")
        dialog.setModal(True)
        dialog.resize(500, 600)
        
        layout = QVBoxLayout()
        
        # Create text display
        text_display = QTextEdit()
        text_display.setReadOnly(True)
        text_display.setPlainText("\n".join(results_text))
        text_display.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 12px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                padding: 10px;
            }
        """)
        layout.addWidget(text_display)
        
        # Add button layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # Copy button
        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText("\n".join(results_text)))
        copy_btn.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        button_layout.addWidget(copy_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                padding: 6px 12px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        dialog.setLayout(layout)
        
        dialog.exec_()
    
    def apply_manual_classification(self):
        """Apply manual classification to selected cells"""
        if not self.selected_object_ids:
            QMessageBox.warning(self, "Warning", "Please select cells to manually classify.")
            return
        
        manual_class = self.manual_class_combo.currentText()
        predicted_class = 0 if manual_class == "Normal" else 1
        
        # Apply manual classification to selected cells
        for cell_id in self.selected_object_ids:
            self.cell_classifications[cell_id] = {
                'predicted_class': predicted_class,
                'confidence': 1.0,  # Manual classification has full confidence
                'class_name': manual_class,
                'probabilities': [1.0, 0.0] if predicted_class == 0 else [0.0, 1.0],
                'manual': True  # Mark as manually classified
            }
        
        # Enable delete buttons if we have classification results
        if TORCH_AVAILABLE and self.cell_classifications:
            self.delete_normal_btn.setEnabled(True)
            self.delete_budding_btn.setEnabled(True)
        
        # Update display
        self.update_classification_display()
        self.update_display()
        
        # Update budding segmentation button state
        self.update_budding_segmentation_button_state()
        
        cell_count = len(self.selected_object_ids)
        self.status_label.setText(f"Manually classified {cell_count} cell{'s' if cell_count > 1 else ''} as {manual_class}")
    
    def delete_normal_cells(self):
        """Delete all cells classified as Normal"""
        if not self.cell_classifications:
            QMessageBox.warning(self, "Warning", "No classification results available.")
            return
        
        # Find all normal cells
        normal_cells = [cell_id for cell_id, result in self.cell_classifications.items() 
                       if result['predicted_class'] == 0]
        
        if not normal_cells:
            QMessageBox.information(self, "Info", "No Normal cells found to delete.")
            return
        
        # Confirm deletion
        reply = QMessageBox.question(
            self, "Confirm Deletion", 
            f"Are you sure you want to delete {len(normal_cells)} Normal cells?\n"
            f"This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.save_state()  # Save state before making changes
            
            # Delete the cells from mask
            for cell_id in normal_cells:
                self.mask_img[self.mask_img == cell_id] = 0
                # Remove from classifications
                if cell_id in self.cell_classifications:
                    del self.cell_classifications[cell_id]
            
            # Clear selection if any deleted cells were selected
            self.selected_object_ids = {cell_id for cell_id in self.selected_object_ids 
                                       if cell_id not in normal_cells}
            
            # Update displays
            self.update_selection_display()
            self.update_classification_display()
            self.update_display()
            self.update_object_list()
            
            # Update budding segmentation button state
            self.update_budding_segmentation_button_state()
            
            self.status_label.setText(f"Deleted {len(normal_cells)} Normal cells")
    
    def delete_budding_cells(self):
        """Delete all cells classified as Budding"""
        if not self.cell_classifications:
            QMessageBox.warning(self, "Warning", "No classification results available.")
            return
        
        # Find all budding cells
        budding_cells = [cell_id for cell_id, result in self.cell_classifications.items() 
                        if result['predicted_class'] == 1]
        
        if not budding_cells:
            QMessageBox.information(self, "Info", "No Budding cells found to delete.")
            return
        
        # Confirm deletion
        reply = QMessageBox.question(
            self, "Confirm Deletion", 
            f"Are you sure you want to delete {len(budding_cells)} Budding cells?\n"
            f"This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.save_state()  # Save state before making changes
            
            # Delete the cells from mask
            for cell_id in budding_cells:
                self.mask_img[self.mask_img == cell_id] = 0
                # Remove from classifications
                if cell_id in self.cell_classifications:
                    del self.cell_classifications[cell_id]
            
            # Clear selection if any deleted cells were selected
            self.selected_object_ids = {cell_id for cell_id in self.selected_object_ids 
                                       if cell_id not in budding_cells}
            
            # Update displays
            self.update_selection_display()
            self.update_classification_display()
            self.update_display()
            self.update_object_list()
            
            # Update budding segmentation button state
            self.update_budding_segmentation_button_state()
            
            self.status_label.setText(f"Deleted {len(budding_cells)} Budding cells")
    
    def export_classification_results(self):
        """Export classification results to JSON file"""
        if not self.cell_classifications:
            QMessageBox.warning(self, "Warning", "No classification results to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Classification Results", "", 
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                # Prepare export data
                export_data = {
                    'metadata': {
                        'export_date': datetime.now().isoformat(),
                        'model_path': self.model_path,
                        'mask_file': self.mask_file_path,
                        'total_cells': len(self.cell_classifications),
                        'normal_count': sum(1 for r in self.cell_classifications.values() if r['predicted_class'] == 0),
                        'budding_count': sum(1 for r in self.cell_classifications.values() if r['predicted_class'] == 1)
                    },
                    'classifications': self.cell_classifications
                }
                
                # Save to JSON
                import json
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                QMessageBox.information(self, "Success", f"Results exported to: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export results: {str(e)}")
    
    def load_unet_model(self):
        """Load a trained UNet segmentation model"""
        if not TORCH_AVAILABLE:
            QMessageBox.warning(self, "Warning", "PyTorch not available. Cannot load UNet model.")
            return
        
        # Default path if exists, otherwise show file dialog
        default_path = "/Volumes/ExFAT/Archive/models/best_model.pth"
        
        if os.path.exists(default_path):
            file_path = default_path
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load UNet Model", "", 
                "PyTorch Models (*.pth *.pt);;All Files (*)"
            )
        
        if file_path:
            try:
                # Initialize UNet model
                self.unet_model = UNet(in_channels=1, num_classes=3)
                
                # Load state dict
                state_dict = torch.load(file_path, map_location=self.device)
                self.unet_model.load_state_dict(state_dict)
                self.unet_model.to(self.device)
                self.unet_model.eval()
                
                self.unet_model_path = file_path
                model_name = os.path.basename(file_path)
                self.unet_status_label.setText(f"✅ {model_name}")
                
                # Enable segmentation buttons
                self.segment_selected_btn.setEnabled(True)
                self.update_budding_segmentation_button_state()
                
                self.status_label.setText(f"UNet model loaded: {model_name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load UNet model: {str(e)}")
                self.unet_model = None
                self.unet_status_label.setText("❌ UNet model load failed")
    
    def segment_selected_cells(self):
        """Segment only the selected cells using UNet"""
        if not self.selected_object_ids or not self.unet_model or self.mask_img is None:
            QMessageBox.warning(self, "Warning", "Please select cells and load UNet model first.")
            return
        
        # Check cell classifications and warn user if needed
        if not self.warn_before_segmentation(list(self.selected_object_ids)):
            return
        
        self.perform_unet_segmentation(list(self.selected_object_ids))
    
    def segment_budding_cells(self):
        """Segment only budding cells in the mask using UNet"""
        if not self.unet_model or self.mask_img is None:
            QMessageBox.warning(self, "Warning", "Please load UNet model and mask first.")
            return
        
        # Get budding cells from classification results
        budding_cell_labels = self.get_budding_cell_labels()
        
        if len(budding_cell_labels) == 0:
            QMessageBox.warning(self, "Warning", "No budding cells found. Please classify cells first or select specific budding cells.")
            return
        
        # Show info about what will be segmented
        reply = QMessageBox.question(
            self, "Segment Budding Cells",
            f"Found {len(budding_cell_labels)} budding cells to segment.\n\n"
            f"Failed cells will be reported with specific reasons.\n\n"
            f"Proceed with segmentation?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            self.perform_unet_segmentation(budding_cell_labels)
    
    def perform_unet_segmentation(self, cell_labels):
        """Perform UNet segmentation on specified cells"""
        if not self.unet_model:
            return
        
        self.save_state()  # Save state before making changes
        
        segmentation_results = []
        processed_count = 0
        failed_count = 0
        
        for cell_id in cell_labels:
            try:
                # Extract cell mask
                cell_mask = (self.mask_img == cell_id).astype(np.float32)
                
                # Check if cell is large enough for segmentation
                if np.sum(cell_mask) < 100:  # Skip very small cells
                    segmentation_results.append(f"Cell {cell_id}: Skipped (too small)")
                    continue
                
                # Predict segmentation
                segmented_mask = self.predict_cell_division_unet(cell_mask)
                
                # Check if segmentation was successful
                success, failure_reason = self.evaluate_segmentation_success(cell_id, segmented_mask)
                
                if success:
                    # Replace original cell with segmented cells
                    self.replace_cell_with_segmentation(cell_id, segmented_mask)
                    segmentation_results.append(f"Cell {cell_id}: Successfully replaced with mother/daughter")
                    processed_count += 1
                else:
                    segmentation_results.append(f"Cell {cell_id}: Failed - {failure_reason}")
                    failed_count += 1
                    
            except Exception as e:
                segmentation_results.append(f"Cell {cell_id}: Error - {str(e)}")
                failed_count += 1
        
        # Update displays
        self.update_display()
        self.update_object_list()
        self.update_segmentation_results(segmentation_results, processed_count, failed_count)
        
        # Provide intelligent feedback with results details
        failed_cells = [result for result in segmentation_results if "Failed -" in result]
        self.provide_segmentation_feedback(processed_count, failed_count, len(cell_labels), failed_cells)
        
        self.status_label.setText(f"Segmentation complete: {processed_count} processed, {failed_count} failed")
    
    def predict_cell_division_unet(self, input_mask):
        """Predict cell division segmentation using UNet"""
        try:
            # Find bounding box
            coords = np.argwhere(input_mask > 0)
            if len(coords) == 0:
                return None
            
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Add padding (ensure we stay within image bounds)
            padding = 10
            y_min_pad = max(0, y_min - padding)
            x_min_pad = max(0, x_min - padding)
            y_max_pad = min(input_mask.shape[0], y_max + padding + 1)  # +1 for inclusive slicing
            x_max_pad = min(input_mask.shape[1], x_max + padding + 1)  # +1 for inclusive slicing
            
            # Crop the region
            cropped = input_mask[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
            original_cropped_shape = cropped.shape
            
            # Resize to target size for UNet input
            resized = cv2.resize(cropped.astype(np.uint8), self.target_size, interpolation=cv2.INTER_NEAREST)
            
            # Convert to tensor
            input_tensor = torch.from_numpy(resized.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            input_tensor = input_tensor.to(self.device)
            
            # Predict
            self.unet_model.eval()
            with torch.no_grad():
                output = self.unet_model(input_tensor)
                prediction = torch.argmax(output, dim=1).cpu().numpy()[0]
            
            # Resize prediction back to original cropped size with exact dimensions
            pred_cropped_size = cv2.resize(prediction.astype(np.uint8), 
                                         (original_cropped_shape[1], original_cropped_shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
            
            # Create result array and place prediction back at exact coordinates
            result = np.zeros_like(input_mask, dtype=np.uint8)
            result[y_min_pad:y_max_pad, x_min_pad:x_max_pad] = pred_cropped_size
            
            # Mask the result to only include pixels where the original input was non-zero
            # This ensures we only segment within the original cell boundaries
            result = result * (input_mask > 0).astype(np.uint8)
            
            return result
            
        except Exception as e:
            print(f"UNet prediction error: {e}")
            return None
    
    def replace_cell_with_segmentation(self, cell_id, segmented_mask):
        """Replace original cell with mother/daughter segmentation"""
        # Clear original cell
        self.mask_img[self.mask_img == cell_id] = 0
        
        # Find unique labels in segmentation (1=daughter, 2=mother)
        daughter_mask = (segmented_mask == 1)
        mother_mask = (segmented_mask == 2)
        
        # Get next available cell IDs
        max_label = np.max(self.mask_img) if np.any(self.mask_img > 0) else 0
        
        # Add daughter and mother cells
        if np.any(daughter_mask):
            daughter_id = max_label + 1
            self.mask_img[daughter_mask] = daughter_id
            max_label = daughter_id
        
        if np.any(mother_mask):
            mother_id = max_label + 1
            self.mask_img[mother_mask] = mother_id
    
    def add_segmentation_as_new_cells(self, segmented_mask):
        """Add segmented mother/daughter as new cells"""
        daughter_mask = (segmented_mask == 1)
        mother_mask = (segmented_mask == 2)
        
        # Get next available cell IDs
        max_label = np.max(self.mask_img) if np.any(self.mask_img > 0) else 0
        new_ids = []
        
        # Add daughter and mother cells
        if np.any(daughter_mask):
            daughter_id = max_label + 1
            self.mask_img[daughter_mask] = daughter_id
            new_ids.append(daughter_id)
            max_label = daughter_id
        
        if np.any(mother_mask):
            mother_id = max_label + 1
            self.mask_img[mother_mask] = mother_id
            new_ids.append(mother_id)
        
        return new_ids
    
    def update_segmentation_results(self, results_list, processed_count, failed_count):
        """Update the segmentation results display"""
        results_text = []
        results_text.append("UNet Segmentation Results:")
        results_text.append("-" * 30)
        
        for result in results_list:
            results_text.append(result)
        
        # Add summary
        total = processed_count + failed_count
        results_text.append("-" * 30)
        results_text.append(f"Summary: {processed_count}/{total} successful ({processed_count/total*100:.1f}%)")
        if failed_count > 0:
            results_text.append(f"Failed: {failed_count} cells")
        
        self.segmentation_results.setText("\n".join(results_text))
    
    def get_budding_cell_labels(self):
        """Get list of cell labels that are classified as budding"""
        budding_labels = []
        
        if self.cell_classifications:
            # Get budding cells from classification results
            for cell_id, classification in self.cell_classifications.items():
                if classification['predicted_class'] == 1:  # 1 = Budding
                    budding_labels.append(cell_id)
        
        return budding_labels
    
    def evaluate_segmentation_success(self, cell_id, segmented_mask):
        """Evaluate if segmentation was successful and return reason if failed"""
        if segmented_mask is None:
            return False, "Model prediction failed"
        
        # Check number of segments
        unique_segments = np.unique(segmented_mask)
        num_segments = len(unique_segments) - 1  # Exclude background (0)
        
        if num_segments < 2:
            return False, "Only one segment produced (no division detected)"
        
        # Get daughter and mother masks
        daughter_mask = (segmented_mask == 1)
        mother_mask = (segmented_mask == 2)
        
        # Check if both segments exist
        if not np.any(daughter_mask):
            return False, "No daughter cell detected"
        if not np.any(mother_mask):
            return False, "No mother cell detected"
        
        # Check for daughter cell containment within mother cell (model bug)
        if self.is_daughter_contained_in_mother(daughter_mask, mother_mask):
            return False, "Daughter cell completely contained within mother cell"
        
        return True, "Success"
    
    def is_daughter_contained_in_mother(self, daughter_mask, mother_mask):
        """Check if daughter cell is completely contained within mother cell boundaries"""
        
        # Get bounding boxes
        daughter_coords = np.argwhere(daughter_mask)
        mother_coords = np.argwhere(mother_mask)
        
        if len(daughter_coords) == 0 or len(mother_coords) == 0:
            return False
        
        # Get daughter bounding box
        d_y_min, d_x_min = daughter_coords.min(axis=0)
        d_y_max, d_x_max = daughter_coords.max(axis=0)
        
        # Get mother bounding box
        m_y_min, m_x_min = mother_coords.min(axis=0)
        m_y_max, m_x_max = mother_coords.max(axis=0)
        
        # Check if daughter bounding box is completely within mother bounding box
        # with some tolerance (allow small overlap)
        tolerance = 3  # pixels
        
        daughter_within_mother = (
            d_y_min >= (m_y_min - tolerance) and
            d_y_max <= (m_y_max + tolerance) and
            d_x_min >= (m_x_min - tolerance) and
            d_x_max <= (m_x_max + tolerance)
        )
        
        # Also check overlap percentage - if daughter is >90% inside mother, it's contained
        if daughter_within_mother:
            # Calculate overlap
            overlap = np.logical_and(daughter_mask, mother_mask)
            overlap_ratio = np.sum(overlap) / np.sum(daughter_mask)
            
            # If more than 90% of daughter overlaps with mother, consider it contained
            if overlap_ratio > 0.9:
                return True
        
        return False
    
    def warn_before_segmentation(self, selected_cells):
        """Warn user about applying segmentation to normal cells"""
        normal_cells = []
        budding_cells = []
        unclassified_cells = []
        
        for cell_id in selected_cells:
            if cell_id in self.cell_classifications:
                classification = self.cell_classifications[cell_id]
                if classification['predicted_class'] == 0:  # Normal
                    normal_cells.append(cell_id)
                else:  # Budding
                    budding_cells.append(cell_id)
            else:
                unclassified_cells.append(cell_id)
        
        # Build warning message
        warning_parts = []
        
        if normal_cells:
            warning_parts.append(f"⚠️ {len(normal_cells)} Normal cells selected")
            warning_parts.append("Segmentation is not recommended for normal cells as they don't have division structures to separate.")
        
        if unclassified_cells:
            warning_parts.append(f"❓ {len(unclassified_cells)} Unclassified cells selected") 
            warning_parts.append("Consider classifying these cells first to identify which are suitable for segmentation.")
        
        if budding_cells:
            warning_parts.append(f"✅ {len(budding_cells)} Budding cells selected")
        
        # Show warning if there are potential issues
        if normal_cells or unclassified_cells:
            warning_msg = "\n".join(warning_parts)
            warning_msg += "\n\nDo you want to proceed with segmentation?"
            
            reply = QMessageBox.question(
                self, "Segmentation Analysis", 
                warning_msg,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            return reply == QMessageBox.Yes
        
        # If only budding cells, proceed without warning
        return True
    
    def provide_segmentation_feedback(self, successful, failed, total, failed_cells=None):
        """Provide user feedback on segmentation quality with context"""
        if total == 0:
            return
        
        success_rate = successful / total * 100
        
        feedback_parts = [
            f"Segmentation Results:",
            f"✅ Successful: {successful}",
            f"❌ Failed: {failed}",
            f"📊 Success Rate: {success_rate:.0f}%"
        ]
        
        # Show failed cells details if requested and available
        if failed > 0 and failed_cells and len(failed_cells) > 0:
            feedback_parts.append(f"\n🔍 Failed Cells Details:")
            
            # Group failures by reason
            failure_reasons = {}
            for failure in failed_cells:
                # Extract reason from "Cell X: Failed - reason"
                if " - " in failure:
                    reason = failure.split(" - ", 1)[1]
                    cell_info = failure.split(":")[0]
                    if reason not in failure_reasons:
                        failure_reasons[reason] = []
                    failure_reasons[reason].append(cell_info)
            
            for reason, cells in failure_reasons.items():
                if len(cells) <= 3:
                    cell_list = ", ".join(cells)
                else:
                    cell_list = ", ".join(cells[:3]) + f" and {len(cells)-3} more"
                feedback_parts.append(f"   • {reason}: {cell_list}")
        elif failed > 0:
            feedback_parts.append(f"\n📝 Note: Some failures are normal - not all budding cells can be cleanly separated.")
        
        feedback_msg = "\n".join(feedback_parts)
        
        # Don't show popup for very small batches, but always show for failed cells details
        if total >= 3 or (failed > 0 and failed_cells):
            QMessageBox.information(self, "Segmentation Complete", feedback_msg)
    
    def update_budding_segmentation_button_state(self):
        """Update the state of the budding segmentation button based on available budding cells"""
        if not hasattr(self, 'segment_budding_btn'):
            return
            
        # Check if UNet model is loaded and we have budding cells
        has_unet_model = self.unet_model is not None
        budding_cells = self.get_budding_cell_labels()
        has_budding_cells = len(budding_cells) > 0
        
        # Enable button only if both conditions are met
        self.segment_budding_btn.setEnabled(has_unet_model and has_budding_cells)
        
        # Update tooltip to show current state
        if not has_unet_model:
            self.segment_budding_btn.setToolTip("Load UNet model first")
        elif not has_budding_cells:
            self.segment_budding_btn.setToolTip("No budding cells found. Classify cells first to identify budding cells.")
        else:
            self.segment_budding_btn.setToolTip(f"Segment {len(budding_cells)} budding cells using UNet")
    
    def update_selection_display(self):
        """Update the selection display labels and input field"""
        if not self.selected_object_ids:
            self.selected_label.setText("Selected: None")
            self.cell_number_input.clear()
            if TORCH_AVAILABLE:
                self.apply_manual_btn.setEnabled(False)
        elif len(self.selected_object_ids) == 1:
            cell_id = next(iter(self.selected_object_ids))
            
            # Include classification info if available
            if cell_id in self.cell_classifications:
                class_info = self.cell_classifications[cell_id]
                class_name = class_info['class_name']
                confidence = class_info['confidence'] * 100
                manual_tag = " (Manual)" if class_info.get('manual', False) else ""
                self.selected_label.setText(f"Selected: Cell {cell_id} - {class_name} ({confidence:.1f}%){manual_tag}")
            else:
                self.selected_label.setText(f"Selected: Cell {cell_id}")
            
            self.cell_number_input.setText(str(cell_id))
            if TORCH_AVAILABLE:
                self.apply_manual_btn.setEnabled(True)
        else:
            sorted_ids = sorted(self.selected_object_ids)
            self.selected_label.setText(f"Selected: {len(self.selected_object_ids)} cells ({', '.join(map(str, sorted_ids[:3]))}{'...' if len(sorted_ids) > 3 else ''})")
            self.cell_number_input.setText("Multiple")
            if TORCH_AVAILABLE:
                self.apply_manual_btn.setEnabled(True)
    
    def update_display(self):
        """Update the visualization with classification colors"""
        self.figure.clear()
        
        if self.mask_img is None:
            self.canvas.draw()
            return
        
        ax = self.figure.add_subplot(111)
        ax.set_axis_off()
        
        # Remove all margins and padding
        ax.margins(0)
        ax.set_position([0, 0, 1, 1])  # Use full figure area
        
        # Create colored visualization
        colored_mask = np.zeros((*self.mask_img.shape, 3))
        unique_labels = np.unique(self.mask_img)
        unique_labels = unique_labels[unique_labels != 0]
        
        for i, label in enumerate(unique_labels):
            mask_region = (self.mask_img == label)
            
            # Use classification-based colors if available
            if label in self.cell_classifications:
                class_result = self.cell_classifications[label]
                if class_result['predicted_class'] == 0:  # Normal
                    color = [0.2, 0.8, 0.2]  # Green for normal
                else:  # Budding
                    color = [0.8, 0.2, 0.2]  # Red for budding
                
                # Adjust brightness based on confidence
                confidence = class_result['confidence']
                color = [c * (0.5 + 0.5 * confidence) for c in color]
            else:
                # Use default colors for unclassified cells
                color = self.object_colors[i % len(self.object_colors)][:3]
            
            colored_mask[mask_region] = color
        
        # Highlight selected objects
        if self.selected_object_ids:
            for obj_id in self.selected_object_ids:
                selected_mask = (self.mask_img == obj_id)
                colored_mask[selected_mask] = [1, 1, 0]  # Yellow for selected
        
        # Display the mask with no padding
        ax.imshow(colored_mask, extent=[0, self.mask_img.shape[1], self.mask_img.shape[0], 0])
        
        # Show object borders if enabled
        if self.show_borders_cb.isChecked():
            for label in unique_labels:
                binary_mask = (self.mask_img == label).astype(np.uint8)
                contours = measure.find_contours(binary_mask, level=0.5)
                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], 'w-', linewidth=1)
        
        # Show cell numbers if enabled
        if self.show_numbers_cb.isChecked():
            # Check if we have many cells and need extra compact display
            num_cells = len(unique_labels)
            compact_mode = num_cells > 20 or self.compact_labels_cb.isChecked()  # Use more compact display for many cells or user preference
            
            for label in unique_labels:
                # Find centroid of the object
                binary_mask = (self.mask_img == label)
                y_coords, x_coords = np.where(binary_mask)
                if len(y_coords) > 0:
                    centroid_y = np.mean(y_coords)
                    centroid_x = np.mean(x_coords)
                    
                    # Calculate adaptive font size and box properties based on cell size
                    cell_area = len(y_coords)
                    cell_diameter = np.sqrt(cell_area / np.pi) * 2  # Approximate diameter
                    
                    # Adaptive font size (smaller and based on cell size)
                    base_font_size = 5 if compact_mode else 6  # Even smaller for many cells
                    font_size = max(3, min(8 if compact_mode else 10, base_font_size + cell_diameter * 0.02))
                    
                    # Adaptive box padding (smaller)
                    base_padding = 0.08 if compact_mode else 0.1  # Even smaller padding for many cells
                    padding = max(0.03, min(0.12 if compact_mode else 0.15, base_padding + cell_diameter * 0.002))
                    
                    # Adaptive alpha (more transparent for small cells and many cells)
                    base_alpha = 0.55 if compact_mode else 0.6
                    alpha = max(0.4, min(0.75 if compact_mode else 0.8, base_alpha + cell_diameter * 0.002))
                    
                    # Add classification info to label if available
                    label_text = str(label)
                    if label in self.cell_classifications:
                        class_name = self.cell_classifications[label]['class_name']
                        confidence = self.cell_classifications[label]['confidence']
                        short_name = "N" if class_name == "Normal" else "B"
                        
                        if compact_mode:
                            # Very compact format for many cells: just show class letter
                            label_text = f"{label}\n{short_name}"
                        else:
                            # Shorter format for smaller display
                            label_text = f"{label}\n{short_name}({confidence:.1f})"  # Reduced precision
                    
                    ax.text(centroid_x, centroid_y, label_text, 
                           color='white', fontsize=font_size, ha='center', va='center',
                           bbox=dict(boxstyle=f'round,pad={padding}', facecolor='black', alpha=alpha))
        
        # Set proper aspect ratio and limits with zoom
        img_width = self.mask_img.shape[1]
        img_height = self.mask_img.shape[0]
        
        # Calculate zoom window dimensions
        zoom_width = img_width / self.zoom_level
        zoom_height = img_height / self.zoom_level
        
        # Calculate center position in image coordinates
        center_x = self.zoom_center_x * img_width
        center_y = self.zoom_center_y * img_height
        
        # Calculate zoom window boundaries
        x_min = max(0, center_x - zoom_width / 2)
        x_max = min(img_width, center_x + zoom_width / 2)
        y_min = max(0, center_y - zoom_height / 2)
        y_max = min(img_height, center_y + zoom_height / 2)
        
        # Adjust if we hit boundaries to maintain zoom level
        if x_max - x_min < zoom_width:
            if x_min == 0:
                x_max = min(img_width, zoom_width)
            elif x_max == img_width:
                x_min = max(0, img_width - zoom_width)
        
        if y_max - y_min < zoom_height:
            if y_min == 0:
                y_max = min(img_height, zoom_height)
            elif y_max == img_height:
                y_min = max(0, img_height - zoom_height)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # Note: y is flipped for images
        ax.set_aspect('equal')
        
        # Ensure tight layout with no padding
        self.figure.tight_layout(pad=0)
        
        self.canvas.draw()
    
    def showEvent(self, event):
        super().showEvent(event)
        # Remove always-on-top flag after showing
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.show()

def main():
    app = QApplication(sys.argv)
    window = MaskEditor()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 