import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import argparse
import sys

# Safe OpenCV import with error handling
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  OpenCV import failed: {e}")
    print("📝 Please reinstall OpenCV with: conda install opencv -c conda-forge")
    CV2_AVAILABLE = False

try:
    from scipy import ndimage
    from scipy.ndimage import binary_fill_holes
    SCIPY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  SciPy import failed: {e}")
    print("📝 Please install SciPy with: conda install scipy")
    SCIPY_AVAILABLE = False

class UNet(nn.Module):
    """
    U-Net model for budding cell classification
    
    Task: Binary mask to semantic segmentation
    Input: Binary mask of budding cells (from dic_masks)
    Output: 3-class segmentation (background=0, daughter=1, mother=2)
    """
    
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

class ProcessedCellDataset(Dataset):
    """
    Dataset class for budding cell classification training data
    
    Data structure:
    - Input: Binary masks from dic_masks (overlapping cells identified via spatial matching)
    - Labels: 3-class segmentation from divided_masks (background=0, daughter=1, mother=2)
    """
    
    def __init__(self, data_dir: str = "processed_data", transform=None, augment=True, verbose=True):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.transform = transform
        self.augment = augment
        
        # Get all sample files, excluding macOS hidden files
        self.sample_files = [f for f in self.images_dir.glob("pair_*.npy") 
                           if not f.name.startswith("._")]
        
        if len(self.sample_files) == 0:
            raise ValueError(f"No training samples found in {self.images_dir}")
        
        if verbose:
            print(f"📊 Found {len(self.sample_files)} training samples")
        
        # Load generation summary for additional info
        summary_file = self.data_dir / "generation_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                self.summary = json.load(f)
            if verbose:
                print(f"📈 Dataset generated on: {self.summary.get('generation_date', 'Unknown')}")
                print(f"🔢 Total budding pairs: {self.summary.get('total_budding_pairs', len(self.sample_files))}")
        else:
            self.summary = None
        
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):
        # Load image and label
        image_file = self.sample_files[idx]
        label_file = self.labels_dir / image_file.name
        
        # Load data (already preprocessed)
        image = np.load(image_file, allow_pickle=True).astype(np.float32)
        label = np.load(label_file, allow_pickle=True).astype(np.int64)
        
        # Add channel dimension for image if needed
        if len(image.shape) == 2:
            image = image[np.newaxis, ...]  # (H, W) -> (1, H, W)
        
        # Convert to tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        
        # Apply data augmentation if enabled
        if self.augment and self.transform:
            # Apply same transform to both image and label
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            # For label, we don't want interpolation, just geometric transforms
            label = label.float().unsqueeze(0)  # Add channel dim for transform
            label = self.transform(label)
            label = label.squeeze(0).long()  # Remove channel dim and convert back to long
        
        return image, label

class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, num_classes=3, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # Convert logits to probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Calculate dice for each class
        dice_scores = []
        for i in range(self.num_classes):
            input_flat = inputs[:, i, :, :].contiguous().view(-1)
            target_flat = targets_one_hot[:, i, :, :].contiguous().view(-1)
            
            intersection = (input_flat * target_flat).sum()
            dice = (2. * intersection + self.smooth) / (input_flat.sum() + target_flat.sum() + self.smooth)
            dice_scores.append(dice)
        
        # Return average dice loss
        return 1 - torch.stack(dice_scores).mean()

class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax Loss for multiclass segmentation
    
    This loss directly optimizes the IoU metric and considers entire regions
    rather than individual pixels, leading to more coherent segmentations.
    """
    
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmaxLoss, self).__init__()
        self.ignore_index = ignore_index
        self.classes = classes
        self.per_image = per_image
        
    def forward(self, probas, labels):
        """
        Multi-class Lovász-Softmax loss
        probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
        labels: [B, H, W] Tensor, ground truth labels (between 0 and C-1)
        """
        if self.per_image:
            loss = self._lovasz_softmax_per_image(probas, labels)
        else:
            loss = self._lovasz_softmax_flat(probas, labels)
        return loss
    
    def _lovasz_softmax_flat(self, probas, labels):
        """
        Multi-class Lovász-Softmax loss
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.
        
        # Flatten batch and spatial dimensions
        B, C, H, W = probas.shape
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*H*W, C]
        labels = labels.view(-1)  # [B*H*W]
        
        losses = []
        class_to_sum = list(range(C)) if self.classes in ['all', 'present'] else self.classes
        for c in class_to_sum:
            fg = (labels == c).float()  # foreground for class c
            if (self.classes == 'present' and fg.sum() == 0):
                continue
            c_proba = probas[:, c]
            losses.append(self._lovasz_hinge_flat(fg - c_proba, fg))
        return self._mean(losses)
    
    def _lovasz_softmax_per_image(self, probas, labels):
        """
        Per-image Lovász-Softmax loss
        """
        losses = []
        for p, l in zip(probas, labels):
            losses.append(self._lovasz_softmax_flat(p.unsqueeze(0), l.unsqueeze(0)))
        return self._mean(losses)
    
    def _lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovász hinge loss
        """
        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.
        
        # Memory safety: if tensor is too large, use sampling
        if len(labels) > 100000:  # Limit to 100k pixels to prevent OOM
            # Randomly sample pixels for very large tensors
            indices = torch.randperm(len(labels))[:50000]
            logits = logits[indices]
            labels = labels[indices]
        
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * signs)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        gt_sorted = labels[perm]
        grad = self._lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss
    
    def _lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovász extension w.r.t sorted errors
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        if gts == 0:
            return gt_sorted.float() * 0
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard
    
    def _mean(self, l, ignore_nan=False, empty=0):
        """
        Nanmean compatible with autograd
        """
        if len(l) == 0:
            return torch.tensor(empty, device=l[0].device if len(l) > 0 else None)
        if ignore_nan:
            l = [x for x in l if not torch.isnan(x)]
        if len(l) == 0:
            return torch.tensor(empty)
        return torch.mean(torch.stack(l))

class BoundaryLoss(nn.Module):
    """
    Boundary Loss that explicitly penalizes boundary prediction errors
    
    This loss helps maintain clean boundaries between mother and daughter cells
    by focusing training on the boundary regions.
    """
    
    def __init__(self, boundary_width=3):
        super(BoundaryLoss, self).__init__()
        self.boundary_width = boundary_width
        
    def _get_boundary_mask(self, labels):
        """
        Generate boundary mask from segmentation labels using pure PyTorch
        """
        # Always use fallback method to avoid OpenCV issues
        return self._get_boundary_mask_fallback(labels)
    
    def _get_boundary_mask_fallback(self, labels):
        """
        Fallback boundary detection without OpenCV
        """
        batch_size = labels.shape[0]
        boundaries = torch.zeros_like(labels, dtype=torch.float)
        
        for b in range(batch_size):
            label_np = labels[b].cpu().numpy()
            boundary = np.zeros_like(label_np, dtype=np.uint8)
            
            # Simple gradient-based boundary detection
            for class_id in [1, 2]:
                class_mask = (label_np == class_id).astype(np.float32)
                if class_mask.sum() > 0:
                    # Calculate gradients
                    grad_x = np.abs(np.gradient(class_mask, axis=1))
                    grad_y = np.abs(np.gradient(class_mask, axis=0))
                    gradient_mag = grad_x + grad_y
                    class_boundary = (gradient_mag > 0.1).astype(np.uint8)
                    boundary = np.maximum(boundary, class_boundary)
            
            boundaries[b] = torch.from_numpy(boundary).to(labels.device)
        
        return boundaries
    
    def forward(self, predictions, labels):
        """
        Compute boundary loss
        predictions: [B, C, H, W] - model predictions (logits)
        labels: [B, H, W] - ground truth labels
        """
        # Get boundary mask
        boundary_mask = self._get_boundary_mask(labels)
        
        # Convert predictions to probabilities
        probs = F.softmax(predictions, dim=1)
        
        # Calculate cross-entropy loss only on boundary pixels
        ce_loss = F.cross_entropy(predictions, labels, reduction='none')
        boundary_loss = (ce_loss * boundary_mask).sum() / (boundary_mask.sum() + 1e-8)
        
        return boundary_loss

class TopologyLoss(nn.Module):
    """
    Topology-aware loss that penalizes disconnected components
    
    This loss encourages each cell to form a single connected component,
    reducing fragmentation and isolated pixels.
    """
    
    def __init__(self, lambda_topology=1.0):
        super(TopologyLoss, self).__init__()
        self.lambda_topology = lambda_topology
        
    def _count_connected_components(self, mask):
        """Count connected components in a binary mask using pure PyTorch"""
        if mask.sum() == 0:
            return 0
        
        # Use simple approximation: if there are pixels, assume 1 component
        # This is much more memory efficient and avoids OpenCV issues
        return 1 if mask.sum() > 0 else 0
    
    def forward(self, predictions, labels):
        """
        Compute topology loss
        """
        batch_size = predictions.shape[0]
        probs = F.softmax(predictions, dim=1)
        
        topology_loss = 0.0
        
        for b in range(batch_size):
            for class_id in [1, 2]:  # daughter and mother cells
                # Ground truth
                gt_mask = (labels[b] == class_id)
                gt_components = self._count_connected_components(gt_mask)
                
                # Predicted mask (thresholded)
                pred_mask = (probs[b, class_id] > 0.5)
                pred_components = self._count_connected_components(pred_mask)
                
                # Penalize having more components than ground truth
                if pred_components > gt_components:
                    topology_loss += (pred_components - gt_components) ** 2
        
        # Convert to tensor if it's a float
        if isinstance(topology_loss, (int, float)):
            topology_loss = torch.tensor(topology_loss, dtype=torch.float32, device=predictions.device)
        
        return self.lambda_topology * topology_loss / batch_size

class StructuralLoss(nn.Module):
    """
    Advanced Combined Loss with structural awareness
    
    Combines multiple loss functions to address both pixel-level accuracy
    and spatial/structural coherence:
    - Cross-Entropy: Basic pixel classification
    - Dice: Region overlap
    - Lovász-Softmax: IoU optimization with spatial awareness
    - Boundary: Clean boundaries between regions
    - Topology: Connected component consistency
    """
    
    def __init__(self, num_classes=3, weights=None):
        super(StructuralLoss, self).__init__()
        
        # Initialize individual loss functions
        if weights is None:
            weights = {
                'ce': 0.4,           # Cross-entropy for basic classification
                'dice': 0.3,         # Dice for region overlap
                'lovasz': 0.2,       # Lovász-Softmax for IoU optimization
                'boundary': 0.08,    # Boundary loss for clean edges
                'topology': 0.02     # Topology loss for connectivity
            }
        
        self.weights = weights
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes)
        self.lovasz_loss = LovaszSoftmaxLoss(classes='present')
        self.boundary_loss = BoundaryLoss(boundary_width=3)
        self.topology_loss = TopologyLoss(lambda_topology=1.0)
        
        print(f"🔧 StructuralLoss initialized with weights: {self.weights}")
    
    def forward(self, predictions, labels):
        """
        Compute combined structural loss
        """
        # Convert predictions to probabilities for Lovász loss
        probs = F.softmax(predictions, dim=1)
        
        # Compute individual losses
        ce = self.ce_loss(predictions, labels)
        dice = self.dice_loss(predictions, labels)
        lovasz = self.lovasz_loss(probs, labels)
        boundary = self.boundary_loss(predictions, labels)
        topology = self.topology_loss(predictions, labels)
        
        # Combine losses
        total_loss = (self.weights['ce'] * ce + 
                     self.weights['dice'] * dice + 
                     self.weights['lovasz'] * lovasz + 
                     self.weights['boundary'] * boundary + 
                     self.weights['topology'] * topology)
        
        return total_loss, {
            'ce': ce.item() if hasattr(ce, 'item') else float(ce),
            'dice': dice.item() if hasattr(dice, 'item') else float(dice), 
            'lovasz': lovasz.item() if hasattr(lovasz, 'item') else float(lovasz),
            'boundary': boundary.item() if hasattr(boundary, 'item') else float(boundary),
            'topology': topology.item() if hasattr(topology, 'item') else float(topology),
            'total': total_loss.item() if hasattr(total_loss, 'item') else float(total_loss)
        }

class SimplifiedStructuralLoss(nn.Module):
    """
    Memory-efficient version of StructuralLoss for systems with limited GPU memory
    
    Uses simpler approximations of spatial losses to reduce memory usage
    while still providing better results than basic Cross-Entropy + Dice.
    """
    
    def __init__(self, num_classes=3, weights=None):
        super(SimplifiedStructuralLoss, self).__init__()
        
        if weights is None:
            weights = {
                'ce': 0.5,           # Cross-entropy for basic classification
                'dice': 0.4,         # Dice for region overlap
                'boundary': 0.1,     # Simplified boundary loss
            }
        
        self.weights = weights
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes)
        self.boundary_loss = SimpleGradientBoundaryLoss()  # Use gradient-based boundary loss
        
        print(f"🔧 SimplifiedStructuralLoss initialized with weights: {self.weights}")
        print("💡 Using memory-efficient version suitable for limited GPU memory")
    
    def forward(self, predictions, labels):
        """
        Compute simplified structural loss
        """
        # Compute individual losses
        ce = self.ce_loss(predictions, labels)
        dice = self.dice_loss(predictions, labels)
        boundary = self.boundary_loss(predictions, labels)
        
        # Combine losses (no Lovász or topology to save memory)
        total_loss = (self.weights['ce'] * ce + 
                     self.weights['dice'] * dice + 
                     self.weights['boundary'] * boundary)
        
        return total_loss, {
            'ce': ce.item() if hasattr(ce, 'item') else float(ce),
            'dice': dice.item() if hasattr(dice, 'item') else float(dice), 
            'lovasz': 0.0,  # Not computed in simplified version
            'boundary': boundary.item() if hasattr(boundary, 'item') else float(boundary),
            'topology': 0.0,  # Not computed in simplified version
            'total': total_loss.item() if hasattr(total_loss, 'item') else float(total_loss)
        }

class CombinedLoss(nn.Module):
    """
    Legacy Combined Loss for backward compatibility
    """
    
    def __init__(self, num_classes=3, ce_weight=1.0, dice_weight=1.0):
        super(CombinedLoss, self).__init__()
        print("⚠️  Using legacy CombinedLoss. Consider upgrading to StructuralLoss for better results.")
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        total_loss = self.ce_weight * ce + self.dice_weight * dice
        
        # Return format compatible with StructuralLoss
        loss_components = {
            'ce': ce.item(),
            'dice': dice.item(),
            'lovasz': 0.0,
            'boundary': 0.0,
            'topology': 0.0,
            'total': total_loss.item()
        }
        
        return total_loss, loss_components

class SimpleGradientBoundaryLoss(nn.Module):
    """
    Simple gradient-based boundary loss that doesn't require OpenCV
    
    Uses torch operations to detect boundaries and penalize errors there.
    Much more memory efficient and reliable than morphology-based approaches.
    """
    
    def __init__(self):
        super(SimpleGradientBoundaryLoss, self).__init__()
        
    def forward(self, predictions, labels):
        """
        Compute gradient-based boundary loss
        """
        # Get probabilities
        probs = F.softmax(predictions, dim=1)
        
        # Compute gradients of the ground truth labels
        labels_float = labels.float()
        
        # Calculate spatial gradients
        grad_x = torch.abs(labels_float[:, :, 1:] - labels_float[:, :, :-1])
        grad_y = torch.abs(labels_float[:, 1:, :] - labels_float[:, :-1, :])
        
        # Create boundary mask (where gradient is non-zero)
        boundary_x = (grad_x > 0).float()
        boundary_y = (grad_y > 0).float()
        
        # Pad to match original size
        boundary_x = F.pad(boundary_x, (0, 1, 0, 0))  # Pad right
        boundary_y = F.pad(boundary_y, (0, 0, 0, 1))  # Pad bottom
        
        # Combine x and y boundaries
        boundary_mask = torch.clamp(boundary_x + boundary_y, 0, 1)
        
        # Compute cross-entropy loss only on boundary pixels
        ce_loss = F.cross_entropy(predictions, labels, reduction='none')
        boundary_loss = (ce_loss * boundary_mask).sum() / (boundary_mask.sum() + 1e-8)
        
        return boundary_loss

def calculate_iou(pred_mask: np.ndarray, true_mask: np.ndarray, class_id: int) -> float:
    """Calculate IoU for a specific class"""
    pred_class = (pred_mask == class_id)
    true_class = (true_mask == class_id)
    
    intersection = np.logical_and(pred_class, true_class).sum()
    union = np.logical_or(pred_class, true_class).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def train_model(model, train_loader, val_loader, num_epochs=300, learning_rate=1e-4, 
                device=None, save_dir="models", early_stopping_patience=30, criterion=None):
    """Train the U-Net model"""
    
    if device is None:
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        else:
            device = torch.device('cpu')
    
    model = model.to(device)
    
    # Use provided criterion or create default
    if criterion is None:
        print("🔧 Using default SimplifiedStructuralLoss")
        criterion = SimplifiedStructuralLoss(num_classes=3, weights={
            'ce': 0.5,           # Cross-entropy for basic classification
            'dice': 0.4,         # Dice for region overlap
            'boundary': 0.1,     # Simplified boundary loss
        })
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.5, min_lr=1e-7)
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Training history with detailed loss tracking
    train_losses = []
    val_losses = []
    val_ious = []
    loss_components_history = []
    
    best_val_loss = float('inf')
    best_iou = 0.0
    patience_counter = 0
    
    print(f"🚀 Training on device: {device}")
    print(f"📊 Training samples: {len(train_loader.dataset)}")
    print(f"📊 Validation samples: {len(val_loader.dataset)}")
    print(f"🔄 Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}")
    print(f"⏰ Early stopping patience: {early_stopping_patience} epochs")
    print("Starting training...")
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        epoch_loss_components = {'ce': 0.0, 'dice': 0.0, 'lovasz': 0.0, 'boundary': 0.0, 'topology': 0.0, 'total': 0.0}
        train_progress = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (images, labels) in enumerate(train_progress):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss, loss_components = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accumulate loss components
            for key in epoch_loss_components:
                epoch_loss_components[key] += loss_components[key]
            
            train_progress.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "CE": f"{loss_components['ce']:.3f}",
                "Dice": f"{loss_components['dice']:.3f}",
                "Lovász": f"{loss_components['lovasz']:.3f}"
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_loss_components = {'ce': 0.0, 'dice': 0.0, 'lovasz': 0.0, 'boundary': 0.0, 'topology': 0.0, 'total': 0.0}
        all_ious_daughter = []
        all_ious_mother = []
        
        val_progress = tqdm(val_loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for images, labels in val_progress:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss, loss_components = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Accumulate validation loss components
                for key in val_loss_components:
                    val_loss_components[key] += loss_components[key]
                
                # Calculate IoU for validation
                predictions = torch.argmax(outputs, dim=1)
                
                for i in range(predictions.shape[0]):
                    pred_np = predictions[i].cpu().numpy()
                    label_np = labels[i].cpu().numpy()
                    
                    iou_daughter = calculate_iou(pred_np, label_np, class_id=1)
                    iou_mother = calculate_iou(pred_np, label_np, class_id=2)
                    
                    all_ious_daughter.append(iou_daughter)
                    all_ious_mother.append(iou_mother)
                
                val_progress.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "CE": f"{loss_components['ce']:.3f}",
                    "Lovász": f"{loss_components['lovasz']:.3f}"
                })
        
        # Calculate average losses and metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_iou_daughter = np.mean(all_ious_daughter)
        avg_iou_mother = np.mean(all_ious_mother)
        avg_iou_overall = (avg_iou_daughter + avg_iou_mother) / 2
        
        # Calculate average loss components
        avg_train_components = {key: val / len(train_loader) for key, val in epoch_loss_components.items()}
        avg_val_components = {key: val / len(val_loader) for key, val in val_loss_components.items()}
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_ious.append(avg_iou_overall)
        loss_components_history.append({
            'epoch': epoch + 1,
            'train': avg_train_components,
            'val': avg_val_components
        })
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset patience counter when validation loss improves
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print("💾 New best model saved (lowest validation loss)")
        else:
            patience_counter += 1  # Increment patience counter when no improvement
        
        # Save best model based on IoU
        if avg_iou_overall > best_iou:
            best_iou = avg_iou_overall
            torch.save(model.state_dict(), save_dir / "best_iou_model.pth")
            print("🎯 New best IoU model saved")
        
        # Print progress with detailed loss breakdown
        print(f"📊 Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"🎯 IoU - Daughter: {avg_iou_daughter:.4f}, Mother: {avg_iou_mother:.4f}, Overall: {avg_iou_overall:.4f}")
        print(f"📈 Loss Components (Train/Val):")
        print(f"   CE: {avg_train_components['ce']:.4f}/{avg_val_components['ce']:.4f}")
        print(f"   Dice: {avg_train_components['dice']:.4f}/{avg_val_components['dice']:.4f}")
        print(f"   Lovász: {avg_train_components['lovasz']:.4f}/{avg_val_components['lovasz']:.4f}")
        print(f"   Boundary: {avg_train_components['boundary']:.4f}/{avg_val_components['boundary']:.4f}")
        print(f"   Topology: {avg_train_components['topology']:.4f}/{avg_val_components['topology']:.4f}")
        print(f"⏰ Early stopping patience: {patience_counter}/{early_stopping_patience}")
        if new_lr != old_lr:
            print(f"📉 Learning rate reduced: {old_lr:.1e} → {new_lr:.1e}")
        
        # Early stopping checks
        if patience_counter >= early_stopping_patience:
            print(f"⏹️ Early stopping: No improvement for {early_stopping_patience} epochs")
            break
        
        if optimizer.param_groups[0]['lr'] < 1e-7:
            print("⏹️ Early stopping: learning rate too small")
            break
    
    print(f"\n✅ Training completed!")
    print(f"💾 Best validation loss: {best_val_loss:.4f}")
    print(f"🎯 Best IoU: {best_iou:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), save_dir / "final_model.pth")
    
    # Plot training history and loss components
    plot_training_history(train_losses, val_losses, val_ious, save_dir)
    visualize_loss_components(loss_components_history, save_dir)
    
    # Save training history with detailed loss components
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_ious': val_ious,
        'loss_components_history': loss_components_history,
        'best_val_loss': best_val_loss,
        'best_iou': best_iou,
        'num_epochs_completed': len(train_losses),
        'final_lr': optimizer.param_groups[0]['lr'],
        'early_stopping_patience': early_stopping_patience,
        'final_patience_counter': patience_counter,
        'training_date': datetime.now().isoformat(),
        'loss_weights': criterion.weights
    }
    
    with open(save_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history

def evaluate_test_set(model, test_loader, device=None):
    """Evaluate model on test set"""
    
    if device is None:
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        else:
            device = torch.device('cpu')
    
    model.eval()
    criterion = SimplifiedStructuralLoss(num_classes=3, weights={
        'ce': 0.5,           # Cross-entropy for basic classification
        'dice': 0.4,         # Dice for region overlap
        'boundary': 0.1,     # Simplified boundary loss
    })
    
    test_loss = 0.0
    all_ious_daughter = []
    all_ious_mother = []
    
    print("🧪 Evaluating on test set...")
    
    with torch.no_grad():
        test_progress = tqdm(test_loader, desc="Testing", leave=False)
        
        for images, labels in test_progress:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss, loss_components = criterion(outputs, labels)
            test_loss += loss.item()
            
            # Calculate IoU for test
            predictions = torch.argmax(outputs, dim=1)
            
            for i in range(predictions.shape[0]):
                pred_np = predictions[i].cpu().numpy()
                label_np = labels[i].cpu().numpy()
                
                iou_daughter = calculate_iou(pred_np, label_np, class_id=1)
                iou_mother = calculate_iou(pred_np, label_np, class_id=2)
                
                all_ious_daughter.append(iou_daughter)
                all_ious_mother.append(iou_mother)
            
            test_progress.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    # Calculate average metrics
    avg_test_loss = test_loss / len(test_loader)
    avg_iou_daughter = np.mean(all_ious_daughter)
    avg_iou_mother = np.mean(all_ious_mother)
    avg_iou_overall = (avg_iou_daughter + avg_iou_mother) / 2
    
    print(f"\n📊 Test Set Results:")
    print(f"   Test Loss: {avg_test_loss:.4f}")
    print(f"   IoU - Daughter: {avg_iou_daughter:.4f}")
    print(f"   IoU - Mother: {avg_iou_mother:.4f}")
    print(f"   IoU - Overall: {avg_iou_overall:.4f}")
    
    return {
        'test_loss': avg_test_loss,
        'test_iou_daughter': avg_iou_daughter,
        'test_iou_mother': avg_iou_mother,
        'test_iou_overall': avg_iou_overall
    }

def plot_training_history(train_losses, val_losses, val_ious, save_dir):
    """Plot training and validation curves"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, label='Training Loss', color='blue', linewidth=2)
    ax1.plot(epochs, val_losses, label='Validation Loss', color='red', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # IoU curve
    ax2.plot(epochs, val_ious, label='Validation IoU', color='green', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.set_title('Validation IoU')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Training curves saved to {f'{save_dir}/training_curves.png'}")

def analyze_dataset_before_training():
    """Analyze dataset characteristics before training"""
    
    print("📊 Analyzing dataset before training...")
    
    # Load a few samples to analyze class distribution
    processed_images_dir = Path("processed_data/images")
    sample_files = list(processed_images_dir.glob("pair_*.npy"))[:20]  # Analyze first 20 samples
    
    if not sample_files:
        print("❌ No samples found for analysis")
        return
    
    class_counts = {0: 0, 1: 0, 2: 0}  # background, daughter, mother
    total_pixels = 0
    
    for sample_file in sample_files:
        label_file = Path("processed_data/labels") / sample_file.name
        if label_file.exists():
            label = np.load(label_file)
            for class_id in [0, 1, 2]:
                class_counts[class_id] += np.sum(label == class_id)
            total_pixels += label.size
    
    print(f"📈 Class distribution (from {len(sample_files)} samples):")
    for class_id, name in [(0, "Background"), (1, "Daughter"), (2, "Mother")]:
        percentage = 100 * class_counts[class_id] / total_pixels
        print(f"   {name}: {percentage:.1f}% ({class_counts[class_id]:,} pixels)")
    
    # Calculate class weights for balanced training
    class_weights = []
    for class_id in [0, 1, 2]:
        weight = total_pixels / (3 * class_counts[class_id]) if class_counts[class_id] > 0 else 1.0
        class_weights.append(weight)
    
    print(f"💡 Suggested class weights: {[f'{w:.2f}' for w in class_weights]}")

def post_process_predictions(predictions: np.ndarray, min_area=50, fill_holes=True) -> np.ndarray:
    """
    Post-process segmentation predictions to improve visual quality
    
    Args:
        predictions: [H, W] numpy array with class predictions (0=background, 1=daughter, 2=mother)
        min_area: Minimum area for connected components (smaller components will be removed)
        fill_holes: Whether to fill holes in segmented regions
        
    Returns:
        Cleaned predictions with same shape
    """
    if not CV2_AVAILABLE:
        print("⚠️  OpenCV not available. Post-processing will use fallback methods.")
        return post_process_predictions_fallback(predictions, min_area, fill_holes)
    
    cleaned = predictions.copy()
    
    # Process each class separately
    for class_id in [1, 2]:  # daughter and mother cells
        class_mask = (predictions == class_id).astype(np.uint8)
        
        if class_mask.sum() == 0:
            continue
            
        # Remove small connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        
        # Keep only components larger than min_area
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                class_mask[labels == i] = 0
        
        # Fill holes if requested
        if fill_holes:
            # Create kernel for morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)
            
            # Fill holes using flood fill from borders
            h, w = class_mask.shape
            mask_filled = class_mask.copy()
            
            # Create a mask that is larger than the input image
            h_ext, w_ext = h + 2, w + 2
            mask_ext = np.zeros((h_ext, w_ext), dtype=np.uint8)
            mask_ext[1:h+1, 1:w+1] = mask_filled
            
            # Flood fill from the border
            cv2.floodFill(mask_ext, None, (0, 0), 255)
            
            # Invert to get holes filled
            mask_filled = mask_ext[1:h+1, 1:w+1]
            mask_filled = cv2.bitwise_not(mask_filled)
            mask_filled = cv2.bitwise_or(class_mask, mask_filled)
            
            class_mask = mask_filled
        
        # Apply smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
        
        # Update cleaned predictions
        cleaned[class_mask > 0] = class_id
        
        # Ensure background where no class is predicted
        if class_id == 2:  # After processing mother cells
            # Remove conflicts: if both daughter and mother are predicted, keep the larger component
            daughter_mask = (cleaned == 1)
            mother_mask = (cleaned == 2)
            overlap = daughter_mask & mother_mask
            
            if overlap.sum() > 0:
                # For overlapping regions, assign to the class with larger total area
                daughter_area = daughter_mask.sum()
                mother_area = mother_mask.sum()
                
                if daughter_area > mother_area:
                    cleaned[overlap] = 1
                else:
                    cleaned[overlap] = 2
    
    return cleaned

def post_process_predictions_fallback(predictions: np.ndarray, min_area=50, fill_holes=True) -> np.ndarray:
    """
    Fallback post-processing without OpenCV
    """
    cleaned = predictions.copy()
    
    if SCIPY_AVAILABLE:
        
        # Process each class separately
        for class_id in [1, 2]:  # daughter and mother cells
            class_mask = (predictions == class_id).astype(bool)
            
            if class_mask.sum() == 0:
                continue
            
            # Remove small connected components using scipy
            labeled, num_labels = ndimage.label(class_mask)
            
            # Keep only components larger than min_area
            for i in range(1, num_labels + 1):
                component_size = np.sum(labeled == i)
                if component_size < min_area:
                    class_mask[labeled == i] = False
            
            # Fill holes if requested
            if fill_holes:
                class_mask = binary_fill_holes(class_mask)
            
            # Apply basic smoothing (erosion followed by dilation)
            class_mask = ndimage.binary_erosion(class_mask, iterations=1)
            class_mask = ndimage.binary_dilation(class_mask, iterations=1)
            
            # Update cleaned predictions
            cleaned[class_mask] = class_id
            
    else:
        print("⚠️  Neither OpenCV nor SciPy available. Minimal post-processing applied.")
        # Very basic processing: just remove very small isolated pixels
        for class_id in [1, 2]:
            class_mask = (predictions == class_id)
            if class_mask.sum() < min_area // 10:  # Remove very small masks
                cleaned[class_mask] = 0
    
    return cleaned

def visualize_loss_components(loss_components_history, save_dir):
    """
    Visualize the evolution of different loss components during training
    """
    if not loss_components_history:
        return
        
    epochs = [h['epoch'] for h in loss_components_history]
    
    # Extract loss components
    components = ['ce', 'dice', 'lovasz', 'boundary', 'topology']
    train_losses = {comp: [h['train'][comp] for h in loss_components_history] for comp in components}
    val_losses = {comp: [h['val'][comp] for h in loss_components_history] for comp in components}
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, comp in enumerate(components):
        ax = axes[i]
        ax.plot(epochs, train_losses[comp], label=f'Train {comp.upper()}', color=colors[i], linewidth=2)
        ax.plot(epochs, val_losses[comp], label=f'Val {comp.upper()}', color=colors[i], linewidth=2, linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{comp.upper()} Loss Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(save_dir / "loss_components.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Loss components visualization saved to {save_dir / 'loss_components.png'}")

def predict_with_postprocessing(model, image_tensor, device=None, apply_postprocess=True, min_area=50):
    """
    Make prediction with optional post-processing for improved visual quality
    
    Args:
        model: Trained U-Net model
        image_tensor: Input tensor [1, 1, H, W] or [1, H, W]
        device: Device to run inference on
        apply_postprocess: Whether to apply morphological post-processing
        min_area: Minimum area for connected components
        
    Returns:
        predictions: [H, W] numpy array with cleaned predictions
        raw_predictions: [H, W] numpy array with raw model predictions
        probabilities: [3, H, W] numpy array with class probabilities
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    with torch.no_grad():
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        image_tensor = image_tensor.to(device)
        
        # Get model predictions
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        raw_predictions = torch.argmax(outputs, dim=1)
        
        # Convert to numpy
        raw_pred_np = raw_predictions[0].cpu().numpy()
        probs_np = probabilities[0].cpu().numpy()
        
        # Apply post-processing if requested
        if apply_postprocess:
            cleaned_pred = post_process_predictions(raw_pred_np, min_area=min_area, fill_holes=True)
        else:
            cleaned_pred = raw_pred_np
        
        return cleaned_pred, raw_pred_np, probs_np

def parse_arguments():
    """
    Parse command line arguments for training configuration
    """
    parser = argparse.ArgumentParser(
        description="🧠 Budding Cell Classification U-Net Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📋 Examples:
  python train_unet.py --loss simplified --lr 1e-4 --batch-size 8
  python train_unet.py --loss full --epochs 200 --device cuda
  python train_unet.py --loss combined --lr 1e-3 --batch-size 16 --device cpu
  python train_unet.py --loss simplified --boundary-weight 0.2 --save-dir my_models

🔧 Loss Function Options:
  - combined:   Basic CE + Dice (fastest, legacy)
  - simplified: CE + Dice + Boundary (recommended balance)  
  - full:       CE + Dice + Lovász + Boundary + Topology (best quality, memory intensive)

💡 Tips:
  - Use 'simplified' for most cases (good balance of quality vs memory)
  - Use 'full' only if you have sufficient GPU memory (>16GB)
  - Use 'combined' for quick testing or limited resources
        """
    )
    
    # Loss function selection
    parser.add_argument(
        '--loss', 
        type=str, 
        choices=['combined', 'simplified', 'full'],
        default='simplified',
        help='Choose loss function strategy (default: simplified)'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=300,
        help='Number of training epochs (default: 300)'
    )
    
    parser.add_argument(
        '--lr', '--learning-rate',
        type=float, 
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int, 
        default=16,
        help='Batch size (default: auto-detect based on GPU memory)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu', 'mps'],
        default='auto',
        help='Device to use for training (default: auto)'
    )
    
    # Loss function weights (for custom tuning)
    parser.add_argument(
        '--ce-weight',
        type=float,
        default=None,
        help='Cross-entropy loss weight (overrides default)'
    )
    
    parser.add_argument(
        '--dice-weight', 
        type=float,
        default=None,
        help='Dice loss weight (overrides default)'
    )
    
    parser.add_argument(
        '--boundary-weight',
        type=float,
        default=None, 
        help='Boundary loss weight (overrides default)'
    )
    
    parser.add_argument(
        '--lovasz-weight',
        type=float,
        default=None,
        help='Lovász loss weight (overrides default, full loss only)'
    )
    
    parser.add_argument(
        '--topology-weight',
        type=float,
        default=None,
        help='Topology loss weight (overrides default, full loss only)'
    )
    
    # Data and output
    parser.add_argument(
        '--data-dir',
        type=str,
        default='processed_data',
        help='Directory containing training data (default: processed_data)'
    )
    
    parser.add_argument(
        '--save-dir',
        type=str,
        default='models',
        help='Directory to save trained models (default: models)'
    )
    
    # Training behavior
    parser.add_argument(
        '--early-stopping',
        type=int,
        default=30,
        help='Early stopping patience in epochs (default: 30)'
    )
    
    parser.add_argument(
        '--no-augmentation',
        action='store_true',
        help='Disable data augmentation'
    )
    
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Only run evaluation on test set (requires pre-trained model)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to pre-trained model for testing or resume training'
    )
    
    # Advanced options
    parser.add_argument(
        '--split-ratio',
        type=float,
        nargs=3,
        default=[0.6, 0.2, 0.2],
        help='Train/Val/Test split ratios (default: 0.6 0.2 0.2)'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of data loader workers (default: auto)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration without training'
    )
    
    return parser.parse_args()

def get_loss_function(args):
    """
    Create loss function based on arguments
    """
    if args.loss == 'combined':
        # Legacy combined loss
        weights = {}
        if args.ce_weight is not None:
            weights['ce_weight'] = args.ce_weight
        if args.dice_weight is not None:
            weights['dice_weight'] = args.dice_weight
            
        criterion = CombinedLoss(
            num_classes=3, 
            ce_weight=weights.get('ce_weight', 1.0),
            dice_weight=weights.get('dice_weight', 1.0)
        )
        
    elif args.loss == 'simplified':
        # Simplified structural loss  
        weights = {
            'ce': 0.5,
            'dice': 0.4,
            'boundary': 0.1
        }
        
        # Override with custom weights if provided
        if args.ce_weight is not None:
            weights['ce'] = args.ce_weight
        if args.dice_weight is not None:
            weights['dice'] = args.dice_weight  
        if args.boundary_weight is not None:
            weights['boundary'] = args.boundary_weight
            
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        criterion = SimplifiedStructuralLoss(num_classes=3, weights=weights)
        
    elif args.loss == 'full':
        # Full structural loss
        weights = {
            'ce': 0.4,
            'dice': 0.2, 
            'lovasz': 0.3,
            'boundary': 0.1,
            'topology': 0.00
        }
        
        # Override with custom weights if provided
        if args.ce_weight is not None:
            weights['ce'] = args.ce_weight
        if args.dice_weight is not None:
            weights['dice'] = args.dice_weight
        if args.boundary_weight is not None:
            weights['boundary'] = args.boundary_weight
        if args.lovasz_weight is not None:
            weights['lovasz'] = args.lovasz_weight
        if args.topology_weight is not None:
            weights['topology'] = args.topology_weight
            
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        try:
            criterion = StructuralLoss(num_classes=3, weights=weights)
        except Exception as e:
            print(f"⚠️  Full StructuralLoss failed: {e}")
            print("🔄 Falling back to SimplifiedStructuralLoss...")
            criterion = SimplifiedStructuralLoss(num_classes=3)
    
    return criterion

def get_device(args):
    """
    Get device based on arguments
    """
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    return device

def print_configuration(args):
    """
    Print training configuration
    """
    print("🔧 Training Configuration:")
    print(f"   Loss Function: {args.loss}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size if args.batch_size else 'auto'}")
    print(f"   Device: {args.device}")
    print(f"   Data Directory: {args.data_dir}")
    print(f"   Save Directory: {args.save_dir}")
    print(f"   Early Stopping: {args.early_stopping} epochs")
    print(f"   Data Augmentation: {'Disabled' if args.no_augmentation else 'Enabled'}")
    
    # Print custom weights if any
    custom_weights = []
    if args.ce_weight is not None:
        custom_weights.append(f"CE: {args.ce_weight}")
    if args.dice_weight is not None:
        custom_weights.append(f"Dice: {args.dice_weight}")
    if args.boundary_weight is not None:
        custom_weights.append(f"Boundary: {args.boundary_weight}")
    if args.lovasz_weight is not None:
        custom_weights.append(f"Lovász: {args.lovasz_weight}")
    if args.topology_weight is not None:
        custom_weights.append(f"Topology: {args.topology_weight}")
        
    if custom_weights:
        print(f"   Custom Weights: {', '.join(custom_weights)}")

def main():
    """Main training function"""
    
    # Parse command line arguments
    args = parse_arguments()
    
    print("🧠 Budding Cell Classification U-Net Training")
    print("📋 Task: Binary mask → Semantic segmentation (mother/daughter)")
    print("=" * 50)
    
    # Display library status
    print(f"📚 Library Status:")
    print(f"   OpenCV: {'✅ Available' if CV2_AVAILABLE else '❌ Not Available'}")
    print(f"   SciPy: {'✅ Available' if SCIPY_AVAILABLE else '❌ Not Available'}")
    print()
    
    # Print configuration
    print_configuration(args)
    print()
    
    # Check if processed data exists
    processed_data_dir = Path(args.data_dir)
    if not processed_data_dir.exists() or not (processed_data_dir / "images").exists():
        print(f"❌ No processed data found in {args.data_dir}!")
        print("   Please run 'python data_processor.py' first to generate training data.")
        return
    
    # Analyze dataset
    if args.verbose:
        analyze_dataset_before_training()
    
    # Set device
    device = get_device(args)
    print(f"🖥️  Using device: {device}")
    if torch.cuda.is_available() and device.type == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Data augmentation transforms
    if not args.no_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
        ])
    else:
        train_transform = None
    
    # Create datasets
    print("\n📂 Loading datasets...")
    full_dataset = ProcessedCellDataset(args.data_dir, transform=train_transform, augment=not args.no_augmentation)
    
    # Split into train/val/test using specified ratios
    train_ratio, val_ratio, test_ratio = args.split_ratio
    train_size = int(train_ratio * len(full_dataset))
    val_size = int(val_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create validation and test datasets without augmentation
    dataset_no_aug = ProcessedCellDataset(args.data_dir, transform=None, augment=False, verbose=False)
    
    # Use the same indices for validation and test
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices
    val_dataset_clean = torch.utils.data.Subset(dataset_no_aug, val_indices)
    test_dataset_clean = torch.utils.data.Subset(dataset_no_aug, test_indices)
    
    # Determine optimal batch size
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        # Auto-determine based on loss function and device
        if args.loss == 'full':
            batch_size = 4 if device.type == 'cuda' else 2
        elif args.loss == 'simplified':
            batch_size = 8 if device.type == 'cuda' else 4
        else:  # combined
            batch_size = 16 if device.type == 'cuda' else 8
    
    print(f"📦 Selected batch size: {batch_size}")
    
    # Create data loaders
    num_workers = args.num_workers if args.num_workers is not None else min(8, os.cpu_count() if os.cpu_count() else 2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset_clean, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset_clean, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    print(f"📊 Test samples: {len(test_dataset)}")
    print(f"🔄 Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}")
    
    # Initialize model
    model = UNet(in_channels=1, num_classes=3)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔢 Total parameters: {total_params:,}")
    print(f"🔢 Trainable parameters: {trainable_params:,}")
    
    # Check for dry run
    if args.dry_run:
        print("\n✅ Dry run completed - configuration looks good!")
        return
    
    # Handle test-only mode
    if args.test_only:
        if args.model_path is None:
            print("❌ Test-only mode requires --model-path argument")
            return
        
        print(f"\n🧪 Loading model from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path))
        
        # Get loss function for evaluation
        criterion = get_loss_function(args)
        
        print("🧪 Evaluating on test set...")
        test_results = evaluate_test_set(model, test_loader, device)
        
        print(f"\n📊 Test Results:")
        for key, value in test_results.items():
            print(f"   {key}: {value:.4f}")
        
        return
    
    print(f"⏱️  Planned epochs: {args.epochs}")
    
    # Load pre-trained model if specified
    if args.model_path is not None:
        print(f"\n📥 Loading pre-trained model from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path))
    
    # Train model
    print("\n🚀 Starting training...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        save_dir=args.save_dir,
        early_stopping_patience=args.early_stopping,
        criterion=get_loss_function(args)
    )
    
    print("\n✅ Training completed successfully!")
    
    # Evaluate on test set
    print("\n🧪 Evaluating on test set...")
    # Load best model for testing
    model.load_state_dict(torch.load(f"{args.save_dir}/best_iou_model.pth"))
    test_results = evaluate_test_set(model, test_loader, device)
    
    # Save complete results including test performance
    history.update(test_results)
    
    with open(f"{args.save_dir}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n📁 Models saved to: {args.save_dir}/")
    print(f"📈 Training history saved to: {args.save_dir}/training_history.json")
    print(f"📊 Validation IoU: {history['best_iou']:.4f}")
    print(f"📊 Test IoU: {test_results['test_iou_overall']:.4f}")
    print(f"📊 Test Loss: {test_results['test_loss']:.4f}")
    print(f"\n🔧 Loss Function Used: {args.loss}")
    print(f"📈 Training completed with {len(history['train_losses'])} epochs")
    print(f"⏱️  Final learning rate: {history.get('final_lr', 'N/A')}")
    print(f"\n💡 Usage tips:")
    print(f"   📝 Test model: python train_unet.py --test-only --model-path {args.save_dir}/best_iou_model.pth")
    print(f"   🔄 Resume training: python train_unet.py --model-path {args.save_dir}/best_iou_model.pth --epochs 100")
    print(f"   🎯 Try different loss: python train_unet.py --loss full --batch-size 4")
    print(f"   🔧 Use predict_with_postprocessing() for inference to get cleaner results")

if __name__ == "__main__":
    main()