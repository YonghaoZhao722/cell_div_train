import os
import re
import pandas as pd
import numpy as np
import tifffile
from pathlib import Path
import cv2
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
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

class CellDataset(Dataset):
    """Dataset class for cell division training data"""
    
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.transform = transform
        
        # Get all sample files, excluding macOS hidden files
        self.sample_files = [f for f in self.images_dir.glob("*.npy") if not f.name.startswith("._")]
        
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):
        # Load image and label
        image_file = self.sample_files[idx]
        label_file = self.labels_dir / image_file.name
        
        # Load data
        image = np.load(image_file, allow_pickle=True).astype(np.float32)
        label = np.load(label_file, allow_pickle=True).astype(np.long)
        
        # Add channel dimension for image
        if len(image.shape) == 2:
            image = image[np.newaxis, ...]  # (H, W) -> (1, H, W)
        
        # Convert to tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        
        if self.transform:
            # Apply same transform to both image and label
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            # For label, we don't want interpolation
            label = label.float().unsqueeze(0)  # Add channel dim for transform
            label = self.transform(label)
            label = label.squeeze(0).long()  # Remove channel dim and convert back to long
        
        return image, label

class CellDivisionPipeline:
    """Complete pipeline for cell division U-Net training and prediction"""
    
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.divided_masks_dir = self.data_root / "divided_masks"
        self.divided_outlines_dir = self.data_root / "divided_outlines" 
        self.processed_data_dir = Path("processed_data")
        self.processed_data_dir.mkdir(exist_ok=True)
        (self.processed_data_dir / "images").mkdir(exist_ok=True)
        (self.processed_data_dir / "labels").mkdir(exist_ok=True)
        
        self.master_df = None
        self.target_size = (256, 256)  # Fixed size for training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def stage1_data_parsing(self) -> pd.DataFrame:
        """
        Stage 1: Data Parsing and Structuring
        Parse TXT files and create a unified DataFrame
        """
        print("=== Stage 1: Data Parsing and Structuring ===")
        
        # Step 1: Get all TIF files from divided_masks directory
        tif_files = [f for f in self.divided_masks_dir.glob("*.tif") if not f.name.startswith("._")]
        print(f"Found {len(tif_files)} TIF files in divided_masks")
        
        all_data = []
        
        # Step 2: Process each corresponding TXT file
        for tif_file in tif_files:
            # Find corresponding TXT file
            # Remove "_shifted" suffix if present and change extension
            base_name = tif_file.stem.replace("_shifted", "")
            txt_file = self.divided_outlines_dir / f"{base_name}.txt"
            
            if txt_file.exists():
                print(f"Processing: {txt_file.name}")
                try:
                    # Parse TXT file to extract cell information
                    cell_data = self._parse_txt_file(txt_file, tif_file)
                    all_data.extend(cell_data)
                except Exception as e:
                    print(f"Error processing {txt_file}: {e}")
            else:
                print(f"Warning: No corresponding TXT file found for {tif_file.name}")
        
        # Step 3: Create master DataFrame
        self.master_df = pd.DataFrame(all_data)
        
        if len(self.master_df) > 0:
            print(f"\nCreated master DataFrame with {len(self.master_df)} entries")
            print(f"Columns: {list(self.master_df.columns)}")
            print(f"Cell types distribution:\n{self.master_df['cell_type'].value_counts()}")
            
            # Save master DataFrame
            self.master_df.to_csv(self.processed_data_dir / "master_data.csv", index=False)
            print(f"Saved master DataFrame to {self.processed_data_dir / 'master_data.csv'}")
        else:
            print("Warning: No data found!")
            
        return self.master_df
    
    def _parse_txt_file(self, txt_file: Path, tif_file: Path) -> List[Dict]:
        """Parse a single TXT file and extract cell information"""
        
        with open(txt_file, 'r') as f:
            content = f.read()
        
        # Extract all cells from the file
        cell_blocks = re.findall(r'CELL\s+Cell_(\d+).*?(?=CELL\s+Cell_\d+|Z_POS\s*$)', content, re.DOTALL)
        
        # Handle the last cell block separately
        last_cell_match = re.search(r'CELL\s+Cell_(\d+).*?Z_POS\s*$', content, re.DOTALL)
        if last_cell_match:
            last_cell_id = last_cell_match.group(1)
            if last_cell_id not in cell_blocks:
                cell_blocks.append(last_cell_id)
        
        extracted_data = []
        
        for cell_id in cell_blocks:
            try:
                full_cell_id = int(cell_id)
                
                # Parse according to user's rules
                budding_pair_id = full_cell_id // 100
                suffix = full_cell_id % 100
                
                # Determine cell type based on suffix
                if suffix == 1:
                    cell_type = 'daughter'
                elif suffix == 2:
                    cell_type = 'mother'
                else:
                    cell_type = 'normal'
                
                extracted_data.append({
                    'image_name': tif_file.name,
                    'image_path': str(tif_file),
                    'txt_file': str(txt_file),
                    'full_cell_id': full_cell_id,
                    'budding_pair_id': budding_pair_id,
                    'cell_type': cell_type,
                    'suffix': suffix
                })
                
            except ValueError as e:
                print(f"Warning: Could not parse cell ID '{cell_id}': {e}")
                continue
        
        return extracted_data
    
    def stage2_generate_training_pairs(self) -> int:
        """
        Stage 2: Generate Training Sample Pairs
        Create (X, y) pairs for model training
        """
        print("\n=== Stage 2: Generate Training Sample Pairs ===")
        
        if self.master_df is None:
            print("Error: Master DataFrame not found. Run stage1_data_parsing() first.")
            return 0
        
        # Step 1: Find budding pairs (pairs with both mother and daughter)
        budding_pairs = self._find_budding_pairs()
        print(f"Found {len(budding_pairs)} valid budding pairs")
        
        if len(budding_pairs) == 0:
            print("No valid budding pairs found!")
            return 0
        
        # Step 2: Generate training samples for each budding pair
        generated_samples = 0
        
        for pair_id, pair_info in budding_pairs.items():
            try:
                # Generate training sample for this pair
                if self._generate_single_training_sample(pair_id, pair_info):
                    generated_samples += 1
                    
            except Exception as e:
                print(f"Error processing budding pair {pair_id}: {e}")
                continue
        
        print(f"Successfully generated {generated_samples} training samples")
        return generated_samples
    
    def _find_budding_pairs(self) -> Dict:
        """Find valid budding pairs (those with both mother and daughter cells)"""
        
        budding_pairs = {}
        
        # Group by budding_pair_id
        grouped = self.master_df.groupby('budding_pair_id')
        
        for pair_id, group in grouped:
            # Check if this group has both mother and daughter
            cell_types = set(group['cell_type'].values)
            
            if 'mother' in cell_types and 'daughter' in cell_types:
                # Get mother and daughter info
                mother_info = group[group['cell_type'] == 'mother'].iloc[0]
                daughter_info = group[group['cell_type'] == 'daughter'].iloc[0]
                
                budding_pairs[pair_id] = {
                    'mother': mother_info,
                    'daughter': daughter_info
                }
        
        return budding_pairs
    
    def _generate_single_training_sample(self, pair_id: int, pair_info: Dict) -> bool:
        """Generate a single training sample from a budding pair"""
        
        mother_info = pair_info['mother']
        daughter_info = pair_info['daughter']
        
        # Check if both cells are from the same image
        if mother_info['image_path'] != daughter_info['image_path']:
            print(f"Warning: Mother and daughter from different images for pair {pair_id}")
            return False
        
        # Load the labeled mask image
        image_path = mother_info['image_path']
        try:
            labeled_mask = tifffile.imread(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return False
        
        # Create mother and daughter masks
        mother_id = mother_info['full_cell_id']
        daughter_id = daughter_info['full_cell_id']
        
        mother_mask = (labeled_mask == mother_id).astype(np.uint8)
        daughter_mask = (labeled_mask == daughter_id).astype(np.uint8)
        
        # Check if masks are non-empty
        if np.sum(mother_mask) == 0 or np.sum(daughter_mask) == 0:
            print(f"Warning: Empty mask found for pair {pair_id}")
            return False
        
        # Create input X (combined mask)
        X = ((mother_mask + daughter_mask) > 0).astype(np.uint8)
        
        # Create label y (three-class segmentation)
        y = np.zeros_like(labeled_mask, dtype=np.uint8)
        y[daughter_mask == 1] = 1  # daughter = 1
        y[mother_mask == 1] = 2    # mother = 2
        # background = 0 (already initialized)
        
        # Find bounding box and crop with padding
        coords = np.argwhere(X > 0)
        if len(coords) == 0:
            return False
            
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Add padding
        padding = 10
        y_min = max(0, y_min - padding)
        x_min = max(0, x_min - padding) 
        y_max = min(X.shape[0], y_max + padding)
        x_max = min(X.shape[1], x_max + padding)
        
        # Crop the regions
        X_cropped = X[y_min:y_max, x_min:x_max]
        y_cropped = y[y_min:y_max, x_min:x_max]
        
        # Resize to target size
        X_resized = cv2.resize(X_cropped, self.target_size, interpolation=cv2.INTER_NEAREST)
        y_resized = cv2.resize(y_cropped, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Save the processed data
        sample_name = f"pair_{pair_id}"
        
        np.save(self.processed_data_dir / "images" / f"{sample_name}.npy", X_resized)
        np.save(self.processed_data_dir / "labels" / f"{sample_name}.npy", y_resized)
        
        return True
    
    def stage3_model_training(self, epochs: int = 100, batch_size: int = 8, learning_rate: float = 1e-4) -> None:
        """
        Stage 3: Model Training
        Train the U-Net model with preprocessed data
        """
        print("\n=== Stage 3: Model Training ===")
        
        # Check if processed data exists
        if not (self.processed_data_dir / "images").exists():
            print("Error: No processed data found. Run stage2_generate_training_pairs() first.")
            return
        
        # Create dataset
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90)
        ])
        
        dataset = CellDataset(self.processed_data_dir, transform=transform)
        
        if len(dataset) == 0:
            print("Error: No training samples found!")
            return
        
        print(f"Found {len(dataset)} training samples")
        
        # Split into train and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        # Initialize model
        self.model = UNet(in_channels=1, num_classes=3).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        
        # Training history
        train_losses = []
        val_losses = []
        
        print(f"Training on device: {self.device}")
        print("Starting training...")
        
        best_val_loss = float('inf')
        
        for epoch in tqdm(range(epochs), desc="Training",total=epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), self.processed_data_dir / "best_model.pth")
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Plot training history
        self._plot_training_history(train_losses, val_losses)
        
        # Save final model
        torch.save(self.model.state_dict(), self.processed_data_dir / "final_model.pth")
        print(f"Model saved to {self.processed_data_dir}")
    
    def _plot_training_history(self, train_losses: List[float], val_losses: List[float]) -> None:
        """Plot training and validation loss curves"""
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.processed_data_dir / "training_history.png")
        plt.close()
        
        print(f"Training history plot saved to {self.processed_data_dir / 'training_history.png'}")
    
    def calculate_iou(self, pred_mask: np.ndarray, true_mask: np.ndarray, class_id: int) -> float:
        """Calculate IoU for a specific class"""
        pred_class = (pred_mask == class_id)
        true_class = (true_mask == class_id)
        
        intersection = np.logical_and(pred_class, true_class).sum()
        union = np.logical_or(pred_class, true_class).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def stage4_evaluation_and_prediction(self, test_size: float = 0.2) -> Dict:
        """
        Stage 4: Evaluation & Prediction
        Evaluate model performance and create prediction function
        """
        print("\n=== Stage 4: Evaluation & Prediction ===")
        
        # Load the best trained model
        if not (self.processed_data_dir / "best_model.pth").exists():
            print("Error: No trained model found. Run stage3_model_training() first.")
            return {}
        
        # Initialize model and load weights
        self.model = UNet(in_channels=1, num_classes=3).to(self.device)
        self.model.load_state_dict(torch.load(self.processed_data_dir / "best_model.pth"))
        self.model.eval()
        
        print("Model loaded successfully!")
        
        # Create test dataset
        dataset = CellDataset(self.processed_data_dir, transform=None)  # No augmentation for test
        
        if len(dataset) == 0:
            print("Error: No test data found!")
            return {}
        
        # Create test set
        test_size_samples = max(1, int(test_size * len(dataset)))
        train_size_samples = len(dataset) - test_size_samples
        _, test_dataset = torch.utils.data.random_split(dataset, [train_size_samples, test_size_samples])
        
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        print(f"Evaluating on {len(test_dataset)} test samples...")
        
        # Evaluation metrics
        ious_daughter = []
        ious_mother = []
        total_accuracy = []
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                # Convert to numpy for evaluation
                pred_np = predictions.cpu().numpy()[0]
                label_np = labels.cpu().numpy()[0]
                
                # Calculate IoU for each class
                iou_daughter = self.calculate_iou(pred_np, label_np, class_id=1)
                iou_mother = self.calculate_iou(pred_np, label_np, class_id=2)
                
                ious_daughter.append(iou_daughter)
                ious_mother.append(iou_mother)
                
                # Calculate pixel accuracy
                correct_pixels = (pred_np == label_np).sum()
                total_pixels = label_np.size
                accuracy = correct_pixels / total_pixels
                total_accuracy.append(accuracy)
                
                # Save sample predictions for visualization
                if i < 3:  # Save first 3 samples
                    self._save_prediction_visualization(images[0], labels[0], predictions[0], i)
        
        # Calculate final metrics
        results = {
            'mean_iou_daughter': np.mean(ious_daughter),
            'mean_iou_mother': np.mean(ious_mother),
            'mean_iou_overall': (np.mean(ious_daughter) + np.mean(ious_mother)) / 2,
            'mean_pixel_accuracy': np.mean(total_accuracy),
            'std_iou_daughter': np.std(ious_daughter),
            'std_iou_mother': np.std(ious_mother)
        }
        
        # Print results
        print("\n=== Evaluation Results ===")
        print(f"Mean IoU (Daughter): {results['mean_iou_daughter']:.4f} ± {results['std_iou_daughter']:.4f}")
        print(f"Mean IoU (Mother): {results['mean_iou_mother']:.4f} ± {results['std_iou_mother']:.4f}")
        print(f"Mean IoU (Overall): {results['mean_iou_overall']:.4f}")
        print(f"Mean Pixel Accuracy: {results['mean_pixel_accuracy']:.4f}")
        
        # Save results
        import json
        with open(self.processed_data_dir / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation results saved to {self.processed_data_dir / 'evaluation_results.json'}")
        
        return results
    
    def _save_prediction_visualization(self, image: torch.Tensor, label: torch.Tensor, 
                                     prediction: torch.Tensor, sample_idx: int) -> None:
        """Save visualization of prediction results"""
        
        # Convert to numpy
        image_np = image.cpu().numpy()[0]  # Remove channel dim
        label_np = label.cpu().numpy()
        pred_np = prediction.cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Input image
        axes[0].imshow(image_np, cmap='gray')
        axes[0].set_title('Input')
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(label_np, cmap='viridis', vmin=0, vmax=2)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(pred_np, cmap='viridis', vmin=0, vmax=2)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.processed_data_dir / f"prediction_sample_{sample_idx}.png", dpi=150)
        plt.close()
    
    def predict_cell_division(self, input_mask: np.ndarray, 
                            original_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Prediction function for new cell division masks
        
        Args:
            input_mask: Binary mask of budding cell (mother + daughter combined)
            original_shape: If provided, resize prediction back to this shape
            
        Returns:
            Predicted segmentation mask (0=background, 1=daughter, 2=mother)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Run stage3_model_training() or load a trained model first.")
        
        # Preprocess input
        # Find bounding box
        coords = np.argwhere(input_mask > 0)
        if len(coords) == 0:
            return np.zeros_like(input_mask)
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Add padding
        padding = 10
        y_min = max(0, y_min - padding)
        x_min = max(0, x_min - padding)
        y_max = min(input_mask.shape[0], y_max + padding)
        x_max = min(input_mask.shape[1], x_max + padding)
        
        # Crop and resize
        cropped = input_mask[y_min:y_max, x_min:x_max]
        resized = cv2.resize(cropped.astype(np.uint8), self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Convert to tensor
        input_tensor = torch.from_numpy(resized.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        input_tensor = input_tensor.to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = torch.argmax(output, dim=1).cpu().numpy()[0]
        
        # Resize prediction back to cropped size
        pred_cropped_size = cv2.resize(prediction.astype(np.uint8), 
                                     (cropped.shape[1], cropped.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
        
        # Place back in original image
        result = np.zeros_like(input_mask, dtype=np.uint8)
        result[y_min:y_max, x_min:x_max] = pred_cropped_size
        
        # If original shape provided, resize to that
        if original_shape is not None and original_shape != result.shape:
            result = cv2.resize(result, (original_shape[1], original_shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
        
        return result
    
    def load_trained_model(self, model_path: str) -> None:
        """Load a previously trained model"""
        self.model = UNet(in_channels=1, num_classes=3).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")
    
    def save_model(self, save_path: str) -> None:
        """Save the current model"""
        if self.model is None:
            print("Error: No model to save")
            return
        
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = CellDivisionPipeline()
    
    # Stage 1: Data parsing
    master_df = pipeline.stage1_data_parsing()
    
    # Stage 2: Generate training pairs
    if master_df is not None and len(master_df) > 0:
        num_samples = pipeline.stage2_generate_training_pairs()
        print(f"\nGenerated {num_samples} training samples")
        
        # Stage 3: Model training (if we have training samples)
        if num_samples > 0:
            pipeline.stage3_model_training(epochs=50, batch_size=4, learning_rate=1e-4)
            
            # Stage 4: Evaluation
            results = pipeline.stage4_evaluation_and_prediction()
            
            # Example prediction on a new mask
            if results:
                print("\n=== Testing Prediction Function ===")
                # Load a sample for testing
                sample_files = list(pipeline.processed_data_dir.glob("images/*.npy"))
                if sample_files:
                    test_input = np.load(sample_files[0])
                    prediction = pipeline.predict_cell_division(test_input)
                    print(f"Prediction shape: {prediction.shape}")
                    print(f"Unique values in prediction: {np.unique(prediction)}")
    else:
        print("No data to process!") 