import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import argparse

# Import necessary classes from the training file
from train_unet import UNet, ProcessedCellDataset, get_loss_function, calculate_iou


def load_trained_model(model_path: str, device=None):
    """Load a pre-trained U-Net model"""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = UNet(in_channels=1, num_classes=3)
    
    # Load trained weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from: {model_path}")
    print(f"🖥️  Using device: {device}")
    
    return model, device


def create_test_dataset(data_dir: str = "processed_data", test_ratio: float = 0.2):
    """Create test dataset using the same split as training"""
    
    # Load full dataset without augmentation
    full_dataset = ProcessedCellDataset(data_dir, transform=None, augment=False)
    
    # Split dataset (same as training: 60%/20%/20%)
    train_size = int(0.6 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    # Use same random split as training
    torch.manual_seed(42)  # Set seed for reproducible splits
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    print(f"📊 Dataset sizes:")
    print(f"   Total: {len(full_dataset)}")
    print(f"   Train: {len(train_dataset)}")
    print(f"   Validation: {len(val_dataset)}")
    print(f"   Test: {len(test_dataset)}")
    
    return test_dataset, full_dataset


def evaluate_model(model, test_loader, device, loss_type='combined', verbose=True):
    """Evaluate model performance on test data"""
    
    model.eval()
    
    # Create a mock args object for get_loss_function
    class MockArgs:
        def __init__(self, loss_type):
            self.loss = loss_type
            self.ce_weight = None
            self.dice_weight = None
            self.boundary_weight = None
            self.lovasz_weight = None
            self.topology_weight = None
    
    mock_args = MockArgs(loss_type)
    criterion = get_loss_function(mock_args)
    
    test_loss = 0.0
    all_ious_daughter = []
    all_ious_mother = []
    all_ious_background = []
    
    if verbose:
        print(f"🧪 Evaluating model on test set using {loss_type} loss...")
    
    with torch.no_grad():
        test_progress = tqdm(test_loader, desc="Testing", disable=not verbose)
        
        for images, labels in test_progress:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss_result = criterion(outputs, labels)
            
            # Handle tuple return from loss function
            if isinstance(loss_result, tuple):
                loss = loss_result[0]
            else:
                loss = loss_result
                
            test_loss += loss.item()
            
            # Calculate IoU for each class
            predictions = torch.argmax(outputs, dim=1)
            
            for i in range(predictions.shape[0]):
                pred_np = predictions[i].cpu().numpy()
                label_np = labels[i].cpu().numpy()
                
                iou_background = calculate_iou(pred_np, label_np, class_id=0)
                iou_daughter = calculate_iou(pred_np, label_np, class_id=1)
                iou_mother = calculate_iou(pred_np, label_np, class_id=2)
                
                all_ious_background.append(iou_background)
                all_ious_daughter.append(iou_daughter)
                all_ious_mother.append(iou_mother)
            
            if verbose:
                test_progress.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    # Calculate average metrics
    avg_test_loss = test_loss / len(test_loader)
    avg_iou_background = np.mean(all_ious_background)
    avg_iou_daughter = np.mean(all_ious_daughter)
    avg_iou_mother = np.mean(all_ious_mother)
    avg_iou_overall = (avg_iou_daughter + avg_iou_mother) / 2  # Overall for cell classes only
    avg_iou_all = (avg_iou_background + avg_iou_daughter + avg_iou_mother) / 3  # All classes
    
    results = {
        'test_loss': avg_test_loss,
        'iou_background': avg_iou_background,
        'iou_daughter': avg_iou_daughter,
        'iou_mother': avg_iou_mother,
        'iou_overall_cells': avg_iou_overall,
        'iou_all_classes': avg_iou_all,
        'num_samples': len(test_loader.dataset),
        'loss_type': loss_type
    }
    
    if verbose:
        print(f"\n📊 Test Results:")
        print(f"   Loss Type: {loss_type}")
        print(f"   Test Loss: {avg_test_loss:.4f}")
        print(f"   IoU - Background: {avg_iou_background:.4f}")
        print(f"   IoU - Daughter: {avg_iou_daughter:.4f}")
        print(f"   IoU - Mother: {avg_iou_mother:.4f}")
        print(f"   IoU - Cell Classes: {avg_iou_overall:.4f}")
        print(f"   IoU - All Classes: {avg_iou_all:.4f}")
        print(f"   Tested on {len(test_loader.dataset)} samples")
    
    return results


def visualize_predictions(model, test_dataset, device, num_samples=5, save_path='test_predictions.png'):
    """Visualize model predictions on test samples"""
    
    model.eval()
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Class colors for visualization
    colors = {0: [0, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0]}  # Background=black, Daughter=red, Mother=green
    
    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            # Get sample
            image, label = test_dataset[i]
            image_batch = image.unsqueeze(0).to(device)
            
            # Get prediction
            output = model(image_batch)
            prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Convert to numpy for visualization
            image_np = image.squeeze().cpu().numpy()
            label_np = label.cpu().numpy()
            
            # Create colored masks
            label_colored = np.zeros((*label_np.shape, 3))
            pred_colored = np.zeros((*prediction.shape, 3))
            
            for class_id, color in colors.items():
                label_colored[label_np == class_id] = color
                pred_colored[prediction == class_id] = color
            
            # Plot
            axes[i, 0].imshow(image_np, cmap='gray')
            axes[i, 0].set_title(f'Input Image {i+1}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(label_colored)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_colored)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
            
            # Overlay
            overlay = image_np.copy()
            overlay = np.stack([overlay] * 3, axis=-1)
            overlay = np.clip(overlay + 0.3 * pred_colored, 0, 1)
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('Overlay')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📸 Visualization saved to: {save_path}")
    
    #plt.show()


def compare_models(model_paths: list, test_loader, device, loss_type='combined'):
    """Compare multiple trained models"""
    
    print(f"🔍 Comparing {len(model_paths)} models using {loss_type} loss...")
    
    results = {}
    
    for model_path in model_paths:
        model_name = Path(model_path).stem
        print(f"\n📊 Testing: {model_name}")
        
        try:
            model, _ = load_trained_model(model_path, device)
            result = evaluate_model(model, test_loader, device, loss_type, verbose=False)
            results[model_name] = result
            
            print(f"   IoU (Cells): {result['iou_overall_cells']:.4f}")
            print(f"   Test Loss: {result['test_loss']:.4f}")
            
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            continue
    
    # Print comparison table
    if results:
        print(f"\n📊 Model Comparison Summary (Loss: {loss_type}):")
        print(f"{'Model':<20} {'IoU (Cells)':<12} {'IoU (Daughter)':<14} {'IoU (Mother)':<12} {'Test Loss':<10}")
        print("-" * 70)
        
        for model_name, result in results.items():
            print(f"{model_name:<20} {result['iou_overall_cells']:<12.4f} "
                  f"{result['iou_daughter']:<14.4f} {result['iou_mother']:<12.4f} "
                  f"{result['test_loss']:<10.4f}")
    
    return results


def main():
    """Main testing function"""
    
    parser = argparse.ArgumentParser(description='Test trained U-Net models')
    parser.add_argument('--model', type=str, default='models_full/best_iou_model.pth',
                       help='Path to trained model file')
    parser.add_argument('--data_dir', type=str, default='processed_data',
                       help='Directory containing processed data')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for testing')
    parser.add_argument('--loss', type=str, default='combined',
                       choices=['combined', 'simplified', 'full'],
                       help='Loss function type: combined (legacy), simplified, or full')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of predictions')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all models in models directory')
    parser.add_argument('--save_results', type=str, default=None,
                       help='Path to save test results JSON')
    
    args = parser.parse_args()
    
    print("🧪 U-Net Model Testing")
    print("=" * 40)
    print(f"🎯 Using loss function: {args.loss}")
    
    # Check if data exists
    if not Path(args.data_dir).exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        print("   Please ensure processed data exists.")
        return
    
    # Set device
    if torch.backends.mps.is_available() or torch.cuda.is_available():
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda')
    else:
        device = torch.device('cpu')
    
    # Create test dataset
    test_dataset, full_dataset = create_test_dataset(args.data_dir)
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=min(4, os.cpu_count() if os.cpu_count() else 2),
        pin_memory=torch.cuda.is_available()
    )
    
    if args.compare:
        # Compare all models in models directory
        models_dir = Path("models_full")
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pth"))
            if model_files:
                results = compare_models([str(f) for f in model_files], test_loader, device, args.loss)
                
                if args.save_results:
                    with open(args.save_results, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"📁 Results saved to: {args.save_results}")
            else:
                print("❌ No model files found in models_full directory")
        else:
            print("❌ Models_full directory not found")
    
    else:
        # Test single model
        if not Path(args.model).exists():
            print(f"❌ Model file not found: {args.model}")
            print("   Available models:")
            models_dir = Path("models_full")
            if models_dir.exists():
                for model_file in models_dir.glob("*.pth"):
                    print(f"   - {model_file}")
            return
        
        # Load and test model
        model, device = load_trained_model(args.model, device)
        results = evaluate_model(model, test_loader, device, args.loss)
        
        # Save results if requested
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"📁 Results saved to: {args.save_results}")
        
        # Create visualizations if requested
        if args.visualize:
            print("\n📸 Creating visualizations...")
            visualize_predictions(
                model, test_dataset, device, 
                num_samples=5, 
                save_path="test_predictions.png"
            )
    
    print("\n✅ Testing completed!")


if __name__ == "__main__":
    main() 