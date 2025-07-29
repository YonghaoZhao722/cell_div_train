import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import random
from typing import List, Tuple, Dict
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from datetime import datetime

# Import model architecture
from cell_classification_model import LightweightCNN, CellClassificationDataset

def load_trained_model(model_path: str, device: str = 'auto') -> torch.nn.Module:
    """Load trained classification model"""
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # Initialize model
    model = LightweightCNN(input_size=256, num_classes=2, dropout=0.3)
    
    # Load weights
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully: {model_path}")
    print(f"🖥️  Device: {device}")
    
    return model, device

def predict_on_dataset(model: torch.nn.Module, dataset: CellClassificationDataset, 
                      device: torch.device) -> Tuple[List, List, List]:
    """Make predictions on dataset"""
    
    print(f"🔮 Making predictions on {len(dataset)} samples...")
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            image, label = dataset[i]
            
            # Add batch dimension
            image = image.unsqueeze(0).to(device)
            
            # Predict
            outputs = model(image)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1)
            
            all_predictions.append(prediction.cpu().numpy()[0])
            all_labels.append(label.item())
            all_probabilities.append(probabilities.cpu().numpy()[0])
    
    return all_predictions, all_labels, all_probabilities

def plot_confusion_matrix_detailed(y_true: List, y_pred: List, save_path: str = None):
    """Plot detailed confusion matrix"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal (0)', 'Budding (1)'], 
                yticklabels=['Normal (0)', 'Budding (1)'],
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add accuracy information
    total = np.sum(cm)
    accuracy = np.trace(cm) / total
    plt.figtext(0.15, 0.02, f'Overall Accuracy: {accuracy:.4f} ({np.trace(cm)}/{total})', 
                fontsize=10, ha='left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_probability_distribution(probabilities: List, labels: List, save_path: str = None):
    """Plot prediction probability distribution"""
    
    probabilities = np.array(probabilities)
    labels = np.array(labels)
    
    # Get budding class probabilities
    budding_probs = probabilities[:, 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: probability distribution by class
    normal_probs = budding_probs[labels == 0]
    budding_probs_true = budding_probs[labels == 1]
    
    ax1.hist(normal_probs, bins=30, alpha=0.7, label='Normal cells', color='skyblue', density=True)
    ax1.hist(budding_probs_true, bins=30, alpha=0.7, label='Budding cells', color='salmon', density=True)
    ax1.axvline(x=0.5, color='red', linestyle='--', label='Decision boundary (0.5)')
    ax1.set_xlabel('Budding Probability')
    ax1.set_ylabel('Density')
    ax1.set_title('Prediction Probability Distribution by Class')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: confidence analysis
    confidence = np.maximum(budding_probs, 1 - budding_probs)
    
    ax2.hist(confidence[labels == 0], bins=30, alpha=0.7, label='Normal cells', color='skyblue', density=True)
    ax2.hist(confidence[labels == 1], bins=30, alpha=0.7, label='Budding cells', color='salmon', density=True)
    ax2.set_xlabel('Prediction Confidence')
    ax2.set_ylabel('Density')
    ax2.set_title('Model Prediction Confidence Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Probability distribution plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_sample_predictions(model: torch.nn.Module, dataset: CellClassificationDataset, 
                                device: torch.device, num_samples: int = 16, 
                                save_path: str = None, sample_type: str = 'all'):
    """Visualize sample prediction results"""
    
    print(f"🎯 Visualizing {num_samples} sample prediction results...")
    
    # Get samples
    all_indices = list(range(len(dataset)))
    
    if sample_type == 'correct':
        # Show only correctly predicted samples
        selected_indices = []
        with torch.no_grad():
            for i in all_indices:
                image, label = dataset[i]
                image_batch = image.unsqueeze(0).to(device)
                outputs = model(image_batch)
                prediction = torch.argmax(outputs, dim=1).item()
                if prediction == label:
                    selected_indices.append(i)
    elif sample_type == 'incorrect':
        # Show only incorrectly predicted samples
        selected_indices = []
        with torch.no_grad():
            for i in all_indices:
                image, label = dataset[i]
                image_batch = image.unsqueeze(0).to(device)
                outputs = model(image_batch)
                prediction = torch.argmax(outputs, dim=1).item()
                if prediction != label:
                    selected_indices.append(i)
    else:
        # Show all samples
        selected_indices = all_indices
    
    if len(selected_indices) == 0:
        print(f"⚠️  No samples found for type '{sample_type}'")
        return
    
    # Randomly select samples
    num_to_show = min(num_samples, len(selected_indices))
    random_indices = random.sample(selected_indices, num_to_show)
    
    # Calculate grid size
    cols = int(np.ceil(np.sqrt(num_to_show)))
    rows = int(np.ceil(num_to_show / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if num_to_show == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Predict and visualize each sample
    with torch.no_grad():
        for i, idx in enumerate(random_indices):
            image, true_label = dataset[idx]
            
            # Predict
            image_batch = image.unsqueeze(0).to(device)
            outputs = model(image_batch)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0, prediction].item()
            
            # Display image
            ax = axes[i]
            image_np = image.numpy()[0]  # Remove channel dimension
            ax.imshow(image_np, cmap='gray')
            
            # Set title
            true_label_text = "Budding" if true_label == 1 else "Normal"
            pred_label_text = "Budding" if prediction == 1 else "Normal"
            
            is_correct = prediction == true_label
            title_color = 'green' if is_correct else 'red'
            
            title = f"True: {true_label_text}\nPred: {pred_label_text}\nConf: {confidence:.3f}"
            ax.set_title(title, fontsize=10, color=title_color)
            ax.axis('off')
            
            # Add border
            for spine in ax.spines.values():
                spine.set_edgecolor(title_color)
                spine.set_linewidth(2)
    
    # Hide extra subplots
    for i in range(num_to_show, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Sample Prediction Results - {sample_type.capitalize()}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🖼️  Sample visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def analyze_model_performance(y_true: List, y_pred: List, probabilities: List) -> Dict:
    """Analyze model performance"""
    
    print("\n📊 Model Performance Analysis:")
    print("=" * 50)
    
    # Basic metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision - Normal/Budding: {precision[0]:.4f} / {precision[1]:.4f}")
    print(f"Recall - Normal/Budding: {recall[0]:.4f} / {recall[1]:.4f}")
    print(f"F1 Score - Normal/Budding: {f1[0]:.4f} / {f1[1]:.4f}")
    
    # Confusion matrix analysis
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  Pred\\True     Normal  Budding")
    print(f"  Normal       {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"  Budding      {cm[1,0]:6d}   {cm[1,1]:6d}")
    
    # Confidence analysis
    probabilities = np.array(probabilities)
    budding_probs = probabilities[:, 1]
    confidence = np.maximum(budding_probs, 1 - budding_probs)
    
    print(f"\nConfidence Statistics:")
    print(f"  Mean confidence: {np.mean(confidence):.4f}")
    print(f"  Confidence std: {np.std(confidence):.4f}")
    print(f"  High confidence samples (>0.9): {np.sum(confidence > 0.9)} ({np.sum(confidence > 0.9)/len(confidence)*100:.1f}%)")
    print(f"  Low confidence samples (<0.6): {np.sum(confidence < 0.6)} ({np.sum(confidence < 0.6)/len(confidence)*100:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1_score': f1.tolist(),
        'confusion_matrix': cm.tolist(),
        'mean_confidence': float(np.mean(confidence)),
        'confidence_std': float(np.std(confidence))
    }

class ManualVerificationGUI:
    """Manual verification interface"""
    
    def __init__(self, model, dataset, device, incorrect_indices, data_dir):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.incorrect_indices = incorrect_indices
        self.current_idx = 0
        self.corrections = {}  # {index: user_label} User selected labels
        self.data_dir = Path(data_dir)
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Manual Cell Classification - Direct Labeling")
        self.root.geometry("800x900")
        
        self.setup_ui()
        self.show_current_sample()
    
    def setup_ui(self):
        """Setup user interface"""
        
        # Top information
        info_frame = ttk.Frame(self.root)
        info_frame.pack(pady=10)
        
        self.progress_label = ttk.Label(info_frame, font=('Arial', 12, 'bold'))
        self.progress_label.pack()
        
        # Image display area
        self.image_label = ttk.Label(self.root)
        self.image_label.pack(pady=20)
        
        # Prediction information
        info_text_frame = ttk.Frame(self.root)
        info_text_frame.pack(pady=10)
        
        self.info_text = tk.Text(info_text_frame, height=6, width=60, font=('Arial', 10))
        self.info_text.pack()
        
        # Question and selection
        question_frame = ttk.Frame(self.root)
        question_frame.pack(pady=20)
        
        question_label = ttk.Label(question_frame, 
                                 text="What type of cell is this?",
                                 font=('Arial', 14, 'bold'))
        question_label.pack(pady=10)
        
        # Selection buttons
        button_frame = ttk.Frame(question_frame)
        button_frame.pack(pady=15)
        
        self.normal_btn = ttk.Button(button_frame, text="🔵 NORMAL CELL", 
                                   command=self.select_normal, style='Normal.TButton')
        self.normal_btn.pack(side=tk.LEFT, padx=30)
        
        self.budding_btn = ttk.Button(button_frame, text="🟠 BUDDING CELL", 
                                    command=self.select_budding, style='Budding.TButton')
        self.budding_btn.pack(side=tk.LEFT, padx=30)
        
        # Navigation buttons
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(pady=20)
        
        self.prev_btn = ttk.Button(nav_frame, text="← Previous", command=self.prev_sample)
        self.prev_btn.pack(side=tk.LEFT, padx=10)
        
        self.next_btn = ttk.Button(nav_frame, text="Next →", command=self.next_sample)
        self.next_btn.pack(side=tk.LEFT, padx=10)
        
        self.finish_btn = ttk.Button(nav_frame, text="🏁 Finish Verification", 
                                   command=self.finish_verification)
        self.finish_btn.pack(side=tk.LEFT, padx=20)
        
        # Setup styles
        style = ttk.Style()
        style.configure('Normal.TButton', foreground='blue')
        style.configure('Budding.TButton', foreground='orange')
        style.configure('Selected.TButton', background='lightgray')
    
    def show_current_sample(self):
        """Display current sample"""
        if self.current_idx >= len(self.incorrect_indices):
            self.finish_verification()
            return
        
        idx = self.incorrect_indices[self.current_idx]
        image, true_label = self.dataset[idx]
        
        # Get model prediction
        with torch.no_grad():
            image_batch = image.unsqueeze(0).to(self.device)
            outputs = self.model(image_batch)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0, prediction].item()
        
        # Update progress
        progress_text = f"Sample {self.current_idx + 1} of {len(self.incorrect_indices)}"
        self.progress_label.config(text=progress_text)
        
        # Display image
        image_np = image.numpy()[0]  # Remove channel dimension
        # Scale image to appropriate size for display
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        image_pil = image_pil.resize((400, 400), Image.Resampling.NEAREST)
        photo = ImageTk.PhotoImage(image_pil)
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep reference
        
        # Display prediction information
        true_label_text = "Budding" if true_label == 1 else "Normal"
        pred_label_text = "Budding" if prediction == 1 else "Normal"
        
        info_text = f"""Original Label: {true_label_text}
Model Prediction: {pred_label_text}
Model Confidence: {confidence:.4f}

The original dataset labeled this cell as '{true_label_text}'.
The model predicted it as '{pred_label_text}'.

Look at the cell image above and choose what you think this cell actually is.
Your choice will be used to correct any labeling errors in the training data."""
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
        
        # Update button states
        self.prev_btn.config(state=tk.NORMAL if self.current_idx > 0 else tk.DISABLED)
        
        # If already marked, highlight corresponding button
        if idx in self.corrections:
            user_choice = self.corrections[idx]
            if user_choice == 0:  # Normal
                self.normal_btn.config(text="🔵 NORMAL CELL (Selected)", style='Selected.TButton')
                self.budding_btn.config(text="🟠 BUDDING CELL", style='Budding.TButton')
            else:  # Budding
                self.normal_btn.config(text="🔵 NORMAL CELL", style='Normal.TButton')
                self.budding_btn.config(text="🟠 BUDDING CELL (Selected)", style='Selected.TButton')
        else:
            self.normal_btn.config(text="🔵 NORMAL CELL", style='Normal.TButton')
            self.budding_btn.config(text="🟠 BUDDING CELL", style='Budding.TButton')
    
    def select_normal(self):
        """Select as Normal cell"""
        idx = self.incorrect_indices[self.current_idx]
        self.corrections[idx] = 0  # 0 = Normal
        self.auto_next()
    
    def select_budding(self):
        """Select as Budding cell"""
        idx = self.incorrect_indices[self.current_idx]
        self.corrections[idx] = 1  # 1 = Budding
        self.auto_next()
    
    def auto_next(self):
        """Automatically advance to next sample"""
        if self.current_idx < len(self.incorrect_indices) - 1:
            self.current_idx += 1
            self.show_current_sample()
        else:
            self.finish_verification()
    
    def prev_sample(self):
        """Previous sample"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_current_sample()
    
    def next_sample(self):
        """Next sample"""
        if self.current_idx < len(self.incorrect_indices) - 1:
            self.current_idx += 1
            self.show_current_sample()
        else:
            self.finish_verification()
    
    def finish_verification(self):
        """Finish verification"""
        # Check if all samples have been marked
        unmarked = len(self.incorrect_indices) - len(self.corrections)
        if unmarked > 0:
            result = messagebox.askyesno("Incomplete Verification", 
                                       f"You have {unmarked} samples unmarked. "
                                       f"Do you want to finish anyway?")
            if not result:
                return
        
        self.root.quit()
        self.root.destroy()
    
    def update_training_data(self, corrections_summary):
        """Update labels in training data"""
        print(f"\n📝 Updating incorrect labels in training data...")
        
        updates_made = 0
        
        for dataset_idx, user_label in self.corrections.items():
            # Get original sample information
            image, original_label = self.dataset[dataset_idx]
            
            # Only update label file when user choice differs from original label
            if user_label != original_label:
                try:
                    # Get corresponding label file path
                    if hasattr(self.dataset, 'sample_files'):
                        image_file = self.dataset.sample_files[dataset_idx]
                        label_file = self.dataset.labels_dir / image_file.name
                    else:
                        print(f"⚠️  Cannot get dataset file path, skipping index {dataset_idx}")
                        continue
                    
                    # Update label file
                    if label_file.exists():
                        # Save new label
                        np.save(label_file, user_label)
                        
                        change_desc = f"{'Normal' if original_label == 0 else 'Budding'} → {'Normal' if user_label == 0 else 'Budding'}"
                        print(f"  ✅ Updated label: {image_file.name} ({change_desc})")
                        updates_made += 1
                    else:
                        print(f"  ❌ Label file does not exist: {label_file}")
                    
                except Exception as e:
                    print(f"  ❌ Error updating label file {dataset_idx}: {e}")
        
        print(f"📊 Training data update completed: {updates_made} labels corrected")
        
        # Update dataset statistics
        if updates_made > 0:
            self._update_dataset_summary(updates_made)
        
        return updates_made
    
    def _update_dataset_summary(self, updates_made):
        """Update dataset summary file"""
        try:
            summary_file = self.data_dir / "classification_generation_summary_enhanced.json"
            
            # Read existing summary
            summary = {}
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
            
            # Add manual correction record
            if 'manual_corrections' not in summary:
                summary['manual_corrections'] = []
            
            correction_record = {
                'date': datetime.now().isoformat(),
                'corrections_made': updates_made,
                'total_corrections': len(summary.get('manual_corrections', [])) + 1
            }
            summary['manual_corrections'].append(correction_record)
            
            # Save updated summary
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"📝 Dataset summary updated: {summary_file}")
            
        except Exception as e:
            print(f"⚠️  Cannot update dataset summary: {e}")
    
    def run(self):
        """运行GUI"""
        self.root.mainloop()
        return self.corrections

def manual_verification_of_incorrect_predictions(model, dataset, device, y_true, y_pred, data_dir):
    """Manual verification of incorrect predictions"""
    
    print("\n🔍 Starting manual labeling of disputed predictions...")
    
    # Find all incorrect prediction indices
    incorrect_indices = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            incorrect_indices.append(i)
    
    if len(incorrect_indices) == 0:
        print("✅ All predictions match original labels!")
        return y_true, y_pred, {}
    
    print(f"📋 Found {len(incorrect_indices)} samples with prediction-label mismatches")
    print("🖱️  Launching GUI interface for manual labeling...")
    
    # Launch GUI
    gui = ManualVerificationGUI(model, dataset, device, incorrect_indices, data_dir)
    user_corrections = gui.run()
    
    # Apply corrections - use user labels as true labels
    corrected_y_true = y_true.copy()
    
    correction_stats = {
        'label_corrected': 0,  # Original label was corrected
        'label_confirmed': 0,  # Original label was confirmed correct
        'not_verified': 0      # Not verified
    }
    
    for i, idx in enumerate(incorrect_indices):
        if idx in user_corrections:
            user_label = user_corrections[idx]
            original_label = y_true[idx]
            
            if user_label != original_label:
                # User thinks original label is wrong, use user label
                corrected_y_true[idx] = user_label
                correction_stats['label_corrected'] += 1
            else:
                # User confirms original label is correct
                correction_stats['label_confirmed'] += 1
        else:
            correction_stats['not_verified'] += 1
    
    # Update training data files
    if user_corrections:
        try:
            updates_made = gui.update_training_data(user_corrections)
            print(f"💾 Training data updated: {updates_made} files relabeled")
        except Exception as e:
            print(f"⚠️  Cannot update training data files: {e}")
    
    print(f"\n📊 Manual labeling results:")
    print(f"  🔄 Labels corrected: {correction_stats['label_corrected']}")
    print(f"  ✅ Labels confirmed: {correction_stats['label_confirmed']}")
    print(f"  ⏭️  Not labeled: {correction_stats['not_verified']}")
    
    return corrected_y_true, y_pred, user_corrections

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Visualize classification model test results')
    parser.add_argument('--data_dir', default='classification_data_final_fixed',
                      help='Test data directory')
    parser.add_argument('--model_path', default='classification_models/best_f1_model.pth',
                      help='Path to trained model')
    parser.add_argument('--output_dir', default='visualization_results',
                      help='Output directory')
    parser.add_argument('--num_samples', type=int, default=16,
                      help='Number of samples to visualize')
    parser.add_argument('--test_split', type=float, default=0.2,
                      help='Test set ratio')
    parser.add_argument('--manual_verify', action='store_true',
                      help='Enable manual verification of incorrect predictions')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🎯 Classification Model Test Results Visualization")
    print("=" * 50)
    
    try:
        # 1. Load model
        model, device = load_trained_model(args.model_path)
        
        # 2. Load test data
        print(f"\n📂 Loading test data from: {args.data_dir}")
        dataset = CellClassificationDataset(args.data_dir, transform=None, augment=False)
        
        # 3. Create test set (using same split method)
        total_size = len(dataset)
        test_size = int(args.test_split * total_size)
        train_size = total_size - test_size
        
        _, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        print(f"📊 Test set size: {len(test_dataset)}")
        
        # 4. Make predictions
        predictions, labels, probabilities = predict_on_dataset(model, test_dataset, device)
        
        # 5. Manual verification (if enabled)
        corrected_labels = labels
        corrections = {}
        if args.manual_verify:
            corrected_labels, predictions, corrections = manual_verification_of_incorrect_predictions(
                model, test_dataset, device, labels, predictions, args.data_dir)
            
            # Save correction information
            correction_info = {
                'corrections': {str(k): int(v) for k, v in corrections.items()},  # Convert to JSON serializable
                'original_labels': [int(x) for x in labels],
                'corrected_labels': [int(x) for x in corrected_labels]
            }
            with open(output_dir / "manual_corrections.json", 'w') as f:
                json.dump(correction_info, f, indent=2)
        
        # 6. Performance analysis (using corrected labels)
        print(f"\n📊 Performance Analysis {'(After Manual Correction)' if args.manual_verify else '(Original Labels)'}")
        performance = analyze_model_performance(corrected_labels, predictions, probabilities)
        
        # If manual verification was performed, also calculate original performance for comparison
        if args.manual_verify:
            print(f"\n📊 Original Performance Analysis (Before Correction):")
            original_performance = analyze_model_performance(labels, predictions, probabilities)
            
            performance['original_performance'] = original_performance
            performance['manual_corrections_applied'] = True
            performance['corrections_summary'] = {
                'total_corrections': len([c for c in corrections.values() if c]),
                'total_incorrect_reviewed': len(corrections)
            }
        
        # 7. Save performance report
        with open(output_dir / "performance_report.json", 'w') as f:
            json.dump(performance, f, indent=2)
        
        # 8. Generate visualizations (using corrected labels)
        print(f"\n🎨 Generating visualization results...")
        
        # Confusion matrix
        plot_confusion_matrix_detailed(corrected_labels, predictions, 
                                     str(output_dir / "confusion_matrix.png"))
        
        # Probability distribution
        plot_probability_distribution(probabilities, corrected_labels, 
                                    str(output_dir / "probability_distribution.png"))
        
        # Sample prediction results
        visualize_sample_predictions(model, test_dataset, device, args.num_samples,
                                   str(output_dir / "sample_predictions_all.png"), 'all')
        
        visualize_sample_predictions(model, test_dataset, device, args.num_samples,
                                   str(output_dir / "sample_predictions_correct.png"), 'correct')
        
        visualize_sample_predictions(model, test_dataset, device, args.num_samples,
                                   str(output_dir / "sample_predictions_incorrect.png"), 'incorrect')
        
        print(f"\n✅ Visualization completed! Results saved to: {output_dir}")
        print(f"📊 {'Corrected ' if args.manual_verify else ''}Overall Accuracy: {performance['accuracy']:.4f}")
        
        if args.manual_verify and 'original_performance' in performance:
            original_acc = performance['original_performance']['accuracy']
            corrected_acc = performance['accuracy']
            improvement = corrected_acc - original_acc
            print(f"📈 Accuracy Improvement: {original_acc:.4f} → {corrected_acc:.4f} (+{improvement:.4f})")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 