import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import random
from typing import List

def show_random_samples(data_dir: str, num_samples: int = 20, class_type: str = 'all'):
    
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    
    # Get all valid samples
    image_files = [f for f in images_dir.glob("cell_*.npy") if not f.name.startswith("._")]
    valid_samples = []
    
    for image_file in image_files:
        label_file = labels_dir / image_file.name
        if label_file.exists():
            try:
                label = np.load(label_file)
                if class_type == 'all' or \
                   (class_type == 'normal' and label == 0) or \
                   (class_type == 'budding' and label == 1):
                    valid_samples.append((image_file, label))
            except Exception as e:
                print(f"⚠️  Error loading {image_file.name}: {e}")
    
    if not valid_samples:
        print(f"❌ No valid samples found for class_type '{class_type}'")
        return
    
    # Random sample
    num_to_show = min(num_samples, len(valid_samples))
    random_samples = random.sample(valid_samples, num_to_show)
    
    # Calculate grid size
    cols = int(np.ceil(np.sqrt(num_to_show)))
    rows = int(np.ceil(num_to_show / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if num_to_show == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    print(f"\n📋 Showing {num_to_show} random samples (class_type: {class_type})")
    print("Note down the filenames of any samples you want to remove:")
    print("-" * 60)
    
    for i, (image_file, label) in enumerate(random_samples):
        try:
            image = np.load(image_file)
            
            ax = axes[i]
            ax.imshow(image, cmap='gray')
            
            label_text = "Budding" if label == 1 else "Normal"
            filename = image_file.name
            
            ax.set_title(f"{filename}\nLabel: {label_text}", fontsize=8)
            ax.axis('off')
            
            print(f"{i+1:2d}. {filename} -> {label_text}")
            
        except Exception as e:
            print(f"⚠️  Error displaying {image_file.name}: {e}")
    
    # Hide extra subplots
    for i in range(num_to_show, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Random Samples for Manual Review - {class_type.capitalize()}', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("-" * 60)
    print(f"Total valid samples of type '{class_type}': {len(valid_samples)}")

def remove_samples_by_filename(data_dir: str, filenames_to_remove: List[str], dry_run: bool = True):
    """根据文件名删除样本"""
    
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    
    removed_count = 0
    not_found_count = 0
    
    print(f"{'🔍 DRY RUN' if dry_run else '🗑️  REMOVING'} - Processing {len(filenames_to_remove)} files:")
    print("-" * 60)
    
    for filename in filenames_to_remove:
        image_file = images_dir / filename
        label_file = labels_dir / filename
        
        image_exists = image_file.exists()
        label_exists = label_file.exists()
        
        if image_exists or label_exists:
            if not dry_run:
                if image_exists:
                    image_file.unlink()
                if label_exists:
                    label_file.unlink()
            
            status = "WOULD REMOVE" if dry_run else "REMOVED"
            files_info = []
            if image_exists:
                files_info.append("image")
            if label_exists:
                files_info.append("label")
            
            print(f"✅ {status}: {filename} ({', '.join(files_info)})")
            removed_count += 1
        else:
            print(f"❌ NOT FOUND: {filename}")
            not_found_count += 1
    
    print("-" * 60)
    print(f"Summary:")
    print(f"  {'Would remove' if dry_run else 'Removed'}: {removed_count} samples")
    print(f"  Not found: {not_found_count} samples")
    
    if dry_run:
        print(f"\n💡 This was a dry run. Use --execute to actually remove files.")
    else:
        print(f"\n✅ Files have been removed. You can now train the model with the remaining samples.")

def count_samples_by_class(data_dir: str):
    """统计每个类别的样本数量"""
    
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    
    # Get all valid samples
    image_files = [f for f in images_dir.glob("cell_*.npy") if not f.name.startswith("._")]
    
    normal_count = 0
    budding_count = 0
    error_count = 0
    
    for image_file in image_files:
        label_file = labels_dir / image_file.name
        if label_file.exists():
            try:
                label = np.load(label_file)
                if label == 0:
                    normal_count += 1
                elif label == 1:
                    budding_count += 1
            except Exception:
                error_count += 1
    
    total = normal_count + budding_count
    
    print(f"📊 Dataset Statistics for {data_dir}:")
    print(f"   Normal cells (0): {normal_count} ({normal_count/total*100:.1f}%)" if total > 0 else "   Normal cells (0): 0")
    print(f"   Budding cells (1): {budding_count} ({budding_count/total*100:.1f}%)" if total > 0 else "   Budding cells (1): 0")
    print(f"   Total valid: {total}")
    if error_count > 0:
        print(f"   Errors: {error_count}")
    
    return normal_count, budding_count, error_count

def main():
    parser = argparse.ArgumentParser(description='Manual data cleanup helper for cell classification')
    parser.add_argument('action', choices=['show', 'remove', 'count'], 
                       help='Action to perform')
    parser.add_argument('--data_dir', default='classification_data_final_fixed',
                       help='Data directory')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples to show (for show action)')
    parser.add_argument('--class_type', choices=['all', 'normal', 'budding'], default='all',
                       help='Class type to show (for show action)')
    parser.add_argument('--files', nargs='+', 
                       help='List of filenames to remove (for remove action)')
    parser.add_argument('--execute', action='store_true',
                       help='Actually execute the removal (default is dry run)')
    
    args = parser.parse_args()
    
    print(f"🔧 Manual Cleanup Helper for Cell Classification")
    print(f"Data directory: {args.data_dir}")
    print("=" * 60)
    
    if args.action == 'show':
        print(f"Showing {args.num_samples} random samples...")
        show_random_samples(args.data_dir, args.num_samples, args.class_type)
        
    elif args.action == 'remove':
        if not args.files:
            print("❌ Error: --files argument is required for remove action")
            print("Example: python manual_cleanup_helper.py remove --files cell_100.npy cell_200.npy")
            return
        
        remove_samples_by_filename(args.data_dir, args.files, dry_run=not args.execute)
        
    elif args.action == 'count':
        count_samples_by_class(args.data_dir)
    
    print("\n💡 Usage examples:")
    print("   Show samples:     python manual_cleanup_helper.py show --num_samples 25 --class_type budding")
    print("   Check removal:    python manual_cleanup_helper.py remove --files cell_100.npy cell_200.npy")
    print("   Actually remove:  python manual_cleanup_helper.py remove --files cell_100.npy cell_200.npy --execute")
    print("   Count samples:    python manual_cleanup_helper.py count")

if __name__ == "__main__":
    main() 