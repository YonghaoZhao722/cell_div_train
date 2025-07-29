#!/usr/bin/env python3
"""
Hint Channel Data Creator for Budding Cell Classification

This script implements 方案一 (Solution 1) to solve the "identity confusion" problem
by adding a hint channel that provides mother cell centroid information to the model.

The script converts single-channel binary masks to dual-channel inputs:
- Channel 1: Original binary mask (mother + daughter regions)  
- Channel 2: Hint channel with mother cell centroid marked as 1

This breaks the symmetry and eliminates model confusion when mother and daughter
cells are similar in size.

Author: Assistant
Date: 2024
"""

import os
import numpy as np
import cv2
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from typing import Tuple, Optional
import sys

# Import from existing modules
sys.path.append('.')
try:
    from data_processor_unet import CellDataProcessor
except ImportError:
    print("⚠️  Could not import CellDataProcessor. Make sure data_processor_unet.py is in the same directory.")

class HintChannelDataCreator:
    """
    Creates hint channel data to solve identity confusion in budding cell classification
    
    Problem: When mother and daughter cells are similar in size, the model cannot
    distinguish which region should be assigned to which output channel.
    
    Solution: Add a second input channel containing a "hint point" at the mother cell centroid.
    This breaks symmetry and provides explicit guidance to the model.
    """
    
    def __init__(self, input_dir: str = "processed_data", output_dir: str = "processed_data_with_hints"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Check if input directory exists
        if not self.input_dir.exists():
            raise ValueError(f"Input directory {self.input_dir} does not exist!")
        
        if not (self.input_dir / "images").exists() or not (self.input_dir / "labels").exists():
            raise ValueError(f"Input directory {self.input_dir} must contain 'images' and 'labels' subdirectories!")
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)
        
        print(f"🔧 Hint Channel Data Creator Initialized")
        print(f"   Input directory: {self.input_dir}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Task: Add mother cell centroid hints to break identity confusion")
        
    def calculate_centroid(self, mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Calculate the centroid (center of mass) of a binary mask
        
        Args:
            mask: Binary mask (0s and 1s)
            
        Returns:
            (y, x) coordinates of centroid, or None if mask is empty
        """
        if mask.sum() == 0:
            return None
            
        # Find all coordinates where mask is 1
        coords = np.argwhere(mask > 0)
        
        # Calculate centroid
        centroid_y = int(np.mean(coords[:, 0]))
        centroid_x = int(np.mean(coords[:, 1]))
        
        return (centroid_y, centroid_x)
    
    def create_hint_channel(self, label_mask: np.ndarray, hint_type: str = "point") -> np.ndarray:
        """
        Create hint channel from label mask
        
        Args:
            label_mask: 3-class segmentation mask (0=background, 1=daughter, 2=mother)
            hint_type: Type of hint ("point" for single pixel, "region" for small region)
            
        Returns:
            Hint channel with mother cell hint
        """
        hint_channel = np.zeros_like(label_mask, dtype=np.float32)
        
        # Extract mother cell region (class 2)
        mother_mask = (label_mask == 2).astype(np.uint8)
        
        if mother_mask.sum() == 0:
            print("⚠️  Warning: No mother cell found in label mask")
            return hint_channel
        
        # Calculate mother cell centroid
        centroid = self.calculate_centroid(mother_mask)
        
        if centroid is None:
            return hint_channel
        
        centroid_y, centroid_x = centroid
        
        if hint_type == "point":
            # Single pixel hint at centroid
            hint_channel[centroid_y, centroid_x] = 1.0
            
        elif hint_type == "region":
            # Small circular region around centroid
            radius = 3
            y_coords, x_coords = np.ogrid[:hint_channel.shape[0], :hint_channel.shape[1]]
            mask_circle = ((y_coords - centroid_y) ** 2 + (x_coords - centroid_x) ** 2) <= radius ** 2
            hint_channel[mask_circle] = 1.0
            
        return hint_channel
    
    def create_dual_channel_input(self, original_input: np.ndarray, hint_channel: np.ndarray) -> np.ndarray:
        """
        Combine original single-channel input with hint channel to create dual-channel input
        
        Args:
            original_input: Original single-channel binary mask [H, W]
            hint_channel: Hint channel with mother centroid [H, W]
            
        Returns:
            Dual-channel input [H, W, 2]
        """
        # Ensure both inputs are 2D
        if len(original_input.shape) == 3:
            original_input = original_input.squeeze()
        if len(hint_channel.shape) == 3:
            hint_channel = hint_channel.squeeze()
        
        # Stack channels
        dual_channel = np.stack([original_input, hint_channel], axis=-1)
        
        return dual_channel.astype(np.float32)
    
    def process_single_sample(self, sample_name: str, hint_type: str = "point") -> bool:
        """
        Process a single training sample to add hint channel
        
        Args:
            sample_name: Name of the sample (e.g., "pair_1")
            hint_type: Type of hint to create
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load original data
            input_path = self.input_dir / "images" / f"{sample_name}.npy"
            label_path = self.input_dir / "labels" / f"{sample_name}.npy"
            
            if not input_path.exists() or not label_path.exists():
                print(f"⚠️  Missing files for sample {sample_name}")
                return False
            
            original_input = np.load(input_path)  # [H, W] binary mask
            label_mask = np.load(label_path)      # [H, W] 3-class segmentation
            
            # Create hint channel
            hint_channel = self.create_hint_channel(label_mask, hint_type)
            
            # Create dual-channel input
            dual_channel_input = self.create_dual_channel_input(original_input, hint_channel)
            
            # Save processed data
            output_input_path = self.output_dir / "images" / f"{sample_name}.npy"
            output_label_path = self.output_dir / "labels" / f"{sample_name}.npy"
            
            np.save(output_input_path, dual_channel_input)
            np.save(output_label_path, label_mask)  # Labels remain unchanged
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing sample {sample_name}: {e}")
            return False
    
    def analyze_sample_statistics(self, sample_name: str) -> dict:
        """
        Analyze statistics for a single sample
        
        Args:
            sample_name: Name of the sample
            
        Returns:
            Dictionary with sample statistics
        """
        try:
            label_path = self.input_dir / "labels" / f"{sample_name}.npy"
            label_mask = np.load(label_path)
            
            # Calculate areas
            mother_area = np.sum(label_mask == 2)
            daughter_area = np.sum(label_mask == 1)
            background_area = np.sum(label_mask == 0)
            
            # Calculate ratio
            size_ratio = daughter_area / mother_area if mother_area > 0 else 0
            
            # Calculate mother centroid
            mother_mask = (label_mask == 2).astype(np.uint8)
            centroid = self.calculate_centroid(mother_mask)
            
            return {
                'sample_name': sample_name,
                'mother_area': int(mother_area),
                'daughter_area': int(daughter_area),
                'background_area': int(background_area),
                'size_ratio': float(size_ratio),
                'mother_centroid': centroid,
                'total_area': int(mother_area + daughter_area + background_area)
            }
            
        except Exception as e:
            print(f"❌ Error analyzing sample {sample_name}: {e}")
            return {}
    
    def process_all_samples(self, hint_type: str = "point", analyze: bool = True) -> dict:
        """
        Process all samples in the input directory
        
        Args:
            hint_type: Type of hint to create ("point" or "region")
            analyze: Whether to perform statistical analysis
            
        Returns:
            Processing results and statistics
        """
        print(f"\n🚀 Processing all samples with hint type: {hint_type}")
        
        # Find all sample files
        sample_files = list(self.input_dir.glob("images/pair_*.npy"))
        sample_names = [f.stem for f in sample_files]
        
        if not sample_names:
            print("❌ No training samples found!")
            return {}
        
        print(f"📊 Found {len(sample_names)} samples to process")
        
        # Process samples
        successful = 0
        failed = 0
        sample_statistics = []
        
        progress_bar = tqdm(sample_names, desc="Processing samples")
        
        for sample_name in progress_bar:
            # Process sample
            if self.process_single_sample(sample_name, hint_type):
                successful += 1
                
                # Analyze if requested
                if analyze:
                    stats = self.analyze_sample_statistics(sample_name)
                    if stats:
                        sample_statistics.append(stats)
            else:
                failed += 1
            
            progress_bar.set_postfix({
                "Success": successful,
                "Failed": failed
            })
        
        # Compile results
        results = {
            'processing_date': datetime.now().isoformat(),
            'hint_type': hint_type,
            'total_samples': len(sample_names),
            'successful': successful,
            'failed': failed,
            'input_dir': str(self.input_dir),
            'output_dir': str(self.output_dir),
            'sample_statistics': sample_statistics
        }
        
        # Save results
        self._save_processing_results(results)
        
        # Print summary
        print(f"\n✅ Processing Complete!")
        print(f"   Total samples: {len(sample_names)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Success rate: {100 * successful / len(sample_names):.1f}%")
        
        if analyze and sample_statistics:
            self._print_statistical_summary(sample_statistics)
        
        return results
    
    def _save_processing_results(self, results: dict):
        """Save processing results to JSON file"""
        
        results_file = self.output_dir / "hint_channel_processing_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Processing results saved to {results_file}")
        
        # Also copy generation summary if it exists
        original_summary = self.input_dir / "generation_summary.json"
        if original_summary.exists():
            new_summary_path = self.output_dir / "original_generation_summary.json"
            import shutil
            shutil.copy2(original_summary, new_summary_path)
            print(f"📄 Original generation summary copied to {new_summary_path}")
    
    def _print_statistical_summary(self, sample_statistics: list):
        """Print statistical summary of processed samples"""
        
        if not sample_statistics:
            return
        
        size_ratios = [s['size_ratio'] for s in sample_statistics if s['size_ratio'] > 0]
        mother_areas = [s['mother_area'] for s in sample_statistics]
        daughter_areas = [s['daughter_area'] for s in sample_statistics]
        
        print(f"\n📊 Statistical Summary:")
        print(f"   Size ratios (daughter/mother):")
        print(f"     Mean: {np.mean(size_ratios):.3f}")
        print(f"     Median: {np.median(size_ratios):.3f}")
        print(f"     Min: {np.min(size_ratios):.3f}")
        print(f"     Max: {np.max(size_ratios):.3f}")
        print(f"     Std: {np.std(size_ratios):.3f}")
        
        print(f"   Mother cell areas:")
        print(f"     Mean: {np.mean(mother_areas):.1f} pixels")
        print(f"     Median: {np.median(mother_areas):.1f} pixels")
        
        print(f"   Daughter cell areas:")
        print(f"     Mean: {np.mean(daughter_areas):.1f} pixels") 
        print(f"     Median: {np.median(daughter_areas):.1f} pixels")
        
        # Identify challenging cases (similar sizes)
        challenging_cases = [s for s in sample_statistics if s['size_ratio'] > 0.7]
        print(f"   Challenging cases (size_ratio > 0.7): {len(challenging_cases)}/{len(sample_statistics)} ({100*len(challenging_cases)/len(sample_statistics):.1f}%)")
    
    def visualize_sample_comparison(self, sample_name: str, save_path: Optional[str] = None):
        """
        Visualize comparison between original and hint-enhanced input
        
        Args:
            sample_name: Name of the sample to visualize
            save_path: Path to save the visualization
        """
        try:
            # Load original data
            original_input = np.load(self.input_dir / "images" / f"{sample_name}.npy")
            label_mask = np.load(self.input_dir / "labels" / f"{sample_name}.npy")
            
            # Load processed data
            dual_channel_input = np.load(self.output_dir / "images" / f"{sample_name}.npy")
            
            # Extract channels
            binary_channel = dual_channel_input[:, :, 0]
            hint_channel = dual_channel_input[:, :, 1]
            
            # Create visualization
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # Original input
            axes[0].imshow(original_input.squeeze(), cmap='gray')
            axes[0].set_title('Original Input\n(Binary Mask)')
            axes[0].axis('off')
            
            # Hint channel
            axes[1].imshow(hint_channel, cmap='Reds')
            axes[1].set_title('Hint Channel\n(Mother Centroid)')
            axes[1].axis('off')
            
            # Combined visualization
            combined = np.zeros((*binary_channel.shape, 3))
            combined[:, :, 0] = binary_channel  # Red for binary mask
            combined[:, :, 1] = hint_channel    # Green for hint
            axes[2].imshow(combined)
            axes[2].set_title('Combined Channels\n(Red: Mask, Green: Hint)')
            axes[2].axis('off')
            
            # Ground truth labels
            label_colored = np.zeros((*label_mask.shape, 3))
            label_colored[label_mask == 1] = [0, 1, 0]    # Green for daughter
            label_colored[label_mask == 2] = [1, 0, 0]    # Red for mother
            axes[3].imshow(label_colored)
            axes[3].set_title('Ground Truth\n(Red: Mother, Green: Daughter)')
            axes[3].axis('off')
            
            plt.suptitle(f'Hint Channel Visualization - {sample_name}', fontsize=14)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Visualization saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            print(f"❌ Error creating visualization for {sample_name}: {e}")

def parse_arguments():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description="🔧 Hint Channel Data Creator for Budding Cell Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📋 Examples:
  python create_hint_channel_data.py --input processed_data --output processed_data_with_hints
  python create_hint_channel_data.py --hint-type region --analyze
  python create_hint_channel_data.py --visualize pair_1 --save-viz sample_viz.png

🎯 Purpose:
  Solves the "identity confusion" problem by adding mother cell centroid hints.
  Converts single-channel binary masks to dual-channel inputs for better model training.

💡 How it works:
  - Channel 1: Original binary mask (mother + daughter regions)
  - Channel 2: Hint channel with mother cell centroid marked
  - This breaks symmetry and eliminates confusion when cells are similar in size
        """
    )
    
    parser.add_argument(
        '--input', '--input-dir',
        type=str,
        default='processed_data',
        help='Input directory containing processed training data (default: processed_data)'
    )
    
    parser.add_argument(
        '--output', '--output-dir', 
        type=str,
        default='processed_data_with_hints',
        help='Output directory for hint-enhanced data (default: processed_data_with_hints)'
    )
    
    parser.add_argument(
        '--hint-type',
        type=str,
        choices=['point', 'region'],
        default='point',
        help='Type of hint to create (default: point)'
    )
    
    parser.add_argument(
        '--no-analyze',
        action='store_true',
        help='Skip statistical analysis'
    )
    
    parser.add_argument(
        '--visualize',
        type=str,
        default=None,
        help='Visualize a specific sample (e.g., pair_1)'
    )
    
    parser.add_argument(
        '--save-viz',
        type=str,
        default=None,
        help='Save visualization to file instead of displaying'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    
    args = parse_arguments()
    
    print("🔧 Hint Channel Data Creator")
    print("=" * 50)
    print("🎯 Solving identity confusion by adding mother cell centroid hints")
    print()
    
    try:
        # Initialize creator
        creator = HintChannelDataCreator(
            input_dir=args.input,
            output_dir=args.output
        )
        
        # Handle visualization request
        if args.visualize:
            print(f"📊 Creating visualization for sample: {args.visualize}")
            creator.visualize_sample_comparison(
                sample_name=args.visualize,
                save_path=args.save_viz
            )
            return
        
        # Process all samples
        results = creator.process_all_samples(
            hint_type=args.hint_type,
            analyze=not args.no_analyze
        )
        
        if results.get('successful', 0) > 0:
            print(f"\n✅ Success! Created {results['successful']} hint-enhanced training samples")
            print(f"📁 Output directory: {args.output}")
            print(f"🔧 Hint type: {args.hint_type}")
            print(f"\n💡 Next steps:")
            print(f"   1. Modify your U-Net model to accept 2 input channels instead of 1")
            print(f"   2. Update your training script to use the new data directory")
            print(f"   3. Train with the hint-enhanced data to solve identity confusion")
            
            if args.visualize is None:
                sample_files = list(Path(args.input).glob("images/pair_*.npy"))
                if sample_files:
                    sample_name = sample_files[0].stem
                    print(f"   4. Visualize results: python create_hint_channel_data.py --visualize {sample_name}")
        else:
            print("❌ No samples were successfully processed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 