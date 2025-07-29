#!/usr/bin/env python3
"""
Data Distribution Analyzer for Budding Cell Classification

This script implements 方案二 (Solution 2) to analyze the distribution of cell size ratios
in the training data. It helps identify potential data bias that could cause model
performance issues when mother and daughter cells are similar in size.

The script analyzes:
- Daughter/Mother area ratios for all training samples
- Distribution patterns and potential bias
- Identification of challenging cases (similar sizes)
- Recommendations for data balancing strategies

Author: Assistant
Date: 2024
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import argparse
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class CellDataDistributionAnalyzer:
    """
    Analyzes cell size distribution in budding cell classification training data
    
    Purpose: Identify data bias that could lead to poor model performance on
    challenging cases where mother and daughter cells are similar in size.
    """
    
    def __init__(self, data_dir: str = "processed_data", output_dir: str = "analysis_results"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Check if data directory exists
        if not self.data_dir.exists():
            raise ValueError(f"Data directory {self.data_dir} does not exist!")
        
        if not (self.data_dir / "images").exists() or not (self.data_dir / "labels").exists():
            raise ValueError(f"Data directory {self.data_dir} must contain 'images' and 'labels' subdirectories!")
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize analysis results
        self.sample_data = []
        self.analysis_complete = False
        
        print(f"📊 Cell Data Distribution Analyzer Initialized")
        print(f"   Data directory: {self.data_dir}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Task: Analyze cell size distribution to identify potential bias")
    
    def analyze_single_sample(self, sample_name: str) -> Optional[Dict]:
        """
        Analyze a single training sample
        
        Args:
            sample_name: Name of the sample (e.g., "pair_1")
            
        Returns:
            Dictionary with sample analysis data or None if failed
        """
        try:
            # Load label mask
            label_path = self.data_dir / "labels" / f"{sample_name}.npy"
            
            if not label_path.exists():
                print(f"⚠️  Label file not found for sample {sample_name}")
                return None
            
            label_mask = np.load(label_path)
            
            # Calculate areas
            mother_area = np.sum(label_mask == 2)
            daughter_area = np.sum(label_mask == 1)
            background_area = np.sum(label_mask == 0)
            total_area = label_mask.size
            
            # Handle edge cases
            if mother_area == 0:
                print(f"⚠️  No mother cell found in sample {sample_name}")
                return None
                
            if daughter_area == 0:
                print(f"⚠️  No daughter cell found in sample {sample_name}")
                return None
            
            # Calculate ratios and metrics
            size_ratio = daughter_area / mother_area
            cell_coverage = (mother_area + daughter_area) / total_area
            relative_mother_size = mother_area / (mother_area + daughter_area)
            relative_daughter_size = daughter_area / (mother_area + daughter_area)
            
            # Calculate shape properties
            mother_mask = (label_mask == 2).astype(np.uint8)
            daughter_mask = (label_mask == 1).astype(np.uint8)
            
            mother_bbox = self._get_bounding_box(mother_mask)
            daughter_bbox = self._get_bounding_box(daughter_mask)
            
            # Categorize difficulty
            difficulty_category = self._categorize_difficulty(size_ratio)
            
            return {
                'sample_name': sample_name,
                'mother_area': int(mother_area),
                'daughter_area': int(daughter_area),
                'background_area': int(background_area),
                'total_area': int(total_area),
                'size_ratio': float(size_ratio),
                'cell_coverage': float(cell_coverage),
                'relative_mother_size': float(relative_mother_size),
                'relative_daughter_size': float(relative_daughter_size),
                'mother_bbox': mother_bbox,
                'daughter_bbox': daughter_bbox,
                'difficulty_category': difficulty_category,
                'is_challenging': size_ratio > 0.7  # Flag for challenging cases
            }
            
        except Exception as e:
            print(f"❌ Error analyzing sample {sample_name}: {e}")
            return None
    
    def _get_bounding_box(self, mask: np.ndarray) -> Dict:
        """Get bounding box properties of a mask"""
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return {'width': 0, 'height': 0, 'aspect_ratio': 0}
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        aspect_ratio = width / height if height > 0 else 0
        
        return {
            'width': int(width),
            'height': int(height), 
            'aspect_ratio': float(aspect_ratio)
        }
    
    def _categorize_difficulty(self, size_ratio: float) -> str:
        """Categorize sample difficulty based on size ratio"""
        if size_ratio < 0.3:
            return "easy"        # Very different sizes
        elif size_ratio < 0.6:
            return "moderate"    # Moderately different sizes
        elif size_ratio < 0.8:
            return "challenging" # Similar sizes
        else:
            return "very_challenging"  # Very similar sizes
    
    def analyze_all_samples(self) -> List[Dict]:
        """
        Analyze all samples in the data directory
        
        Returns:
            List of sample analysis dictionaries
        """
        print(f"\n🔍 Analyzing all samples in {self.data_dir}")
        
        # Find all sample files
        sample_files = list(self.data_dir.glob("labels/pair_*.npy"))
        sample_names = [f.stem for f in sample_files]
        
        if not sample_names:
            print("❌ No training samples found!")
            return []
        
        print(f"📊 Found {len(sample_names)} samples to analyze")
        
        # Analyze samples
        self.sample_data = []
        failed_samples = []
        
        progress_bar = tqdm(sample_names, desc="Analyzing samples")
        
        for sample_name in progress_bar:
            result = self.analyze_single_sample(sample_name)
            if result:
                self.sample_data.append(result)
            else:
                failed_samples.append(sample_name)
            
            progress_bar.set_postfix({
                "Success": len(self.sample_data),
                "Failed": len(failed_samples)
            })
        
        self.analysis_complete = True
        
        print(f"\n✅ Analysis Complete!")
        print(f"   Successfully analyzed: {len(self.sample_data)} samples")
        print(f"   Failed: {len(failed_samples)} samples")
        
        if failed_samples:
            print(f"⚠️  Failed samples: {failed_samples}")
        
        return self.sample_data
    
    def generate_distribution_report(self) -> Dict:
        """
        Generate comprehensive distribution analysis report
        
        Returns:
            Dictionary containing analysis results
        """
        if not self.analysis_complete or not self.sample_data:
            print("❌ No analysis data available. Run analyze_all_samples() first.")
            return {}
        
        print("\n📈 Generating distribution analysis report...")
        
        # Extract data for analysis
        size_ratios = [s['size_ratio'] for s in self.sample_data]
        mother_areas = [s['mother_area'] for s in self.sample_data]
        daughter_areas = [s['daughter_area'] for s in self.sample_data]
        cell_coverages = [s['cell_coverage'] for s in self.sample_data]
        difficulties = [s['difficulty_category'] for s in self.sample_data]
        
        # Calculate statistics
        ratio_stats = {
            'mean': float(np.mean(size_ratios)),
            'median': float(np.median(size_ratios)),
            'std': float(np.std(size_ratios)),
            'min': float(np.min(size_ratios)),
            'max': float(np.max(size_ratios)),
            'q25': float(np.percentile(size_ratios, 25)),
            'q75': float(np.percentile(size_ratios, 75))
        }
        
        # Analyze difficulty distribution
        difficulty_counts = pd.Series(difficulties).value_counts()
        difficulty_percentages = (difficulty_counts / len(difficulties) * 100).round(1)
        
        # Identify challenging cases
        challenging_samples = [s for s in self.sample_data if s['is_challenging']]
        challenging_ratio = len(challenging_samples) / len(self.sample_data)
        
        # Identify potential bias
        bias_analysis = self._analyze_bias(size_ratios)
        
        # Create comprehensive report
        report = {
            'analysis_date': datetime.now().isoformat(),
            'total_samples': len(self.sample_data),
            'data_directory': str(self.data_dir),
            
            # Size ratio statistics
            'size_ratio_statistics': ratio_stats,
            
            # Difficulty distribution
            'difficulty_distribution': {
                'counts': difficulty_counts.to_dict(),
                'percentages': difficulty_percentages.to_dict()
            },
            
            # Challenging cases analysis
            'challenging_cases': {
                'count': len(challenging_samples),
                'percentage': float(challenging_ratio * 100),
                'threshold': 0.7,
                'samples': [s['sample_name'] for s in challenging_samples]
            },
            
            # Bias analysis
            'bias_analysis': bias_analysis,
            
            # Area statistics
            'area_statistics': {
                'mother_areas': {
                    'mean': float(np.mean(mother_areas)),
                    'median': float(np.median(mother_areas)),
                    'std': float(np.std(mother_areas))
                },
                'daughter_areas': {
                    'mean': float(np.mean(daughter_areas)),
                    'median': float(np.median(daughter_areas)),
                    'std': float(np.std(daughter_areas))
                }
            },
            
            # Coverage statistics
            'coverage_statistics': {
                'mean': float(np.mean(cell_coverages)),
                'median': float(np.median(cell_coverages)),
                'std': float(np.std(cell_coverages))
            }
        }
        
        return report
    
    def _analyze_bias(self, size_ratios: List[float]) -> Dict:
        """Analyze potential bias in size ratio distribution"""
        
        # Define ratio ranges
        ranges = {
            'very_small': (0.0, 0.2),   # Daughter much smaller
            'small': (0.2, 0.4),       # Daughter smaller
            'moderate': (0.4, 0.6),    # Moderate difference
            'similar': (0.6, 0.8),     # Similar sizes
            'very_similar': (0.8, 1.0) # Very similar sizes
        }
        
        # Count samples in each range
        range_counts = {}
        for range_name, (min_val, max_val) in ranges.items():
            count = sum(1 for ratio in size_ratios if min_val <= ratio < max_val)
            range_counts[range_name] = count
        
        total_samples = len(size_ratios)
        range_percentages = {k: (v / total_samples * 100) for k, v in range_counts.items()}
        
        # Identify bias
        bias_indicators = []
        
        # Check for under-representation of challenging cases
        challenging_ratio = (range_counts['similar'] + range_counts['very_similar']) / total_samples
        if challenging_ratio < 0.2:
            bias_indicators.append({
                'type': 'underrepresented_challenging_cases',
                'description': f'Only {challenging_ratio:.1%} of samples are challenging (ratio > 0.6)',
                'recommendation': 'Consider oversampling challenging cases or data augmentation'
            })
        
        # Check for over-representation of easy cases
        easy_ratio = (range_counts['very_small'] + range_counts['small']) / total_samples
        if easy_ratio > 0.7:
            bias_indicators.append({
                'type': 'overrepresented_easy_cases',
                'description': f'{easy_ratio:.1%} of samples are easy cases (ratio < 0.4)',
                'recommendation': 'Dataset is heavily biased toward easy cases'
            })
        
        # Check for extreme skewness
        if np.std(size_ratios) < 0.15:
            bias_indicators.append({
                'type': 'low_variance',
                'description': 'Very low variance in size ratios',
                'recommendation': 'Consider adding more diverse samples'
            })
        
        return {
            'range_counts': range_counts,
            'range_percentages': range_percentages,
            'bias_indicators': bias_indicators,
            'overall_bias_score': len(bias_indicators)  # Higher = more biased
        }
    
    def create_visualizations(self, save_plots: bool = True) -> None:
        """
        Create comprehensive visualizations of the data distribution
        
        Args:
            save_plots: Whether to save plots to files
        """
        if not self.analysis_complete or not self.sample_data:
            print("❌ No analysis data available. Run analyze_all_samples() first.")
            return
        
        print("\n📊 Creating visualizations...")
        
        # Extract data
        size_ratios = [s['size_ratio'] for s in self.sample_data]
        mother_areas = [s['mother_area'] for s in self.sample_data]
        daughter_areas = [s['daughter_area'] for s in self.sample_data]
        difficulties = [s['difficulty_category'] for s in self.sample_data]
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Size ratio histogram
        ax1 = plt.subplot(2, 3, 1)
        plt.hist(size_ratios, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(size_ratios), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(size_ratios):.3f}')
        plt.axvline(np.median(size_ratios), color='green', linestyle='--', 
                   label=f'Median: {np.median(size_ratios):.3f}')
        plt.axvline(0.7, color='orange', linestyle=':', 
                   label='Challenging threshold: 0.7')
        plt.xlabel('Daughter/Mother Area Ratio')
        plt.ylabel('Frequency')
        plt.title('Distribution of Size Ratios')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Difficulty categories
        ax2 = plt.subplot(2, 3, 2)
        difficulty_counts = pd.Series(difficulties).value_counts()
        colors = ['green', 'yellow', 'orange', 'red']
        difficulty_counts.plot(kind='bar', color=colors[:len(difficulty_counts)])
        plt.title('Distribution by Difficulty Category')
        plt.xlabel('Difficulty Category')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        # Add percentages on bars
        total = len(difficulties)
        for i, (category, count) in enumerate(difficulty_counts.items()):
            percentage = count / total * 100
            plt.text(i, count + max(difficulty_counts) * 0.01, 
                    f'{percentage:.1f}%', ha='center', va='bottom')
        
        # 3. Scatter plot: Mother vs Daughter areas
        ax3 = plt.subplot(2, 3, 3)
        scatter = plt.scatter(mother_areas, daughter_areas, 
                            c=size_ratios, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Size Ratio')
        plt.xlabel('Mother Cell Area (pixels)')
        plt.ylabel('Daughter Cell Area (pixels)')
        plt.title('Mother vs Daughter Cell Areas')
        
        # Add diagonal line for equal sizes
        max_area = max(max(mother_areas), max(daughter_areas))
        plt.plot([0, max_area], [0, max_area], 'r--', alpha=0.5, label='Equal size line')
        plt.legend()
        
        # 4. Box plot of size ratios by difficulty
        ax4 = plt.subplot(2, 3, 4)
        df_temp = pd.DataFrame({'ratio': size_ratios, 'difficulty': difficulties})
        sns.boxplot(data=df_temp, x='difficulty', y='ratio')
        plt.title('Size Ratio Distribution by Difficulty')
        plt.xticks(rotation=45)
        
        # 5. Cumulative distribution
        ax5 = plt.subplot(2, 3, 5)
        sorted_ratios = np.sort(size_ratios)
        cumulative = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
        plt.plot(sorted_ratios, cumulative, linewidth=2)
        plt.axvline(0.7, color='red', linestyle='--', alpha=0.7, 
                   label='Challenging threshold')
        plt.xlabel('Size Ratio')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution of Size Ratios')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 6. Range analysis
        ax6 = plt.subplot(2, 3, 6)
        ranges = {
            'Very Small\n(0.0-0.2)': sum(1 for r in size_ratios if 0.0 <= r < 0.2),
            'Small\n(0.2-0.4)': sum(1 for r in size_ratios if 0.2 <= r < 0.4),
            'Moderate\n(0.4-0.6)': sum(1 for r in size_ratios if 0.4 <= r < 0.6),
            'Similar\n(0.6-0.8)': sum(1 for r in size_ratios if 0.6 <= r < 0.8),
            'Very Similar\n(0.8-1.0)': sum(1 for r in size_ratios if 0.8 <= r < 1.0)
        }
        
        range_names = list(ranges.keys())
        range_counts = list(ranges.values())
        colors = ['darkgreen', 'green', 'yellow', 'orange', 'red']
        
        bars = plt.bar(range_names, range_counts, color=colors)
        plt.title('Sample Distribution by Size Ratio Ranges')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45, ha='right')
        
        # Add percentages on bars
        total = len(size_ratios)
        for bar, count in zip(bars, range_counts):
            percentage = count / total * 100
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(range_counts) * 0.01,
                    f'{percentage:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / "distribution_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualizations saved to {plot_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_analysis_results(self, report: Dict) -> None:
        """Save analysis results to JSON and CSV files"""
        
        # Save comprehensive report
        report_path = self.output_dir / "distribution_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Analysis report saved to {report_path}")
        
        # Save sample data as CSV
        if self.sample_data:
            df = pd.DataFrame(self.sample_data)
            # Flatten nested dictionaries for CSV
            for col in ['mother_bbox', 'daughter_bbox']:
                if col in df.columns:
                    for key in ['width', 'height', 'aspect_ratio']:
                        df[f'{col}_{key}'] = df[col].apply(lambda x: x.get(key, 0))
                    df = df.drop(columns=[col])
            
            csv_path = self.output_dir / "sample_analysis_data.csv"
            df.to_csv(csv_path, index=False)
            print(f"📄 Sample data saved to {csv_path}")
        
        # Save challenging cases list
        challenging_samples = [s['sample_name'] for s in self.sample_data if s['is_challenging']]
        if challenging_samples:
            challenging_path = self.output_dir / "challenging_samples.txt"
            with open(challenging_path, 'w') as f:
                f.write("# Challenging samples (size_ratio > 0.7)\n")
                f.write("# These samples may benefit from oversampling during training\n\n")
                for sample in challenging_samples:
                    f.write(f"{sample}\n")
            print(f"📄 Challenging samples list saved to {challenging_path}")
    
    def print_summary(self, report: Dict) -> None:
        """Print a formatted summary of the analysis"""
        
        if not report:
            print("❌ No report data available")
            return
        
        print("\n" + "="*60)
        print("📊 DATA DISTRIBUTION ANALYSIS SUMMARY")
        print("="*60)
        
        # Basic statistics
        print(f"\n📈 BASIC STATISTICS:")
        print(f"   Total samples: {report['total_samples']}")
        stats = report['size_ratio_statistics']
        print(f"   Size ratio - Mean: {stats['mean']:.3f}, Median: {stats['median']:.3f}")
        print(f"   Size ratio - Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"   Size ratio - Std Dev: {stats['std']:.3f}")
        
        # Difficulty distribution
        print(f"\n🎯 DIFFICULTY DISTRIBUTION:")
        diff_dist = report['difficulty_distribution']['percentages']
        for category, percentage in diff_dist.items():
            print(f"   {category.capitalize()}: {percentage:.1f}%")
        
        # Challenging cases
        print(f"\n⚠️  CHALLENGING CASES:")
        challenging = report['challenging_cases']
        print(f"   Count: {challenging['count']}/{report['total_samples']} ({challenging['percentage']:.1f}%)")
        print(f"   Threshold: ratio > {challenging['threshold']}")
        
        # Bias analysis
        print(f"\n🔍 BIAS ANALYSIS:")
        bias = report['bias_analysis']
        print(f"   Bias score: {bias['overall_bias_score']}/3 (lower is better)")
        
        if bias['bias_indicators']:
            print("   ⚠️  Potential issues detected:")
            for indicator in bias['bias_indicators']:
                print(f"     - {indicator['type']}: {indicator['description']}")
                print(f"       Recommendation: {indicator['recommendation']}")
        else:
            print("   ✅ No significant bias detected")
        
        # Range distribution
        print(f"\n📊 RANGE DISTRIBUTION:")
        ranges = bias['range_percentages']
        for range_name, percentage in ranges.items():
            print(f"   {range_name.replace('_', ' ').title()}: {percentage:.1f}%")
        
        print("\n" + "="*60)

def parse_arguments():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description="📊 Data Distribution Analyzer for Budding Cell Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📋 Examples:
  python analyze_data_distribution.py --data processed_data --output analysis_results
  python analyze_data_distribution.py --no-plots --export-challenging
  python analyze_data_distribution.py --verbose --threshold 0.8

🎯 Purpose:
  Analyze size ratio distribution to identify potential bias in training data.
  Helps understand why models might struggle with similar-sized mother/daughter cells.

💡 What it analyzes:
  - Daughter/Mother area ratios for all samples
  - Distribution patterns and bias indicators  
  - Challenging cases identification
  - Recommendations for data balancing
        """
    )
    
    parser.add_argument(
        '--data', '--data-dir',
        type=str,
        default='processed_data',
        help='Directory containing processed training data (default: processed_data)'
    )
    
    parser.add_argument(
        '--output', '--output-dir',
        type=str,
        default='analysis_results',
        help='Output directory for analysis results (default: analysis_results)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='Threshold for defining challenging cases (default: 0.7)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip creating visualization plots'
    )
    
    parser.add_argument(
        '--export-challenging',
        action='store_true',
        help='Export list of challenging samples'
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
    
    print("📊 Cell Data Distribution Analyzer")
    print("=" * 50)
    print("🎯 Analyzing size ratio distribution to identify potential bias")
    print()
    
    try:
        # Initialize analyzer
        analyzer = CellDataDistributionAnalyzer(
            data_dir=args.data,
            output_dir=args.output
        )
        
        # Analyze all samples
        print("🔍 Step 1: Analyzing all samples...")
        sample_data = analyzer.analyze_all_samples()
        
        if not sample_data:
            print("❌ No valid samples found for analysis!")
            return 1
        
        # Generate report
        print("📈 Step 2: Generating distribution report...")
        report = analyzer.generate_distribution_report()
        
        # Create visualizations
        if not args.no_plots:
            print("📊 Step 3: Creating visualizations...")
            analyzer.create_visualizations(save_plots=True)
        
        # Save results
        print("💾 Step 4: Saving analysis results...")
        analyzer.save_analysis_results(report)
        
        # Print summary
        analyzer.print_summary(report)
        
        # Additional outputs
        if args.export_challenging:
            challenging_count = report['challenging_cases']['count']
            print(f"\n📋 Exported {challenging_count} challenging samples to challenging_samples.txt")
        
        print(f"\n✅ Analysis complete! Results saved to: {args.output}")
        print(f"📁 Check the following files:")
        print(f"   - distribution_analysis_report.json (comprehensive report)")
        print(f"   - sample_analysis_data.csv (detailed sample data)")
        if not args.no_plots:
            print(f"   - distribution_analysis.png (visualizations)")
        print(f"   - challenging_samples.txt (list of difficult cases)")
        
        # Recommendations
        bias_score = report['bias_analysis']['overall_bias_score']
        challenging_ratio = report['challenging_cases']['percentage']
        
        print(f"\n💡 RECOMMENDATIONS:")
        
        if bias_score > 0:
            print(f"   ⚠️  Data bias detected (score: {bias_score}/3)")
            if challenging_ratio < 20:
                print(f"   🔄 Consider oversampling challenging cases ({challenging_ratio:.1f}% < 20%)")
                print(f"   📈 Implement data augmentation for similar-sized cells")
        else:
            print(f"   ✅ Data distribution looks balanced")
            
        if challenging_ratio > 0:
            print(f"   🎯 Use hint channel method to help with {challenging_ratio:.1f}% challenging cases")
            print(f"   💡 Run: python create_hint_channel_data.py --input {args.data}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 