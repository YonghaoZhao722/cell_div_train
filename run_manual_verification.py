#!/usr/bin/env python3
"""
Example script to run manual verification of cell classification results.
This script demonstrates how to use the manual verification GUI to correct
mislabeled data and get more accurate performance metrics.
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_manual_verification(data_dir='classification_data_final_fixed', 
                          model_path='classification_models/best_f1_model.pth',
                          output_dir='verification_results'):
    """Run the manual verification process"""
    
    print("🔍 Starting Manual Cell Classification Verification")
    print("=" * 60)
    print()
    print("This tool will:")
    print("1. Load your trained model and test data")
    print("2. Identify predictions that don't match original labels") 
    print("3. Show you each 'incorrect' prediction in a GUI")
    print("4. Let you decide if the model is actually right")
    print("5. Recalculate performance with corrected labels")
    print()
    
    # Check if required files exist
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("Please train a model first or specify the correct path.")
        return False
    
    if not Path(data_dir).exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please specify the correct data directory.")
        return False
    
    # Run the visualization script with manual verification enabled
    cmd = [
        sys.executable, 'visualize_test_results.py',
        '--data_dir', data_dir,
        '--model_path', model_path,
        '--output_dir', output_dir,
        '--manual_verify',
        '--num_samples', '20'
    ]
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ Manual verification completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📄 Check manual_corrections.json for correction details")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running manual verification: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️  Process interrupted by user")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run manual verification of cell classification')
    parser.add_argument('--data_dir', default='classification_data_final_fixed',
                      help='Directory containing test data')
    parser.add_argument('--model_path', default='classification_models/best_f1_model.pth',
                      help='Path to trained model')
    parser.add_argument('--output_dir', default='verification_results', 
                      help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    success = run_manual_verification(args.data_dir, args.model_path, args.output_dir)
    
    if success:
        print("\n🎉 Manual verification process completed!")
        print("\n📋 Next steps:")
        print(f"  1. Check {args.output_dir}/performance_report.json for detailed metrics")
        print(f"  2. Review {args.output_dir}/manual_corrections.json for what was corrected")
        print(f"  3. Look at the generated visualizations in {args.output_dir}/")
        print("  4. Use the corrected performance metrics for model evaluation")
    else:
        print("\n❌ Manual verification failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 