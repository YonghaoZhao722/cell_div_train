import os
import re
import pandas as pd
import numpy as np
import tifffile
from pathlib import Path
import cv2
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime

class CellDataProcessor:
    """
    Standalone data processor for cell division analysis
    Handles mask-to-mask classification for mother/daughter cell identification
    
    Task: Given segmentation masks, classify which regions are mother vs daughter cells
    Input: Binary masks (mother + daughter regions)
    Output: 3-class segmentation (background=0, daughter=1, mother=2)
    """
    
    def __init__(self, data_root: str = "data", output_dir: str = "processed_data"):
        self.data_root = Path(data_root)
        self.divided_masks_dir = self.data_root / "divided_masks"  # Segmentation masks (labels)
        self.dic_masks_dir = self.data_root / "dic_masks"         # Original DIC images (inputs)
        self.divided_outlines_dir = self.data_root / "divided_outlines"  # Cell annotations
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)
        
        self.master_df = None
        self.target_size = (256, 256)  # Fixed size for training
        
        # Print initialization info
        print(f"=== Cell Data Processor Initialized ===")
        print(f"Data root: {self.data_root}")
        print(f"Divided masks (for labels): {self.divided_masks_dir}")
        print(f"DIC masks (segmentation masks): {self.dic_masks_dir}")
        print(f"Divided outlines (annotations): {self.divided_outlines_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Target image size: {self.target_size}")
        print(f"📋 Task: Mask-to-mask classification (mother/daughter identification)")
        
    def find_matching_files(self) -> List[Tuple[Path, Path, Path]]:
        """
        Find matching file triplets: (divided_mask, dic_mask, outline_txt)
        
        Returns:
            List of (divided_mask_file, dic_mask_file, txt_file) tuples
        """
        print("\n=== Finding Matching File Triplets ===")
        
        # Get all valid files (excluding macOS hidden files)
        divided_mask_files = [f for f in self.divided_masks_dir.glob("*.tif") 
                             if not f.name.startswith("._")]
        dic_mask_files = [f for f in self.dic_masks_dir.glob("*.tif") 
                         if not f.name.startswith("._")]
        txt_files = [f for f in self.divided_outlines_dir.glob("*.txt") 
                    if not f.name.startswith("._")]
        
        print(f"Found {len(divided_mask_files)} divided mask files")
        print(f"Found {len(dic_mask_files)} DIC mask files")
        print(f"Found {len(txt_files)} TXT annotation files")
        
        matched_triplets = []
        unmatched_masks = []
        
        for divided_mask_file in divided_mask_files:
            # Find matching DIC mask and TXT file
            dic_mask_file = self._find_matching_file(divided_mask_file, dic_mask_files)
            txt_file = self._find_matching_file(divided_mask_file, txt_files, target_ext='.txt')
            
            if dic_mask_file and txt_file:
                matched_triplets.append((divided_mask_file, dic_mask_file, txt_file))
                print(f"✅ Complete match:")
                print(f"   Divided mask: {divided_mask_file.name}")
                print(f"   DIC mask:     {dic_mask_file.name}")
                print(f"   Annotations:  {txt_file.name}")
            else:
                unmatched_masks.append(divided_mask_file)
                missing = []
                if not dic_mask_file:
                    missing.append("DIC mask")
                if not txt_file:
                    missing.append("TXT annotation")
                print(f"❌ Incomplete: {divided_mask_file.name} (missing: {', '.join(missing)})")
        
        print(f"\n📊 Matching Summary:")
        print(f"  - Complete triplets: {len(matched_triplets)}")
        print(f"  - Incomplete matches: {len(unmatched_masks)}")
        
        if unmatched_masks:
            print(f"\n⚠️  Files with incomplete matches:")
            for mask_file in unmatched_masks:
                print(f"    - {mask_file.name}")
        
        return matched_triplets
    
    def _find_matching_file(self, reference_file: Path, candidate_files: List[Path], 
                           target_ext: str = None) -> Optional[Path]:
        """
        Find matching file for a reference file using flexible matching
        
        Args:
            reference_file: File to find a match for
            candidate_files: List of candidate files to search in
            target_ext: Target extension (e.g., '.txt'), if None uses same as candidates
        """
        ref_stem = reference_file.stem
        
        # Strategy 1: Exact name match
        if target_ext:
            exact_match = ref_stem + target_ext
        else:
            exact_match = reference_file.name
            
        for candidate in candidate_files:
            if candidate.name == exact_match:
                return candidate
        
        # Strategy 2: Remove "_shifted" from reference and match
        if "_shifted" in ref_stem:
            base_name = ref_stem.replace("_shifted", "")
            if target_ext:
                target_name = base_name + target_ext
            else:
                target_name = base_name + reference_file.suffix
                
            for candidate in candidate_files:
                if candidate.name == target_name:
                    return candidate
        
        # Strategy 3: Add "_shifted" to reference name
        if "_shifted" not in ref_stem:
            shifted_name = ref_stem + "_shifted"
            if target_ext:
                target_name = shifted_name + target_ext
            else:
                target_name = shifted_name + reference_file.suffix
                
            for candidate in candidate_files:
                if candidate.name == target_name:
                    return candidate
        
        return None
    
    def parse_txt_file(self, txt_file: Path, divided_mask_file: Path, dic_mask_file: Path) -> List[Dict]:
        """Parse a single TXT file and extract cell information"""
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f"⚠️  Unicode decode error for {txt_file.name}, skipping...")
            return []
        
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
                
                # Parse according to naming rules
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
                    'divided_mask_name': divided_mask_file.name,
                    'divided_mask_path': str(divided_mask_file),
                    'dic_mask_name': dic_mask_file.name,
                    'dic_mask_path': str(dic_mask_file),
                    'txt_file': str(txt_file),
                    'full_cell_id': full_cell_id,
                    'budding_pair_id': budding_pair_id,
                    'cell_type': cell_type,
                    'suffix': suffix
                })
                
            except ValueError as e:
                print(f"⚠️  Could not parse cell ID '{cell_id}' in {txt_file.name}: {e}")
                continue
        
        return extracted_data
    
    def stage1_data_parsing(self) -> pd.DataFrame:
        """
        Stage 1: Data Parsing and Structuring
        Parse file triplets and create a unified DataFrame
        """
        print("\n=== Stage 1: Data Parsing and Structuring ===")
        
        # Find matching file triplets
        matched_triplets = self.find_matching_files()
        
        if not matched_triplets:
            print("❌ No matching file triplets found!")
            return pd.DataFrame()
        
        all_data = []
        
        # Process each matched triplet
        for divided_mask_file, dic_mask_file, txt_file in matched_triplets:
            print(f"Processing: {txt_file.name}")
            try:
                cell_data = self.parse_txt_file(txt_file, divided_mask_file, dic_mask_file)
                all_data.extend(cell_data)
            except Exception as e:
                print(f"❌ Error processing {txt_file}: {e}")
        
        # Create master DataFrame
        self.master_df = pd.DataFrame(all_data)
        
        if len(self.master_df) > 0:
            print(f"\n✅ Created master DataFrame with {len(self.master_df)} entries")
            print(f"📋 Columns: {list(self.master_df.columns)}")
            
            # Show distribution statistics
            if 'cell_type' in self.master_df.columns:
                print(f"📊 Cell types distribution:")
                print(self.master_df['cell_type'].value_counts())
                
                # Calculate actual budding pairs
                mothers = len(self.master_df[self.master_df['cell_type'] == 'mother'])
                daughters = len(self.master_df[self.master_df['cell_type'] == 'daughter'])
                normal = len(self.master_df[self.master_df['cell_type'] == 'normal'])
                
                print(f"👩 Mother cells: {mothers}")
                print(f"👶 Daughter cells: {daughters}")
                print(f"🔗 Potential budding pairs: {min(mothers, daughters)}")
                
                if normal > 0:
                    print(f"⚠️  Warning: {normal} 'normal' cells found in divided_masks (should be 0!)")
            
            # Save master DataFrame
            output_file = self.output_dir / "master_data.csv"
            self.master_df.to_csv(output_file, index=False)
            print(f"💾 Saved master DataFrame to {output_file}")
            
            # Save processing summary
            self._save_processing_summary(matched_triplets)
            
        else:
            print("❌ No data found!")
            
        return self.master_df
    
    def _save_processing_summary(self, matched_triplets: List[Tuple[Path, Path, Path]]):
        """Save a summary of the processing"""
        
        # Calculate actual budding pairs count
        # Since ALL cells in divided_masks are budding cells, count mothers and daughters
        mothers = len(self.master_df[self.master_df['cell_type'] == 'mother']) if len(self.master_df) > 0 else 0
        daughters = len(self.master_df[self.master_df['cell_type'] == 'daughter']) if len(self.master_df) > 0 else 0
        actual_budding_pairs = min(mothers, daughters)  # The limiting factor
        
        summary = {
            'processing_date': datetime.now().isoformat(),
            'total_matched_triplets': len(matched_triplets),
            'total_cells_found': len(self.master_df),
            'mother_cells': mothers,
            'daughter_cells': daughters,
            'potential_budding_pairs': actual_budding_pairs,
            'note': 'ALL cells in divided_masks are budding cells',
            'matched_files': [
                {
                    'divided_mask_file': divided_mask_file.name,
                    'dic_mask_file': dic_mask_file.name,
                    'txt_file': txt_file.name,
                    'cells_found': len(self.master_df[self.master_df['divided_mask_name'] == divided_mask_file.name])
                }
                for divided_mask_file, dic_mask_file, txt_file in matched_triplets
            ]
        }
        
        import json
        summary_file = self.output_dir / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📄 Saved processing summary to {summary_file}")
    
    def find_budding_pairs(self) -> Dict:
        """Find valid budding pairs - since ALL cells in divided_masks are budding cells"""
        
        if self.master_df is None or len(self.master_df) == 0:
            print("❌ No master DataFrame found. Run stage1_data_parsing() first.")
            return {}
        
        print("🔍 Finding budding pairs...")
        print(f"📊 Total cells in divided_masks: {len(self.master_df)}")
        
        # Since ALL cells in divided_masks are budding cells,
        # we need to pair mothers and daughters based on a different logic
        
        budding_pairs = {}
        pair_counter = 1
        
        # Group by image file first
        grouped_by_image = self.master_df.groupby('divided_mask_name')
        
        for image_name, image_group in grouped_by_image:
            print(f"📷 Processing {image_name}: {len(image_group)} cells")
            
            # Get mothers and daughters in this image
            mothers = image_group[image_group['cell_type'] == 'mother'].copy()
            daughters = image_group[image_group['cell_type'] == 'daughter'].copy()
            
            print(f"   👩 Mothers: {len(mothers)}, 👶 Daughters: {len(daughters)}")
            
            # Method 1: Pair by closest full_cell_id values
            # This assumes mothers and daughters with similar IDs are paired
            for _, mother in mothers.iterrows():
                mother_id = mother['full_cell_id']
                
                # Find the closest daughter ID
                # Look for daughters with similar IDs (within the same hundred or close by)
                possible_daughters = daughters[
                    (daughters['full_cell_id'] // 100 == mother_id // 100) |
                    (abs(daughters['full_cell_id'] - mother_id) <= 1)
                ]
                
                if len(possible_daughters) > 0:
                    # Take the closest one
                    closest_daughter = possible_daughters.iloc[
                        (possible_daughters['full_cell_id'] - mother_id).abs().argmin()
                    ]
                    
                    budding_pairs[pair_counter] = {
                        'mother': mother,
                        'daughter': closest_daughter
                    }
                    
                    # Remove this daughter from further pairing
                    daughters = daughters[daughters['full_cell_id'] != closest_daughter['full_cell_id']]
                    pair_counter += 1
                else:
                    print(f"⚠️  No matching daughter found for mother {mother_id}")
        
        print(f"✅ Found {len(budding_pairs)} complete budding pairs")
        return budding_pairs
    
    def generate_training_sample(self, pair_id: int, pair_info: Dict) -> bool:
        """Generate a single training sample from a budding pair"""
        
        mother_info = pair_info['mother']
        daughter_info = pair_info['daughter']
        
        # Check if both cells are from the same image
        if mother_info['divided_mask_path'] != daughter_info['divided_mask_path']:
            print(f"⚠️  Mother and daughter from different images for pair {pair_id}")
            return False
        
        # Load masks (both dic_mask and divided_mask should be segmentation masks)
        input_mask_path = mother_info['dic_mask_path']  # Input segmentation mask
        divided_mask_path = mother_info['divided_mask_path']  # Labeled mask for creating ground truth
        
        try:
            input_mask = tifffile.imread(input_mask_path)  # Input segmentation mask
            labeled_mask = tifffile.imread(divided_mask_path)  # Segmentation mask (for labels)
        except Exception as e:
            print(f"❌ Error loading masks for pair {pair_id}: {e}")
            return False
        
        # Create mother and daughter masks from the labeled mask
        mother_id = mother_info['full_cell_id']
        daughter_id = daughter_info['full_cell_id']
        
        mother_mask = (labeled_mask == mother_id).astype(np.uint8)
        daughter_mask = (labeled_mask == daughter_id).astype(np.uint8)
        
        # Check if masks are non-empty
        if np.sum(mother_mask) == 0 or np.sum(daughter_mask) == 0:
            print(f"⚠️  Empty mask found for pair {pair_id}")
            return False
        
        # Create input X (combined binary mask of mother + daughter)
        combined_mask = ((mother_mask + daughter_mask) > 0).astype(np.uint8)
        
        # Create label y (three-class segmentation)
        y = np.zeros_like(labeled_mask, dtype=np.uint8)
        y[daughter_mask == 1] = 1  # daughter = 1
        y[mother_mask == 1] = 2    # mother = 2
        # background = 0 (already initialized)
        
        # Find bounding box and crop with padding
        coords = np.argwhere(combined_mask > 0)
        if len(coords) == 0:
            return False
            
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Add padding
        padding = 10
        y_min = max(0, y_min - padding)
        x_min = max(0, x_min - padding) 
        y_max = min(combined_mask.shape[0], y_max + padding)
        x_max = min(combined_mask.shape[1], x_max + padding)
        
        # Crop the regions from both input mask and label
        X_cropped = combined_mask[y_min:y_max, x_min:x_max]  # Binary mask input
        y_cropped = y[y_min:y_max, x_min:x_max]              # Label crop
        
        # Resize to target size
        X_resized = cv2.resize(X_cropped.astype(np.float32), self.target_size, interpolation=cv2.INTER_NEAREST)
        y_resized = cv2.resize(y_cropped, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Input is already binary (0 or 1), no need for normalization
        
        # Save the processed data
        sample_name = f"pair_{pair_id}"
        
        np.save(self.output_dir / "images" / f"{sample_name}.npy", X_resized)
        np.save(self.output_dir / "labels" / f"{sample_name}.npy", y_resized)
        
        print(f"✅ Generated training sample: {sample_name}")
        return True
    
    def stage2_generate_training_pairs(self) -> int:
        """
        Stage 2: Generate Training Sample Pairs
        Create (X, y) pairs for model training using DIC images as input
        """
        print("\n=== Stage 2: Generate Training Sample Pairs ===")
        
        # Find budding pairs
        budding_pairs = self.find_budding_pairs()
        print(f"🔗 Found {len(budding_pairs)} valid budding pairs")
        
        if len(budding_pairs) == 0:
            print("❌ No valid budding pairs found!")
            return 0
        
        # Generate training samples for each budding pair
        generated_samples = 0
        
        for pair_id, pair_info in budding_pairs.items():
            try:
                if self.generate_training_sample(pair_id, pair_info):
                    generated_samples += 1
            except Exception as e:
                print(f"❌ Error processing budding pair {pair_id}: {e}")
                continue
        
        print(f"\n✅ Successfully generated {generated_samples} training samples")
        
        # Save generation summary
        self._save_generation_summary(budding_pairs, generated_samples)
        
        return generated_samples
    
    def _save_generation_summary(self, budding_pairs: Dict, generated_samples: int):
        """Save summary of training sample generation"""
        
        summary = {
            'generation_date': datetime.now().isoformat(),
            'total_budding_pairs': len(budding_pairs),
            'successfully_generated': generated_samples,
            'target_image_size': self.target_size,
            'preprocessing_method': 'spatial_overlap_detection',
            'input_source': 'dic_masks - overlapping cells identified through spatial matching with divided_masks',
            'label_source': 'divided_masks - 3-class segmentation (background=0, daughter=1, mother=2)',
            'model_task': 'Binary mask to semantic segmentation (budding cell classification)',
            'budding_pairs_details': [
                {
                    'pair_id': int(pair_id),
                    'mother_cell_id': int(pair_info['mother']['full_cell_id']),
                    'daughter_cell_id': int(pair_info['daughter']['full_cell_id']),
                    'divided_mask_file': str(pair_info['mother']['divided_mask_name']),
                    'dic_mask_file': str(pair_info['mother']['dic_mask_name'])
                }
                for pair_id, pair_info in budding_pairs.items()
            ]
        }
        
        import json
        summary_file = self.output_dir / "generation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📄 Saved generation summary to {summary_file}")
    
    def run_full_pipeline(self) -> int:
        """Run the complete data processing pipeline"""
        
        print("🚀 Starting Full Data Processing Pipeline")
        print("=" * 50)
        
        # Stage 1: Parse data
        master_df = self.stage1_data_parsing()
        
        if master_df is None or len(master_df) == 0:
            print("❌ Pipeline stopped: No data found in Stage 1")
            return 0
        
        # Stage 2: Generate training pairs
        num_samples = self.stage2_generate_training_pairs()
        
        print("\n" + "=" * 50)
        print(f"✅ Pipeline Complete! Generated {num_samples} training samples")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📝 Input: Binary masks (mother + daughter regions)")
        print(f"🏷️  Labels: 3-class segmentation (background=0, daughter=1, mother=2)")
        
        return num_samples


def main():
    """Main function for command-line usage"""
    
    parser = argparse.ArgumentParser(description='Cell Division Data Processor')
    parser.add_argument('--data_root', default='data', 
                      help='Root directory containing divided_masks, dic_masks, and divided_outlines')
    parser.add_argument('--output_dir', default='processed_data',
                      help='Output directory for processed data')
    parser.add_argument('--target_size', nargs=2, type=int, default=[256, 256],
                      help='Target image size (width height)')
    parser.add_argument('--stage', choices=['1', '2', 'all'], default='all',
                      help='Which stage to run (1=parsing, 2=training pairs, all=both)')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = CellDataProcessor(
        data_root=args.data_root,
        output_dir=args.output_dir
    )
    processor.target_size = tuple(args.target_size)
    
    # Run requested stages
    if args.stage == '1':
        processor.stage1_data_parsing()
    elif args.stage == '2':
        if processor.master_df is None:
            # Try to load existing data
            master_file = processor.output_dir / "master_data.csv"
            if master_file.exists():
                processor.master_df = pd.read_csv(master_file)
                print(f"📂 Loaded existing master data from {master_file}")
            else:
                print("❌ No existing master data found. Run stage 1 first.")
                return
        processor.stage2_generate_training_pairs()
    else:  # 'all'
        processor.run_full_pipeline()


if __name__ == "__main__":
    main() 