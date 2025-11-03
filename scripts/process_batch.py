#!/usr/bin/env python3
"""
Batch processing of multiple subfolders.
Handles cutting, classification, and measurement for full dataset.
"""

import os
import configparser
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from cut_images import cut_folder
from binary_process import classify_images, measure_hyphae
from feature_extraction import FeatureExtractor


def load_config(config_path='config.ini'):
    """Load configuration."""
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def find_subfolders_needing_processing(data_dir):
    """
    Find subfolders with images that need cutting or processing.
    
    Returns:
        List of (subfolder_path, needs_cutting) tuples
    """
    subfolders = []
    
    for item in sorted(os.listdir(data_dir)):
        subfolder_path = os.path.join(data_dir, item)
        
        if not os.path.isdir(subfolder_path):
            continue
        
        # Check for images in main folder
        images = [f for f in os.listdir(subfolder_path) 
                 if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        if not images:
            continue  # No images, skip
        
        cut_folder_path = os.path.join(subfolder_path, 'cut')
        
        # Check if cutting needed
        needs_cutting = False
        if not os.path.exists(cut_folder_path):
            needs_cutting = True
        else:
            # Check if all images are cut
            cut_images = [f for f in os.listdir(cut_folder_path) if f.endswith('.png')]
            if len(cut_images) < len(images):
                needs_cutting = True
        
        subfolders.append((subfolder_path, needs_cutting))
    
    return subfolders


def process_all_subfolders(data_dir, results_dir, bad_clf, semi_clf, extractor):
    """
    Process all subfolders: cut if needed, classify, measure.
    
    Returns:
        Combined DataFrame with all results
    """
    print("="*70)
    print(f"BATCH PROCESSING: {data_dir}")
    print("="*70)
    
    # Find subfolders
    subfolders = find_subfolders_needing_processing(data_dir)
    
    if not subfolders:
        print("No subfolders with images found")
        return pd.DataFrame()
    
    print(f"Found {len(subfolders)} subfolders to process")
    
    # Process each subfolder
    all_results = []
    
    for subfolder_path, needs_cutting in subfolders:
        subfolder_name = os.path.basename(subfolder_path)
        print(f"\n{'='*70}")
        print(f"Processing: {subfolder_name}")
        print(f"{'='*70}")
        
        cut_folder_path = os.path.join(subfolder_path, 'cut')
        results_subfolder = os.path.join(results_dir, subfolder_name)
        os.makedirs(results_subfolder, exist_ok=True)
        
        # Step 1: Cut images if needed
        if needs_cutting:
            print(f"Cutting images...")
            count = cut_folder(subfolder_path, cut_folder_path)
            print(f"  Cut {count} images")
        else:
            print(f"  Using existing cut images")
        
        # Step 2: Check if cut folder has images
        cut_images = [f for f in os.listdir(cut_folder_path) if f.endswith('.png')]
        if not cut_images:
            print(f"  No cut images found, skipping")
            continue
        
        print(f"  Processing {len(cut_images)} images...")
        
        # Step 3: Check if classifications already exist
        class_file = os.path.join(results_subfolder, 'classifications.csv')
        
        if os.path.exists(class_file):
            print(f"  Found existing classifications, loading...")
            classifications = pd.read_csv(class_file)
            print(f"  Loaded {len(classifications)} classifications")
        else:
            print(f"  Classifying images...")
            classifications = classify_images(cut_folder_path, bad_clf, semi_clf, extractor)
            
            if classifications.empty:
                print(f"  No images could be classified")
                continue
            
            # Save classifications
            classifications.to_csv(class_file, index=False)
            print(f"  Saved classifications to {class_file}")
        
        # Step 4: Measure hyphae for each image
        print(f"  Measuring hyphae...")
        for _, row in classifications.iterrows():
            img_path = os.path.join(cut_folder_path, row['filename'])
            
            # Measure
            try:
                measurements = measure_hyphae(img_path, row['category'], results_subfolder)
            except Exception as e:
                print(f"  Error measuring {row['filename']}: {e}")
                measurements = {'hyphae_length': 0}
            
            # Store result with subfolder prefix
            result = {
                'image': f"{subfolder_name}/{row['filename']}",
                'category': row['category'],
                'confidence': row['confidence'],
                **measurements
            }
            all_results.append(result)
        
        # Summary
        print(f"  ✓ Completed {len(classifications)} images")
        print(f"    Categories: {classifications['category'].value_counts().to_dict()}")
    
    # Combine all results
    if not all_results:
        print("\nNo results generated")
        return pd.DataFrame()
    
    combined_df = pd.DataFrame(all_results)
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total images processed: {len(combined_df)}")
    print(f"\nOverall category distribution:")
    print(combined_df['category'].value_counts())
    
    return combined_df


def main():
    # Load config
    config = load_config()
    
    data_dir = config['PATHS']['data_dir']
    results_dir = config['PATHS']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Data directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    
    # Load models
    print(f"\nLoading models...")
    bad_clf = joblib.load(config['PATHS']['bad_model'])
    semi_clf = joblib.load(config['PATHS']['semi_model'])
    print(f"  ✓ Models loaded")
    
    # Initialize feature extractor
    gaussian_scales = [float(x.strip()) for x in config['FEATURE_EXTRACTION']['gaussian_scales'].split(',')]
    threshold_factors = [float(x.strip()) for x in config['FEATURE_EXTRACTION']['threshold_factors'].split(',')]
    
    extractor = FeatureExtractor(
        resize_shape=(
            int(config['FEATURE_EXTRACTION']['resize_height']),
            int(config['FEATURE_EXTRACTION']['resize_width'])
        ),
        gaussian_scales=gaussian_scales,
        threshold_factors=threshold_factors
    )
    print(f"  ✓ Feature extractor initialized")
    
    # Process all subfolders
    combined_results = process_all_subfolders(data_dir, results_dir, bad_clf, semi_clf, extractor)
    
    if combined_results.empty:
        print("\nNo results to save")
        return
    
    # Save combined results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(results_dir, f'combined_results_{timestamp}.tsv')
    combined_results.to_csv(output_file, sep='\t', index=False)
    
    print(f"\n{'='*70}")
    print(f"RESULTS SAVED")
    print(f"{'='*70}")
    print(f"Combined results: {output_file}")
    print(f"  - Total rows: {len(combined_results)}")
    print(f"  - Columns: {list(combined_results.columns)}")
    
    # Save summary statistics
    summary_file = os.path.join(results_dir, f'summary_{timestamp}.txt')
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BATCH PROCESSING SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total images: {len(combined_results)}\n\n")
        f.write("Category distribution:\n")
        f.write(str(combined_results['category'].value_counts()) + "\n\n")
        f.write("Mean confidence by category:\n")
        f.write(str(combined_results.groupby('category')['confidence'].mean()) + "\n")
    
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()