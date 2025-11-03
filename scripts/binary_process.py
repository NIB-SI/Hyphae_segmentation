#!/usr/bin/env python3
"""
Process images using trained classifiers and measure hyphae.
Uses config.ini for all settings.
"""

import os
import configparser
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
from skimage import io, color, morphology, exposure, measure
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from scipy import ndimage

from feature_extraction import FeatureExtractor


def load_config(config_path='config.ini'):
    """Load configuration."""
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def classify_images(image_folder, bad_clf, semi_clf, extractor):
    """
    Extract features and classify all images in folder.
    
    Returns:
        DataFrame with columns: filename, category, confidence, bad_prob, semi_prob
    """
    image_files = [f for f in os.listdir(image_folder) 
                   if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    if not image_files:
        return pd.DataFrame()
    
    # Extract features
    features_list = []
    filenames = []
    
    for img_file in tqdm(image_files, desc="Extracting features"):
        try:
            img_path = os.path.join(image_folder, img_file)
            features = extractor.extract_all_features(img_path)
            features_list.append(features)
            filenames.append(img_file)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    if not features_list:
        return pd.DataFrame()
    
    X = pd.DataFrame(features_list, index=filenames)
    
    # Classify
    bad_pred = bad_clf.predict(X)
    bad_prob = bad_clf.predict_proba(X)[:, 1]
    
    semi_pred = semi_clf.predict(X)
    semi_prob = semi_clf.predict_proba(X)[:, 1]
    
    # Combine predictions: bad > semi_good > good
    categories = []
    confidences = []
    
    for i in range(len(X)):
        if bad_pred[i] == 1:
            categories.append('bad')
            confidences.append(bad_prob[i])
        elif semi_pred[i] == 1:
            categories.append('semi_good')
            confidences.append(semi_prob[i])
        else:
            categories.append('good')
            confidences.append(1.0 - max(bad_prob[i], semi_prob[i]))
    
    results = pd.DataFrame({
        'filename': filenames,
        'category': categories,
        'confidence': confidences,
        'bad_probability': bad_prob,
        'semi_good_probability': semi_prob
    })
    
    return results


def create_circular_mask(h, w, center=None, radius=None, border_removal=0):
    """Create circular mask with optional border removal."""
    if center is None:
        center = (int(w/2), int(h/2))
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])
    
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    
    if border_removal > 0:
        outer_mask = dist_from_center <= radius
        inner_mask = dist_from_center <= (radius - border_removal)
        return outer_mask, inner_mask
    else:
        mask = dist_from_center <= radius
        return mask, mask


def enhance_contrast(image, low, high):
    """Enhance contrast using percentile rescaling."""
    p_low, p_high = np.percentile(image, (low * 100, high * 100))
    return exposure.rescale_intensity(image, in_range=(p_low, p_high))


def apply_enhanced_sato_filter(image, scale_range, intensity_factor):
    """Enhanced vessel detection with intensity reduction."""
    from skimage.filters import sato
    vessel_img = sato(image, sigmas=range(*scale_range), black_ridges=True)
    
    max_intensity = vessel_img.max()
    intensity_threshold = max_intensity * (1 - intensity_factor)
    vessel_img[vessel_img < intensity_threshold] *= intensity_factor
    vessel_img = vessel_img / vessel_img.max()
    
    return vessel_img


def enhance_vessel_response(vessel_img, category='good'):
    """Enhancement using local statistics and connectivity."""
    from skimage import morphology, measure
    from scipy import ndimage
    
    if category == 'semi_good':
        VESSEL_ENHANCEMENT_ITERATIONS = 2
        ENHANCEMENT_FACTOR = 1.3
        sensitivity_factor = 0.4
        min_area = 4
    else:
        VESSEL_ENHANCEMENT_ITERATIONS = 2
        ENHANCEMENT_FACTOR = 1.2
        sensitivity_factor = 0.5
        min_area = 5
    
    enhanced = vessel_img.copy()
    local_mean = ndimage.uniform_filter(vessel_img, size=15)
    local_std = np.sqrt(np.maximum(ndimage.uniform_filter(vessel_img**2, size=15) - local_mean**2, 1e-10))
    
    for iteration in range(VESSEL_ENHANCEMENT_ITERATIONS):
        binary = enhanced > (local_mean + sensitivity_factor * local_std)
        labels = morphology.label(binary)
        
        for region in measure.regionprops(labels):
            if region.area > min_area:
                coords = region.coords
                current_factor = ENHANCEMENT_FACTOR * (1 + 0.1 * iteration) if category == 'semi_good' else ENHANCEMENT_FACTOR
                enhanced[coords[:, 0], coords[:, 1]] *= current_factor
    
    if category == 'semi_good':
        enhanced = enhanced ** 0.9
        from skimage.morphology import white_tophat, disk
        tophat = white_tophat(enhanced, disk(1))
        enhanced = enhanced + 0.1 * tophat
    
    enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())
    return enhanced


def threshold_image(image, threshold_offset):
    """Threshold image using Otsu with offset."""
    from skimage.filters import threshold_otsu
    otsu_thresh = threshold_otsu(image)
    adjusted_thresh = otsu_thresh * threshold_offset
    return image > adjusted_thresh


def clean_binary_image(binary_image, min_size):
    """Clean binary image with skeletonization pre-thinning."""
    from skimage import morphology
    cleaned = morphology.remove_small_objects(binary_image, min_size=min_size)
    cleaned = morphology.closing(cleaned, morphology.disk(2))
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=64)
    cleaned = morphology.skeletonize(cleaned)
    cleaned = morphology.dilation(cleaned, morphology.disk(1))
    return cleaned


def determine_parameters(image_category):
    """Determine processing parameters based on image category."""
    base_params = {
        'contrast_low': 0.01,
        'contrast_high': 0.7,
        'length_threshold': 100,
        'threshold_offset': 1.1,
        'intensity_factor': 0.9,
        'scale_range': (1, 3),
        'border_removal': 4,
        'category': image_category
    }
    
    if image_category == 'semi_good':
        return {
            **base_params,
            'threshold_offset': 0.85,
            'intensity_factor': 0.94,
            'contrast_high': 0.8,
            'contrast_low': 0.008,
            'vessel_boost': True,
        }
    elif image_category == 'bad':
        return {
            **base_params,
            'threshold_offset': 1.23,
            'length_threshold': 120,
            'intensity_factor': 0.85,
            'scale_range': (1, 4),
            'contrast_high': 0.75,
        }
    else:
        return base_params


def measure_hyphae(image_path, category, output_folder):
    """
    Process image and measure hyphae length.
    Full pipeline: grayscale → vessel detection → binarization → skeletonization → measurement.
    """
    from skimage import io, color, morphology, exposure
    from skimage.util import img_as_ubyte
    from skimage.color import rgb2gray
    
    # Get parameters based on category
    params = determine_parameters(category)
    params['output_folder'] = output_folder
    
    # Load image
    img = io.imread(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create output subfolder
    img_output_folder = os.path.join(output_folder, base_name)
    os.makedirs(img_output_folder, exist_ok=True)
    
    # Create circular masks
    outer_mask, inner_mask = create_circular_mask(img.shape[0], img.shape[1], 
                                                  border_removal=params['border_removal'])
    
    # Step 1: Grayscale conversion
    if img.ndim == 3:
        if img.shape[2] == 4:
            white_background = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
            alpha = img[:, :, 3] / 255.0
            rgb_channels = img[:, :, :3]
            blended = (1 - alpha[:, :, np.newaxis]) * white_background + alpha[:, :, np.newaxis] * rgb_channels
            img_gray = rgb2gray(blended.astype(np.uint8))
        else:
            img_gray = rgb2gray(img)
    else:
        img_gray = img
    
    img_gray = img_gray * outer_mask
    io.imsave(os.path.join(img_output_folder, f"{base_name}_1gray.png"), img_as_ubyte(img_gray))
    
    # Step 2: Contrast adjustment
    img_adjusted = enhance_contrast(img_gray, params['contrast_low'], params['contrast_high'])
    if params.get('pre_enhance', False):
        img_adjusted = exposure.equalize_adapthist(img_adjusted, clip_limit=0.02)
    io.imsave(os.path.join(img_output_folder, f"{base_name}_2adjust.png"), img_as_ubyte(img_adjusted))
    
    # Step 3: Vessel detection (Sato filter)
    img_vessel = apply_enhanced_sato_filter(img_adjusted, params['scale_range'], params['intensity_factor'])
    io.imsave(os.path.join(img_output_folder, f"{base_name}_3vessel.png"), img_as_ubyte(img_vessel))
    
    # Step 4: Vessel enhancement
    img_vessel_enhanced = enhance_vessel_response(img_vessel, category=params['category'])
    
    if params.get('vessel_boost', False):
        img_vessel_enhanced = exposure.equalize_adapthist(img_vessel_enhanced, clip_limit=0.01)
        img_vessel_enhanced = exposure.adjust_gamma(img_vessel_enhanced, gamma=0.95)
        p2, p98 = np.percentile(img_vessel_enhanced, (2, 98))
        img_vessel_enhanced = exposure.rescale_intensity(img_vessel_enhanced, in_range=(p2, p98))
        img_vessel_enhanced = np.clip(img_vessel_enhanced * 1.05, 0, 1)
        img_vessel_enhanced = (img_vessel_enhanced - img_vessel_enhanced.min()) / (img_vessel_enhanced.max() - img_vessel_enhanced.min())
    
    io.imsave(os.path.join(img_output_folder, f"{base_name}_3vessel_enhanced.png"), img_as_ubyte(img_vessel_enhanced))
    
    # Step 5: Binarization
    img_bw = threshold_image(img_vessel_enhanced, params['threshold_offset'])
    io.imsave(os.path.join(img_output_folder, f"{base_name}_4binary.png"), img_as_ubyte(img_bw))
    
    # Step 6: Clean binary
    img_clean = clean_binary_image(img_bw, params['length_threshold'])
    io.imsave(os.path.join(img_output_folder, f"{base_name}_5clean.png"), img_as_ubyte(img_clean))
    
    # Step 7: Skeletonization
    img_skeleton = morphology.skeletonize(img_clean)
    io.imsave(os.path.join(img_output_folder, f"{base_name}_6skeleton_full.png"), img_as_ubyte(img_skeleton))
    
    # Apply inner mask to remove border
    img_skeleton_no_border = img_skeleton & inner_mask
    io.imsave(os.path.join(img_output_folder, f"{base_name}_6skeleton_no_border.png"), img_as_ubyte(img_skeleton_no_border))
    
    # Measure hyphae length (count skeleton pixels)
    hyphae_length = np.sum(img_skeleton_no_border)
    
    return {
        'hyphae_length': hyphae_length
    }


def process_folder(input_folder, output_folder, bad_clf, semi_clf, extractor, subfolder_name=''):
    """
    Process all images in a folder.
    
    Args:
        input_folder: Path to cut images
        output_folder: Path to save results
        subfolder_name: Name to prepend to filenames (e.g., 'experiment1/')
    
    Returns:
        DataFrame with results
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if classifications already exist
    category_file = os.path.join(output_folder, 'classifications.csv')
    
    if os.path.exists(category_file):
        print(f"  Loading existing classifications from {category_file}")
        classifications = pd.read_csv(category_file)
    else:
        # Classify images
        print(f"Processing: {subfolder_name if subfolder_name else input_folder}")
        classifications = classify_images(input_folder, bad_clf, semi_clf, extractor)
        
        if classifications.empty:
            print(f"  No images found or processed")
            return pd.DataFrame()
        
        # Save classifications
        classifications.to_csv(category_file, index=False)
    
    # Measure hyphae for each image
    results = []
    for _, row in tqdm(classifications.iterrows(), total=len(classifications), desc="Measuring"):
        img_path = os.path.join(input_folder, row['filename'])
        
        # Measure hyphae
        try:
            measurements = measure_hyphae(img_path, row['category'], output_folder)
        except Exception as e:
            print(f"Error measuring {row['filename']}: {e}")
            measurements = {'hyphae_length': 0}
        
        # Combine results
        result = {
            'image': f"{subfolder_name}{row['filename']}" if subfolder_name else row['filename'],
            'category': row['category'],
            'confidence': row['confidence'],
            **measurements
        }
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    print(f"  Processed {len(results_df)} images")
    print(f"  Categories: {results_df['category'].value_counts().to_dict()}")
    
    return results_df


def main():
    """Process a single folder (called by process_batch.py)."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', help='Folder with cut images')
    parser.add_argument('output_folder', help='Output folder for results')
    parser.add_argument('--subfolder-name', default='', help='Name to prepend to filenames')
    parser.add_argument('--config', default='config.ini', help='Config file path')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load models
    bad_clf = joblib.load(config['PATHS']['bad_model'])
    semi_clf = joblib.load(config['PATHS']['semi_model'])
    
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
    
    # Process folder
    results = process_folder(
        args.input_folder,
        args.output_folder,
        bad_clf,
        semi_clf,
        extractor,
        args.subfolder_name
    )
    
    # Save results
    if not results.empty:
        results_file = os.path.join(args.output_folder, 'measurements.csv')
        results.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()