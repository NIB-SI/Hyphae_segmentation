#!/usr/bin/env python3
"""
Comprehensive Feature Extraction Module for Hyphae Image Analysis

This module extracts a rich set of image features for classification, combining:
1. Original features from the existing pipeline (intensity, edge, texture, morphology)
2. Ilastik-inspired multi-scale features (Gaussian, LoG, Hessian, Structure Tensor)

Features are computed consistently for both training and inference.
Can be run standalone to extract and save features to TSV files.

Author: Image Analysis Pipeline
Date: 2025
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Image processing imports
from skimage import io, color, filters, feature, morphology, exposure, util, measure
from skimage.transform import resize
from scipy import ndimage
from scipy.stats import skew, kurtosis

# Version compatibility for peak detection
try:
    from skimage.feature import peak_local_maxima
except ImportError:
    peak_local_maxima = None


class FeatureExtractor:
    """
    Comprehensive feature extraction for microscopy images.
    
    Extracts ~150+ features across multiple categories:
    - Basic intensity statistics
    - Multi-scale Gaussian features (Ilastik-inspired)
    - Edge and gradient features
    - Texture features (LBP, GLCM)
    - Morphological features (shape, connectivity)
    - Hessian-based features (multi-scale)
    - Structure tensor features (multi-scale)
    - Laplacian of Gaussian features (multi-scale)
    - Frequency domain features
    """
    
    def __init__(self, 
                 resize_shape: Tuple[int, int] = (512, 512),
                 gaussian_scales: List[float] = [0.7, 1.0, 1.6, 3.5, 5.0, 10.0],
                 threshold_factors: List[float] = [0.8, 1.0, 1.2]):
        """
        Initialize feature extractor with parameters.
        
        Args:
            resize_shape: Target size for image resizing (height, width)
            gaussian_scales: Sigma values for multi-scale Gaussian features (Ilastik-inspired)
            threshold_factors: Threshold multipliers for morphological analysis
        """
        self.resize_shape = resize_shape
        self.gaussian_scales = gaussian_scales
        self.threshold_factors = threshold_factors
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image to grayscale float format.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed grayscale image as float [0, 1]
        """
        # Load the image
        image = io.imread(image_path)
        
        # Convert RGBA to RGB if needed (handle transparency)
        if image.ndim == 3 and image.shape[2] == 4:
            white_background = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
            alpha = image[:, :, 3] / 255.0
            rgb_channels = image[:, :, :3]
            blended = (1 - alpha[:, :, np.newaxis]) * white_background + alpha[:, :, np.newaxis] * rgb_channels
            image = blended.astype(np.uint8)
        
        # Convert to grayscale if needed
        if image.ndim == 3:
            image = color.rgb2gray(image)
        
        # Normalize to float [0, 1]
        image = util.img_as_float(image)
        
        # Resize for consistent processing
        resized_img = resize(image, self.resize_shape, anti_aliasing=True)
        
        return resized_img
    
    def extract_intensity_features(self, img: np.ndarray) -> Dict[str, float]:
        """Extract basic intensity statistics."""
        features = {}
        
        # Basic statistics
        features['mean_intensity'] = np.mean(img)
        features['std_intensity'] = np.std(img)
        features['min_intensity'] = np.min(img)
        features['max_intensity'] = np.max(img)
        features['intensity_range'] = features['max_intensity'] - features['min_intensity']
        features['median_intensity'] = np.median(img)
        
        # Percentiles for distribution characterization
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        for p in percentiles:
            features[f'p{p}'] = np.percentile(img, p)
        
        # Distribution shape
        features['intensity_skewness'] = skew(img.flatten())
        features['intensity_kurtosis'] = kurtosis(img.flatten())
        
        # Coefficient of variation (normalized std)
        features['intensity_cv'] = features['std_intensity'] / (features['mean_intensity'] + 1e-10)
        
        return features
    
    def extract_multiscale_gaussian_features(self, img: np.ndarray) -> Dict[str, float]:
        """
        Extract Ilastik-inspired multi-scale Gaussian smoothed features.
        Captures information at different spatial scales.
        """
        features = {}
        
        for sigma in self.gaussian_scales:
            # Gaussian smoothing at this scale
            smoothed = filters.gaussian(img, sigma=sigma, preserve_range=True)
            
            # Statistics of smoothed image
            prefix = f'gauss_s{sigma:.1f}_'
            features[f'{prefix}mean'] = np.mean(smoothed)
            features[f'{prefix}std'] = np.std(smoothed)
            features[f'{prefix}min'] = np.min(smoothed)
            features[f'{prefix}max'] = np.max(smoothed)
            features[f'{prefix}range'] = features[f'{prefix}max'] - features[f'{prefix}min']
            
        return features
    
    def extract_edge_gradient_features(self, img: np.ndarray) -> Dict[str, float]:
        """Extract edge detection and gradient features."""
        features = {}
        
        # Sobel edges (horizontal and vertical)
        sobel_h = filters.sobel_h(img)
        sobel_v = filters.sobel_v(img)
        sobel_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
        
        features['sobel_mean'] = np.mean(sobel_magnitude)
        features['sobel_std'] = np.std(sobel_magnitude)
        features['sobel_max'] = np.max(sobel_magnitude)
        features['strong_edges_count'] = np.sum(sobel_magnitude > np.percentile(sobel_magnitude, 90))
        features['strong_edges_ratio'] = features['strong_edges_count'] / img.size
        
        # Gradient direction variance
        gradient_direction = np.arctan2(sobel_v, sobel_h)
        mean_cos = np.mean(np.cos(2 * gradient_direction))
        mean_sin = np.mean(np.sin(2 * gradient_direction))
        features['gradient_direction_variance'] = 1 - np.sqrt(mean_cos**2 + mean_sin**2)
        
        # Prewitt edges
        prewitt_h = filters.prewitt_h(img)
        prewitt_v = filters.prewitt_v(img)
        prewitt_magnitude = np.sqrt(prewitt_h**2 + prewitt_v**2)
        features['prewitt_mean'] = np.mean(prewitt_magnitude)
        features['prewitt_std'] = np.std(prewitt_magnitude)
        
        # Scharr edges (more accurate derivative approximation)
        scharr_h = filters.scharr_h(img)
        scharr_v = filters.scharr_v(img)
        scharr_magnitude = np.sqrt(scharr_h**2 + scharr_v**2)
        features['scharr_mean'] = np.mean(scharr_magnitude)
        features['scharr_std'] = np.std(scharr_magnitude)
        
        return features
    
    def extract_texture_features(self, img: np.ndarray) -> Dict[str, float]:
        """Extract texture features (LBP, GLCM approximation)."""
        features = {}
        
        # Local Binary Pattern (LBP)
        try:
            from skimage.feature import local_binary_pattern
            img_uint8 = util.img_as_ubyte(img)
            lbp = local_binary_pattern(img_uint8, P=8, R=1, method='uniform')
            features['lbp_variance'] = np.var(lbp)
            features['lbp_mean'] = np.mean(lbp)
            
            # LBP histogram features
            lbp_hist, _ = np.histogram(lbp.flatten(), bins=10, density=True)
            features['lbp_uniformity'] = np.sum(lbp_hist**2)
            features['lbp_entropy'] = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
        except Exception:
            features['lbp_variance'] = 0
            features['lbp_mean'] = 0
            features['lbp_uniformity'] = 0
            features['lbp_entropy'] = 0
        
        # GLCM approximation (directional contrasts)
        diff_h = np.abs(np.diff(img, axis=1))
        diff_v = np.abs(np.diff(img, axis=0))
        features['horizontal_contrast'] = np.var(diff_h)
        features['vertical_contrast'] = np.var(diff_v)
        features['horizontal_homogeneity'] = 1 / (1 + np.mean(diff_h))
        features['vertical_homogeneity'] = 1 / (1 + np.mean(diff_v))
        features['contrast_anisotropy'] = np.abs(features['horizontal_contrast'] - features['vertical_contrast'])
        
        return features
    
    def extract_morphological_features(self, img: np.ndarray) -> Dict[str, float]:
        """
        Extract morphological features at multiple threshold levels.
        Critical for distinguishing bacterial (round) vs hyphal (elongated) structures.
        """
        features = {}
        
        # Compute Otsu threshold as baseline
        try:
            otsu_thresh = filters.threshold_otsu(img)
        except Exception:
            otsu_thresh = np.mean(img)
        
        # Analyze at different threshold levels
        for thresh_factor in self.threshold_factors:
            thresh = otsu_thresh * thresh_factor
            binary = img > thresh
            
            # Connected components analysis
            labeled = measure.label(binary)
            props = measure.regionprops(labeled)
            
            prefix = f'thresh_{int(thresh_factor*10)}_'
            features[f'{prefix}num_objects'] = len(props)
            features[f'{prefix}foreground_ratio'] = np.sum(binary) / binary.size
            
            if len(props) > 0:
                # Size features
                areas = [prop.area for prop in props]
                features[f'{prefix}mean_area'] = np.mean(areas)
                features[f'{prefix}std_area'] = np.std(areas)
                features[f'{prefix}max_area'] = np.max(areas)
                features[f'{prefix}total_area'] = np.sum(areas)
                
                # Shape features (CRITICAL for bacteria vs hyphae distinction)
                eccentricities = [prop.eccentricity for prop in props]
                features[f'{prefix}mean_eccentricity'] = np.mean(eccentricities)
                features[f'{prefix}std_eccentricity'] = np.std(eccentricities)
                features[f'{prefix}max_eccentricity'] = np.max(eccentricities)
                
                solidities = [prop.solidity for prop in props]
                features[f'{prefix}mean_solidity'] = np.mean(solidities)
                features[f'{prefix}std_solidity'] = np.std(solidities)
                
                # Aspect ratios (hyphae have higher ratios)
                aspect_ratios = [prop.major_axis_length / (prop.minor_axis_length + 1e-10) 
                               for prop in props]
                features[f'{prefix}mean_aspect_ratio'] = np.mean(aspect_ratios)
                features[f'{prefix}max_aspect_ratio'] = np.max(aspect_ratios)
                features[f'{prefix}std_aspect_ratio'] = np.std(aspect_ratios)
                
                # Perimeter to area ratios (complexity measure)
                perimeter_area_ratios = [prop.perimeter / (prop.area + 1e-10) 
                                        for prop in props]
                features[f'{prefix}mean_perimeter_area_ratio'] = np.mean(perimeter_area_ratios)
                
                # Count elongated objects (likely hyphae)
                elongated_objects = sum(1 for ecc in eccentricities if ecc > 0.8)
                features[f'{prefix}elongated_objects'] = elongated_objects
                features[f'{prefix}elongated_ratio'] = elongated_objects / len(props) if len(props) > 0 else 0
                
                # Count round objects (likely bacteria)
                round_objects = sum(1 for ecc in eccentricities if ecc < 0.5)
                features[f'{prefix}round_objects'] = round_objects
                features[f'{prefix}round_ratio'] = round_objects / len(props) if len(props) > 0 else 0
                
                # Convexity measures
                convex_areas = [prop.convex_area for prop in props]
                features[f'{prefix}mean_convexity'] = np.mean([prop.area / (prop.convex_area + 1e-10) 
                                                               for prop in props])
                
            else:
                # No objects detected - set all to zero
                zero_features = ['mean_area', 'std_area', 'max_area', 'total_area',
                               'mean_eccentricity', 'std_eccentricity', 'max_eccentricity',
                               'mean_solidity', 'std_solidity',
                               'mean_aspect_ratio', 'max_aspect_ratio', 'std_aspect_ratio',
                               'mean_perimeter_area_ratio',
                               'elongated_objects', 'elongated_ratio',
                               'round_objects', 'round_ratio', 'mean_convexity']
                for feat_name in zero_features:
                    features[f'{prefix}{feat_name}'] = 0
        
        return features
    
    def extract_hessian_features(self, img: np.ndarray) -> Dict[str, float]:
        """
        Extract Hessian-based features at multiple scales (Ilastik-inspired).
        Hessian eigenvalues are excellent for detecting line-like structures (hyphae).
        """
        features = {}
        
        for sigma in self.gaussian_scales:
            try:
                # Compute Hessian matrix eigenvalues
                hessian_eigvals = feature.hessian_matrix_eigvals(
                    feature.hessian_matrix(img, sigma=sigma, order='rc')
                )
                
                # hessian_eigvals returns (eig1, eig2) where eig1 >= eig2
                eig_large = hessian_eigvals[0]  # Larger eigenvalue
                eig_small = hessian_eigvals[1]  # Smaller eigenvalue
                
                prefix = f'hessian_s{sigma:.1f}_'
                
                # Statistics of larger eigenvalue (indicates strength of line-like structures)
                features[f'{prefix}eig1_mean'] = np.mean(np.abs(eig_large))
                features[f'{prefix}eig1_std'] = np.std(eig_large)
                features[f'{prefix}eig1_max'] = np.max(np.abs(eig_large))
                
                # Statistics of smaller eigenvalue
                features[f'{prefix}eig2_mean'] = np.mean(np.abs(eig_small))
                features[f'{prefix}eig2_std'] = np.std(eig_small)
                
                # Ratio of eigenvalues (linearity measure)
                ratio = np.abs(eig_large) / (np.abs(eig_small) + 1e-10)
                features[f'{prefix}eig_ratio_mean'] = np.mean(ratio)
                features[f'{prefix}eig_ratio_max'] = np.max(ratio)
                
                # Frangi vesselness-like measure (for tubular structures)
                vesselness = np.abs(eig_small) * (1 - np.exp(-ratio**2 / 2))
                features[f'{prefix}vesselness_mean'] = np.mean(vesselness)
                features[f'{prefix}vesselness_max'] = np.max(vesselness)
                
            except Exception as e:
                # If Hessian computation fails, set to zero
                prefix = f'hessian_s{sigma:.1f}_'
                for suffix in ['eig1_mean', 'eig1_std', 'eig1_max', 'eig2_mean', 'eig2_std',
                              'eig_ratio_mean', 'eig_ratio_max', 'vesselness_mean', 'vesselness_max']:
                    features[f'{prefix}{suffix}'] = 0
        
        return features
    
    def extract_structure_tensor_features(self, img: np.ndarray) -> Dict[str, float]:
        """
        Extract Structure Tensor features at multiple scales (Ilastik-inspired).
        Structure tensor eigenvalues capture local orientation and texture.
        """
        features = {}
        
        for sigma in self.gaussian_scales:
            try:
                # Compute structure tensor
                Axx, Axy, Ayy = feature.structure_tensor(img, sigma=sigma)
                
                # Compute eigenvalues of structure tensor
                # For 2x2 symmetric matrix [[Axx, Axy], [Axy, Ayy]]
                trace = Axx + Ayy
                det = Axx * Ayy - Axy**2
                discriminant = np.sqrt(np.maximum(trace**2 - 4*det, 0))
                
                eig1 = (trace + discriminant) / 2  # Larger eigenvalue
                eig2 = (trace - discriminant) / 2  # Smaller eigenvalue
                
                prefix = f'struct_s{sigma:.1f}_'
                
                # Eigenvalue statistics
                features[f'{prefix}eig1_mean'] = np.mean(eig1)
                features[f'{prefix}eig1_std'] = np.std(eig1)
                features[f'{prefix}eig1_max'] = np.max(eig1)
                
                features[f'{prefix}eig2_mean'] = np.mean(eig2)
                features[f'{prefix}eig2_std'] = np.std(eig2)
                
                # Coherence (measure of local anisotropy)
                coherence = (eig1 - eig2) / (eig1 + eig2 + 1e-10)
                features[f'{prefix}coherence_mean'] = np.mean(coherence)
                features[f'{prefix}coherence_std'] = np.std(coherence)
                
                # Energy (sum of eigenvalues)
                energy = eig1 + eig2
                features[f'{prefix}energy_mean'] = np.mean(energy)
                features[f'{prefix}energy_std'] = np.std(energy)
                
            except Exception:
                prefix = f'struct_s{sigma:.1f}_'
                for suffix in ['eig1_mean', 'eig1_std', 'eig1_max', 'eig2_mean', 'eig2_std',
                              'coherence_mean', 'coherence_std', 'energy_mean', 'energy_std']:
                    features[f'{prefix}{suffix}'] = 0
        
        return features
    
    def extract_laplacian_features(self, img: np.ndarray) -> Dict[str, float]:
        """
        Extract Laplacian of Gaussian (LoG) features at multiple scales (Ilastik-inspired).
        LoG is excellent for blob detection and spotting local intensity variations.
        """
        features = {}
        
        for sigma in self.gaussian_scales:
            try:
                # Laplacian of Gaussian
                log_img = -filters.laplace(filters.gaussian(img, sigma=sigma))
                
                prefix = f'log_s{sigma:.1f}_'
                
                # LoG statistics
                features[f'{prefix}mean'] = np.mean(log_img)
                features[f'{prefix}std'] = np.std(log_img)
                features[f'{prefix}max'] = np.max(log_img)
                features[f'{prefix}min'] = np.min(log_img)
                features[f'{prefix}abs_mean'] = np.mean(np.abs(log_img))
                
                # Count of positive and negative responses (blobs)
                features[f'{prefix}positive_ratio'] = np.sum(log_img > 0) / log_img.size
                features[f'{prefix}negative_ratio'] = np.sum(log_img < 0) / log_img.size
                
                # Strong responses (potential blob centers)
                strong_threshold = np.percentile(np.abs(log_img), 90)
                features[f'{prefix}strong_responses'] = np.sum(np.abs(log_img) > strong_threshold) / log_img.size
                
            except Exception:
                prefix = f'log_s{sigma:.1f}_'
                for suffix in ['mean', 'std', 'max', 'min', 'abs_mean',
                              'positive_ratio', 'negative_ratio', 'strong_responses']:
                    features[f'{prefix}{suffix}'] = 0
        
        return features
    
    def extract_frequency_features(self, img: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features using FFT."""
        features = {}
        
        try:
            # 2D FFT
            fft = np.fft.fft2(img)
            fft_shift = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shift)
            power_spectrum = magnitude_spectrum**2
            
            # Radial frequency analysis
            h, w = img.shape
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            # Radial bins
            r_bins = np.linspace(0, np.max(r), 10)
            features['freq_low_power'] = np.sum(power_spectrum[r < r_bins[2]]) / np.sum(power_spectrum)
            features['freq_mid_power'] = np.sum(power_spectrum[(r >= r_bins[2]) & (r < r_bins[5])]) / np.sum(power_spectrum)
            features['freq_high_power'] = np.sum(power_spectrum[r >= r_bins[5]]) / np.sum(power_spectrum)
            
            # Spectral entropy
            power_norm = power_spectrum / (np.sum(power_spectrum) + 1e-10)
            features['freq_entropy'] = -np.sum(power_norm * np.log2(power_norm + 1e-10))
            
            # Peak frequency
            center_mask = r > r_bins[1]  # Exclude DC component
            if np.sum(center_mask) > 0:
                features['freq_peak_magnitude'] = np.max(magnitude_spectrum[center_mask])
            else:
                features['freq_peak_magnitude'] = 0
                
        except Exception:
            features['freq_low_power'] = 0
            features['freq_mid_power'] = 0
            features['freq_high_power'] = 0
            features['freq_entropy'] = 0
            features['freq_peak_magnitude'] = 0
        
        return features
    
    def extract_all_features(self, image_path: str) -> Dict[str, float]:
        """
        Extract all features from an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary of feature names and values
        """
        # Preprocess image
        img = self.preprocess_image(image_path)
        
        # Extract all feature groups
        features = {}
        features.update(self.extract_intensity_features(img))
        features.update(self.extract_multiscale_gaussian_features(img))
        features.update(self.extract_edge_gradient_features(img))
        features.update(self.extract_texture_features(img))
        features.update(self.extract_morphological_features(img))
        features.update(self.extract_hessian_features(img))
        features.update(self.extract_structure_tensor_features(img))
        features.update(self.extract_laplacian_features(img))
        features.update(self.extract_frequency_features(img))
        
        return features
    
    def extract_features_from_folder(self, 
                                     input_folder: str,
                                     output_file: str,
                                     file_extensions: Tuple[str] = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')) -> pd.DataFrame:
        """
        Extract features from all images in a folder and save to TSV.
        
        Args:
            input_folder: Path to folder containing images
            output_file: Path to output TSV file
            file_extensions: Tuple of valid image file extensions
            
        Returns:
            DataFrame with features (rows=images, columns=features)
        """
        # Find all image files
        image_files = []
        for ext in file_extensions:
            image_files.extend(Path(input_folder).glob(f'*{ext}'))
            image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))
        
        image_files = sorted([str(f) for f in image_files])
        
        if len(image_files) == 0:
            raise ValueError(f"No image files found in {input_folder}")
        
        print(f"Found {len(image_files)} images to process")
        
        # Extract features from all images
        features_list = []
        filenames = []
        
        for img_path in tqdm(image_files, desc="Extracting features"):
            try:
                features = self.extract_all_features(img_path)
                features_list.append(features)
                filenames.append(Path(img_path).name)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if len(features_list) == 0:
            raise ValueError("No features could be extracted from any image")
        
        # Create DataFrame
        df = pd.DataFrame(features_list, index=filenames)
        
        # Save to TSV
        os.makedirs(Path(output_file).parent, exist_ok=True)
        df.to_csv(output_file, sep='\t', index=True)
        print(f"\nFeatures saved to: {output_file}")
        print(f"  - Number of images: {len(df)}")
        print(f"  - Number of features: {len(df.columns)}")
        print(f"  - Output format: TSV (tab-separated)")
        
        return df


def main():
    """
    Command-line interface for feature extraction.
    
    Usage:
        python feature_extraction.py <input_folder> <output_file>
    
    Example:
        python feature_extraction.py ./data/images ./features/extracted_features.tsv
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract comprehensive features from microscopy images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract features from a folder
  python feature_extraction.py ./data/train/good ./features/good_features.tsv
  
  # Extract features from multiple folders
  for category in good semi_good bad; do
    python feature_extraction.py ./data/train/$category ./features/${category}_features.tsv
  done
        """
    )
    
    parser.add_argument('input_folder', type=str,
                       help='Path to folder containing images')
    parser.add_argument('output_file', type=str,
                       help='Path to output TSV file')
    parser.add_argument('--resize', type=int, nargs=2, default=[512, 512],
                       metavar=('HEIGHT', 'WIDTH'),
                       help='Image resize dimensions (default: 512 512)')
    parser.add_argument('--scales', type=float, nargs='+', 
                       default=[0.7, 1.0, 1.6, 3.5, 5.0, 10.0],
                       help='Gaussian scales for multi-scale features (default: 0.7 1.0 1.6 3.5 5.0 10.0)')
    
    args = parser.parse_args()
    
    # Create feature extractor
    extractor = FeatureExtractor(
        resize_shape=tuple(args.resize),
        gaussian_scales=args.scales
    )
    
    # Extract features
    print(f"Input folder: {args.input_folder}")
    print(f"Output file: {args.output_file}")
    print(f"Resize shape: {args.resize}")
    print(f"Gaussian scales: {args.scales}")
    print()
    
    try:
        df = extractor.extract_features_from_folder(
            args.input_folder,
            args.output_file
        )
        
        print("\n✓ Feature extraction completed successfully!")
        print(f"\nFeature statistics:")
        print(f"  - Total features: {len(df.columns)}")
        print(f"  - Feature categories:")
        
        # Count features by category
        categories = {
            'Intensity': len([c for c in df.columns if 'intensity' in c.lower() or c.startswith('p')]),
            'Gaussian': len([c for c in df.columns if 'gauss_' in c]),
            'Edge': len([c for c in df.columns if any(x in c for x in ['sobel', 'prewitt', 'scharr', 'gradient'])]),
            'Texture': len([c for c in df.columns if 'lbp' in c or 'contrast' in c or 'homogeneity' in c]),
            'Morphology': len([c for c in df.columns if 'thresh_' in c]),
            'Hessian': len([c for c in df.columns if 'hessian_' in c]),
            'Structure': len([c for c in df.columns if 'struct_' in c]),
            'LoG': len([c for c in df.columns if 'log_' in c]),
            'Frequency': len([c for c in df.columns if 'freq_' in c])
        }
        
        for cat, count in categories.items():
            print(f"    {cat}: {count}")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()