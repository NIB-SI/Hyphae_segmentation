#!/usr/bin/env python3
"""
Train binary classifiers for hyphae image quality detection.
Uses feature_extraction module and config.ini for all settings.
"""

import os
import configparser
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from feature_extraction import FeatureExtractor


def load_config(config_path='config.ini'):
    """Load configuration from ini file."""
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def extract_features_from_training_data(training_dir, extractor):
    """Extract features from all training images."""
    data = []
    
    # Collect all images with labels
    for category in ['good', 'semi_good', 'bad']:
        category_path = os.path.join(training_dir, category)
        if not os.path.exists(category_path):
            print(f"Warning: {category} folder not found at {category_path}")
            continue
        
        image_files = [f for f in os.listdir(category_path) 
                      if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        print(f"Found {len(image_files)} images in '{category}'")
        
        for img_file in image_files:
            data.append({
                'filepath': os.path.join(category_path, img_file),
                'filename': img_file,
                'label': category
            })
    
    # Extract features
    print(f"\nExtracting features from {len(data)} images...")
    features_list = []
    filenames = []
    labels = []
    
    for item in tqdm(data, desc="Processing images"):
        try:
            features = extractor.extract_all_features(item['filepath'])
            features_list.append(features)
            filenames.append(item['filename'])
            labels.append(item['label'])
        except Exception as e:
            print(f"Error processing {item['filepath']}: {e}")
    
    # Create DataFrame
    X = pd.DataFrame(features_list, index=filenames)
    y = pd.Series(labels, index=filenames)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Label distribution:\n{y.value_counts()}")
    
    return X, y


def train_binary_classifier(X, y, target_class, clf_params):
    """Train a binary classifier for one target class."""
    # Create binary labels
    y_binary = (y == target_class).astype(int)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=clf_params['test_size'], 
        random_state=clf_params['random_state']
    )
    
    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=clf_params['n_estimators'],
        max_depth=clf_params['max_depth'],
        min_samples_split=clf_params['min_samples_split'],
        min_samples_leaf=clf_params['min_samples_leaf'],
        max_features=clf_params['max_features'],
        random_state=clf_params['random_state'],
        class_weight='balanced'
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    if len(X_test) > 0:
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        report = {
            'accuracy': accuracy,
            'classification_report': classification_report(
                y_test, y_pred, 
                target_names=[f'Not {target_class}', target_class],
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    else:
        report = None
    
    # Feature importance
    feature_importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    return clf, report, feature_importance


def main():
    # Load configuration
    config = load_config()
    
    training_dir = config['PATHS']['training_data']
    output_dir = config['PATHS']['model_output']
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("TRAINING BINARY CLASSIFIERS WITH NEW FEATURE EXTRACTION")
    print("="*70)
    print(f"Training data: {training_dir}")
    print(f"Output directory: {output_dir}")
    
    # Initialize feature extractor with config parameters
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
    
    # Extract features from training data
    X, y = extract_features_from_training_data(training_dir, extractor)
    
    # Save features
    feature_file = os.path.join(output_dir, "training_features.tsv")
    X_with_labels = X.copy()
    X_with_labels['label'] = y
    X_with_labels.to_csv(feature_file, sep='\t')
    print(f"\nTraining features saved to: {feature_file}")
    
    # Classifier parameters from config
    clf_params = {
        'n_estimators': int(config['CLASSIFIER']['n_estimators']),
        'max_depth': int(config['CLASSIFIER']['max_depth']),
        'min_samples_split': int(config['CLASSIFIER']['min_samples_split']),
        'min_samples_leaf': int(config['CLASSIFIER']['min_samples_leaf']),
        'max_features': config['CLASSIFIER']['max_features'],
        'random_state': int(config['CLASSIFIER']['random_state']),
        'test_size': float(config['CLASSIFIER']['test_size'])
    }
    
    # Train BAD detector
    print("\n" + "="*70)
    print("TRAINING BAD IMAGE DETECTOR")
    print("="*70)
    bad_clf, bad_report, bad_importance = train_binary_classifier(X, y, 'bad', clf_params)
    
    if bad_report:
        print(f"\nAccuracy: {bad_report['accuracy']:.3f}")
        print("\nClassification Report:")
        print(classification_report(
            [0, 1] * len(bad_report['confusion_matrix']),
            [0, 1] * len(bad_report['confusion_matrix']),
            target_names=['Not bad', 'Bad'],
            labels=[0, 1]
        ))
    
    print("\nTop 10 features for BAD detection:")
    print(bad_importance.head(10))
    
    # Train SEMI-GOOD detector
    print("\n" + "="*70)
    print("TRAINING SEMI-GOOD IMAGE DETECTOR")
    print("="*70)
    semi_clf, semi_report, semi_importance = train_binary_classifier(X, y, 'semi_good', clf_params)
    
    if semi_report:
        print(f"\nAccuracy: {semi_report['accuracy']:.3f}")
        print("\nClassification Report:")
        print(classification_report(
            [0, 1] * len(semi_report['confusion_matrix']),
            [0, 1] * len(semi_report['confusion_matrix']),
            target_names=['Not semi-good', 'Semi-good'],
            labels=[0, 1]
        ))
    
    print("\nTop 10 features for SEMI-GOOD detection:")
    print(semi_importance.head(10))
    
    # Save models
    bad_model_path = os.path.join(output_dir, "bad_detector_model.joblib")
    semi_model_path = os.path.join(output_dir, "semi_good_detector_model.joblib")
    
    joblib.dump(bad_clf, bad_model_path)
    joblib.dump(semi_clf, semi_model_path)
    
    print("\n" + "="*70)
    print("MODELS SAVED")
    print("="*70)
    print(f"BAD detector: {bad_model_path}")
    print(f"SEMI-GOOD detector: {semi_model_path}")
    
    # Test combined approach on full training data
    print("\n" + "="*70)
    print("COMBINED CLASSIFIER PERFORMANCE ON TRAINING DATA")
    print("="*70)
    
    bad_predictions = bad_clf.predict(X)
    semi_predictions = semi_clf.predict(X)
    
    # Combine predictions: bad > semi_good > good
    final_predictions = []
    for i in range(len(X)):
        if bad_predictions[i] == 1:
            final_predictions.append('bad')
        elif semi_predictions[i] == 1:
            final_predictions.append('semi_good')
        else:
            final_predictions.append('good')
    
    final_predictions = pd.Series(final_predictions, index=X.index)
    
    # Confusion matrix
    comparison_df = pd.DataFrame({
        'True': y,
        'Predicted': final_predictions
    })
    
    confusion = comparison_df.groupby(['True', 'Predicted']).size().unstack(fill_value=0)
    print("\nConfusion Matrix:")
    print(confusion)
    
    # Overall accuracy
    accuracy = (y == final_predictions).sum() / len(y)
    print(f"\nOverall Training Accuracy: {accuracy:.3f}")
    
    # Save training report
    report_path = os.path.join(output_dir, "training_report.txt")
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BINARY CLASSIFIER TRAINING REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Training images: {len(X)}\n")
        f.write(f"Features extracted: {len(X.columns)}\n")
        f.write(f"Label distribution:\n{y.value_counts()}\n\n")
        
        f.write("BAD DETECTOR\n")
        f.write("-"*70 + "\n")
        if bad_report:
            f.write(f"Accuracy: {bad_report['accuracy']:.3f}\n")
        f.write(f"\nTop 10 features:\n{bad_importance.head(10)}\n\n")
        
        f.write("SEMI-GOOD DETECTOR\n")
        f.write("-"*70 + "\n")
        if semi_report:
            f.write(f"Accuracy: {semi_report['accuracy']:.3f}\n")
        f.write(f"\nTop 10 features:\n{semi_importance.head(10)}\n\n")
        
        f.write("COMBINED CLASSIFIER\n")
        f.write("-"*70 + "\n")
        f.write(f"Overall Training Accuracy: {accuracy:.3f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(confusion) + "\n")
    
    print(f"\nTraining report saved to: {report_path}")
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
