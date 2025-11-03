# Hyphae_segmentation
Image quality feature extraction, using them for shallow learning to train 2 different classifiers (good vs. bad images; semi-good vs. good image) to separate images into 3 groups. Use group specific parameters to segment and count hyphae.

![Hyphae_segmentation](./data/PS-216%20srfA%20+Fo_1%20vs%205_E4_6skeleton_no_border.png)

# Setup

Currently, we have hard-coded 2 classifiers and their associated parameters for image-processing.
If experimental setup requires different setting --> changes need to be made in the binary_train.py and binary_process.py

process_batch.py is also experimental design (folder structure) specific

Change config.ini for current use:

    [PATHS]
    # Base directories
    base_dir = /PATH/TO
    
    # Training
    training_data = %(base_dir)s/data/trainig_set/2025-10-29
    model_output = %(base_dir)s/results/classifiers/train_2025-10-29
    
    # Processing (for later use)
    data_dir = /PATH/TO/data/
    results_dir = %(base_dir)s/results/2025-10-23
    
    # Feature extraction
    feature_dir = %(base_dir)s/results/features
    
    # Models to use for processing
    bad_model = %(model_output)s/bad_detector_model.joblib
    semi_model = %(model_output)s/semi_good_detector_model.joblib
    
    [FEATURE_EXTRACTION]
    # Image preprocessing
    resize_height = 1440
    resize_width = 1440
    
    # Multi-scale parameters (comma-separated)
    gaussian_scales = 0.7, 1.0, 1.6, 3.5, 5.0, 10.0
    threshold_factors = 0.8, 1.0, 1.2
    
    [CLASSIFIER]
    # Random Forest parameters
    n_estimators = 200
    max_depth = 10
    min_samples_split = 2
    min_samples_leaf = 1
    max_features = sqrt
    random_state = 42
    test_size = 0.2

feature_extraction.py is good for general re-use to extract features for image classifiers training.

    Usage:
        python feature_extraction.py <input_folder> <output_file>
    
    Example:
        python feature_extraction.py ./data/images ./features/extracted_features.tsv