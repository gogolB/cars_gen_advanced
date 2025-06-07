# scripts/validate_dataset.py

import os
import glob
import csv
from typing import List, Dict, Tuple, Any

import cv2
import numpy as np
from skimage.measure import shannon_entropy
from tqdm import tqdm

# --- Configuration ---
# Define paths relative to the project root (cars_gen_advanced)
RAW_DATA_DIR = "data/raw_thyroid_cars"
HEALTHY_SUBDIR = "healthy"
CANCEROUS_SUBDIR = "cancerous"
REPORTS_DIR = "data/data_validation_reports"
OUTPUT_CSV_NAME = "dataset_validation_report.csv"

# Define thresholds for flagging. These might need tuning based on your specific dataset.
# For UINT16 images (0-65535 range)
LOW_VARIANCE_THRESHOLD = 50.0  # Pixel variance below this might be too flat
HIGH_MEAN_FOR_WHITE_STATIC = 60000.0 # If mean is above this and variance is low
BRIGHTNESS_CONTRAST_STD_DEV_FACTOR = 3.0 # Number of std devs from mean to be an outlier
LAPLACIAN_VARIANCE_LOW_THRESHOLD = 50.0 # Lower values indicate more blur
ENTROPY_LOW_THRESHOLD = 2.0 # Very low entropy might indicate blank/simple images
ENTROPY_HIGH_THRESHOLD = 14.0 # Max for UINT16 is ~16. Very high might be pure noise.
DOMINANT_INTENSITY_PERCENTAGE = 0.95 # If >95% of pixels are in a narrow band
DOMINANT_INTENSITY_RANGE_DIVISOR = 200 # Range is considered narrow if < (max_val / divisor)

# --- Helper Functions ---

def calculate_image_metrics(image_path: str) -> Dict[str, Any]:
    """
    Calculates various metrics for a single image.
    Args:
        image_path (str): Path to the image file.
    Returns:
        Dict[str, Any]: A dictionary containing image metrics.
                        Returns None if the image cannot be read.
    """
    try:
        # Read as UINT16, unchanged
        img_uint16 = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img_uint16 is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return None

        # Ensure it's a 2D grayscale image
        if img_uint16.ndim != 2:
            print(f"Warning: Image {image_path} is not 2D grayscale (dims: {img_uint16.ndim}). Skipping.")
            return None
        
        # Convert to float32 for calculations to avoid overflow with UINT16 intermediates
        img_float = img_uint16.astype(np.float32)

        metrics = {}
        metrics['path'] = image_path
        metrics['mean_intensity'] = np.mean(img_float)
        metrics['std_intensity'] = np.std(img_float) # Contrast
        metrics['pixel_variance'] = np.var(img_float)
        
        # Laplacian variance for sharpness
        # Apply Laplacian to the float32 version of the image and use a float output depth
        laplacian = cv2.Laplacian(img_float, cv2.CV_32F, ksize=3) 
        metrics['laplacian_variance'] = np.var(laplacian) # laplacian is already float32

        # Shannon entropy
        # skimage.measure.shannon_entropy expects image with intensity values in a typical range
        # For UINT16, it's fine, but it normalizes internally for its calculation.
        # If images are mostly flat, entropy might be low.
        metrics['shannon_entropy'] = shannon_entropy(img_uint16)

        # Dominant intensity check
        # This section remains unchanged as the error was not related to it.
        # The existing low pixel_variance check serves as a good proxy for "flat" or "dominant intensity".
        # More complex histogram analysis could be added if needed.
        metrics['is_mostly_flat'] = metrics['pixel_variance'] < LOW_VARIANCE_THRESHOLD * 10 # Stricter for this specific flag

        return metrics

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def analyze_dataset_metrics(all_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Analyzes the collected metrics from the entire dataset to find overall stats.
    Args:
        all_metrics (List[Dict[str, Any]]): List of metrics dictionaries for each image.
    Returns:
        Dict[str, float]: Dataset-wide statistics (mean, std dev for relevant metrics).
    """
    dataset_stats = {}
    if not all_metrics:
        return dataset_stats

    mean_intensities = [m['mean_intensity'] for m in all_metrics if m]
    std_intensities = [m['std_intensity'] for m in all_metrics if m]
    # laplacian_variances = [m['laplacian_variance'] for m in all_metrics if m] # Not used for dataset stats currently
    # entropies = [m['shannon_entropy'] for m in all_metrics if m] # Not used for dataset stats currently


    if mean_intensities:
        dataset_stats['avg_mean_intensity'] = np.mean(mean_intensities)
        dataset_stats['std_mean_intensity'] = np.std(mean_intensities)
    if std_intensities:
        dataset_stats['avg_std_intensity'] = np.mean(std_intensities)
        dataset_stats['std_std_intensity'] = np.std(std_intensities)
    # For laplacian and entropy, we use fixed thresholds in flag_outliers
    
    return dataset_stats


def flag_outliers(image_metrics: Dict[str, Any], dataset_stats: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Flags an image as a potential outlier based on its metrics and dataset statistics.
    Args:
        image_metrics (Dict[str, Any]): Metrics for the current image.
        dataset_stats (Dict[str, float]): Dataset-wide statistics.
    Returns:
        Tuple[bool, List[str]]: (is_outlier_flag, list_of_reasons)
    """
    is_outlier = False
    reasons = []

    # 1. Low Pixel Variance (too flat)
    if image_metrics['pixel_variance'] < LOW_VARIANCE_THRESHOLD:
        reasons.append(f"LowPixelVariance ({image_metrics['pixel_variance']:.2f} < {LOW_VARIANCE_THRESHOLD})")
        is_outlier = True

    # 2. "White Static" like (very high mean, very low variance)
    if image_metrics['mean_intensity'] > HIGH_MEAN_FOR_WHITE_STATIC and \
       image_metrics['pixel_variance'] < LOW_VARIANCE_THRESHOLD * 2.0: # Slightly higher var threshold for this specific case
        reasons.append(f"WhiteStaticLike (Mean: {image_metrics['mean_intensity']:.2f}, Var: {image_metrics['pixel_variance']:.2f})")
        is_outlier = True
        
    # 3. Brightness Outlier
    if 'avg_mean_intensity' in dataset_stats and 'std_mean_intensity' in dataset_stats:
        lower_bound = dataset_stats['avg_mean_intensity'] - BRIGHTNESS_CONTRAST_STD_DEV_FACTOR * dataset_stats['std_mean_intensity']
        upper_bound = dataset_stats['avg_mean_intensity'] + BRIGHTNESS_CONTRAST_STD_DEV_FACTOR * dataset_stats['std_mean_intensity']
        if not (lower_bound <= image_metrics['mean_intensity'] <= upper_bound):
            reasons.append(f"BrightnessOutlier ({image_metrics['mean_intensity']:.2f} not in [{lower_bound:.2f}, {upper_bound:.2f}])")
            is_outlier = True

    # 4. Contrast Outlier
    if 'avg_std_intensity' in dataset_stats and 'std_std_intensity' in dataset_stats:
        lower_bound_contrast = max(0.1, dataset_stats['avg_std_intensity'] - BRIGHTNESS_CONTRAST_STD_DEV_FACTOR * dataset_stats['std_std_intensity']) # Ensure contrast isn't practically zero
        upper_bound_contrast = dataset_stats['avg_std_intensity'] + BRIGHTNESS_CONTRAST_STD_DEV_FACTOR * dataset_stats['std_std_intensity']
        if not (lower_bound_contrast <= image_metrics['std_intensity'] <= upper_bound_contrast):
            # Also flag if std_intensity is extremely low (e.g. < 1.0 for UINT16 scaled data)
            if image_metrics['std_intensity'] < 1.0 :
                 reasons.append(f"VeryLowContrast ({image_metrics['std_intensity']:.2f})")
            else:
                reasons.append(f"ContrastOutlier ({image_metrics['std_intensity']:.2f} not in [{lower_bound_contrast:.2f}, {upper_bound_contrast:.2f}])")
            is_outlier = True
            
    # 5. Blur (Low Laplacian Variance)
    if image_metrics['laplacian_variance'] < LAPLACIAN_VARIANCE_LOW_THRESHOLD:
        reasons.append(f"Blurry (LaplacianVar {image_metrics['laplacian_variance']:.2f} < {LAPLACIAN_VARIANCE_LOW_THRESHOLD})")
        is_outlier = True

    # 6. Entropy Outlier
    if image_metrics['shannon_entropy'] < ENTROPY_LOW_THRESHOLD:
        reasons.append(f"LowEntropy ({image_metrics['shannon_entropy']:.2f} < {ENTROPY_LOW_THRESHOLD})")
        is_outlier = True
    elif image_metrics['shannon_entropy'] > ENTROPY_HIGH_THRESHOLD:
        reasons.append(f"HighEntropy ({image_metrics['shannon_entropy']:.2f} > {ENTROPY_HIGH_THRESHOLD})")
        is_outlier = True
        
    return is_outlier, reasons


# --- Main Script ---
def main():
    print("Starting dataset validation...")
    
    # Ensure reports directory exists
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    image_files_with_labels = []

    # Gather all image paths and their labels
    for class_label, subdir in [(HEALTHY_SUBDIR, HEALTHY_SUBDIR), (CANCEROUS_SUBDIR, CANCEROUS_SUBDIR)]:
        class_path = os.path.join(RAW_DATA_DIR, subdir)
        # Supporting common image types, though CARS are likely .tif
        current_files = []
        for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"): 
            current_files.extend(glob.glob(os.path.join(class_path, ext)))
        
        for f_path in current_files:
            image_files_with_labels.append({'path': f_path, 'label': class_label})
            
    if not image_files_with_labels:
        print(f"No image files found in {RAW_DATA_DIR}. Please check the paths and file extensions.")
        return

    print(f"Found {len(image_files_with_labels)} images to analyze.")

    # --- First Pass: Calculate metrics for all images ---
    all_image_metrics_data = []
    # Using relative paths in the report for portability
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    print("Pass 1: Calculating metrics for all images...")
    for item in tqdm(image_files_with_labels, desc="Calculating Metrics"):
        image_file_abs_path = item['path']
        class_label = item['label']
        
        # Get relative path for report
        try:
            image_file_rel_path = os.path.relpath(image_file_abs_path, project_root)
        except ValueError: # Happens if paths are on different drives (e.g. Windows)
            image_file_rel_path = image_file_abs_path # Fallback to absolute path

        metrics = calculate_image_metrics(image_file_abs_path)
        if metrics:
            # Override path with relative path for the report
            metrics['path'] = image_file_rel_path 
            metrics['class_label'] = class_label # Add class label to metrics
            all_image_metrics_data.append(metrics)
            
    if not all_image_metrics_data:
        print("No image metrics could be calculated. Exiting.")
        return

    # --- Analyze dataset-wide statistics ---
    print("Analyzing overall dataset statistics...")
    dataset_overall_stats = analyze_dataset_metrics(all_image_metrics_data)
    if not dataset_overall_stats:
        print("Could not compute dataset-wide statistics. Thresholds for brightness/contrast outliers might not work as expected.")
        dataset_overall_stats = { 
            'avg_mean_intensity': 32768.0, 'std_mean_intensity': 10000.0, 
            'avg_std_intensity': 5000.0, 'std_std_intensity': 2000.0
        }


    # --- Second Pass: Flag outliers and write report ---
    report_path = os.path.join(REPORTS_DIR, OUTPUT_CSV_NAME)
    print(f"Pass 2: Flagging outliers and writing report to {report_path}...")
    
    num_outliers = 0
    with open(report_path, 'w', newline='') as csvfile:
        if not all_image_metrics_data:
             print("No metrics data to write to report.")
             return

        first_valid_metric = next(iter(all_image_metrics_data), None)
        if not first_valid_metric:
            print("Critical error: No valid metrics found after processing images.")
            return
            
        fieldnames = list(first_valid_metric.keys()) 
        if 'is_potential_outlier' not in fieldnames:
            fieldnames.append('is_potential_outlier')
        if 'reasons_for_flagging' not in fieldnames:
            fieldnames.append('reasons_for_flagging')
            
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for metrics_dict in tqdm(all_image_metrics_data, desc="Flagging Outliers"):
            is_outlier, reasons = flag_outliers(metrics_dict, dataset_overall_stats)
            # Ensure these keys exist before assignment if metrics_dict is fresh
            metrics_dict_to_write = metrics_dict.copy()
            metrics_dict_to_write['is_potential_outlier'] = is_outlier
            metrics_dict_to_write['reasons_for_flagging'] = "; ".join(reasons) if reasons else "N/A"
            writer.writerow(metrics_dict_to_write)
            if is_outlier:
                num_outliers += 1
                
    print(f"Dataset validation complete. Report saved to {report_path}")
    print(f"Total images processed: {len(all_image_metrics_data)}")
    print(f"Potential outliers flagged: {num_outliers}")
    if num_outliers > 0:
        print("Please review the CSV report and visually inspect the flagged images.")

if __name__ == "__main__":
    main()

