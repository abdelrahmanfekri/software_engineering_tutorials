# Module 1: Plant Phenotyping Fundamentals

## Overview
Plant phenotyping is the quantitative assessment of plant traits and characteristics. This module covers the fundamentals of image-based plant analysis, from image acquisition to trait quantification, preparing you for building ML solutions in agricultural and biological research.

## 1. Introduction to Plant Phenotyping

### What is Plant Phenotyping?
Plant phenotyping involves measuring and analyzing observable characteristics of plants, including:
- **Morphological traits**: Size, shape, color, structure
- **Physiological traits**: Growth rate, biomass accumulation
- **Biochemical traits**: Chlorophyll content, nutrient levels
- **Disease and stress indicators**: Pathogen presence, drought stress

### Applications
- Crop breeding and selection
- Disease detection and monitoring
- Growth analysis and yield prediction
- Stress response assessment
- Quality control in agriculture

### Image-Based Phenotyping Advantages
- Non-destructive measurement
- High-throughput analysis
- Objective and reproducible
- Scalable to large datasets
- Enables temporal analysis

## 2. Image Acquisition for Plants

### Imaging Modalities

#### RGB Imaging
- Standard color cameras
- Most common and cost-effective
- Suitable for: morphology, color analysis, disease symptoms
- Limitations: Limited spectral information

```python
import cv2
import numpy as np
from PIL import Image

def load_rgb_image(image_path):
    """Load RGB image for plant analysis"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def extract_color_channels(image):
    """Extract R, G, B channels"""
    r, g, b = cv2.split(image)
    return r, g, b
```

#### Multispectral Imaging
- Captures specific wavelength bands
- Common bands: Red, Green, Blue, Red Edge, Near-Infrared (NIR)
- Applications: Vegetation indices (NDVI, EVI), stress detection
- Equipment: Multispectral cameras, modified cameras

#### Hyperspectral Imaging
- Captures hundreds of narrow spectral bands
- Provides detailed spectral signatures
- Applications: Chemical composition analysis, disease early detection
- More expensive and complex

### Image Acquisition Best Practices
- Consistent lighting conditions
- Controlled background (green screen, white background)
- Proper camera calibration
- Standardized distance and angle
- Multiple views for 3D reconstruction
- Temporal consistency for time-series

## 3. Plant Trait Quantification

### Key Traits to Measure

#### 1. Leaf Area
- Critical for photosynthesis assessment
- Methods: Pixel counting, contour analysis, segmentation

```python
def calculate_leaf_area(binary_mask, pixel_size_mm2=1.0):
    """
    Calculate leaf area from binary mask
    
    Args:
        binary_mask: Binary image where 1 = leaf, 0 = background
        pixel_size_mm2: Area of one pixel in mm²
    
    Returns:
        Total leaf area in mm²
    """
    leaf_pixels = np.sum(binary_mask == 1)
    area_mm2 = leaf_pixels * pixel_size_mm2
    return area_mm2

def segment_leaves(image):
    """Segment leaves from background using color-based segmentation"""
    # Convert to HSV for better color separation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Define green color range for leaves
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Create mask
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Morphological operations to clean mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask
```

#### 2. Plant Height
- Vertical growth measurement
- Methods: Depth estimation, stereo vision, LiDAR

```python
def estimate_plant_height(image, reference_height_pixels, reference_height_cm):
    """
    Estimate plant height using reference object
    
    Args:
        image: Plant image
        reference_height_pixels: Height of reference object in pixels
        reference_height_cm: Actual height of reference object in cm
    
    Returns:
        Estimated plant height in cm
    """
    # Detect plant top and bottom (simplified)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    plant_height_pixels = h
    
    # Convert to cm
    pixel_to_cm = reference_height_cm / reference_height_pixels
    plant_height_cm = plant_height_pixels * pixel_to_cm
    
    return plant_height_cm
```

#### 3. Biomass Estimation
- Total plant mass
- Methods: Volume estimation, allometric relationships, ML regression

#### 4. Disease Detection
- Visual symptoms identification
- Methods: Color analysis, texture analysis, deep learning

```python
def detect_disease_symptoms(image):
    """Detect disease symptoms using color analysis"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Yellow/brown spots (common disease symptoms)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([20, 255, 255])
    
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    disease_mask = cv2.bitwise_or(yellow_mask, brown_mask)
    disease_percentage = (np.sum(disease_mask > 0) / disease_mask.size) * 100
    
    return disease_mask, disease_percentage
```

#### 5. Plant Counting
- Number of plants or plant parts
- Methods: Object detection, blob detection, instance segmentation

```python
def count_plants(image):
    """Count plants using blob detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area (remove noise)
    min_area = 100
    plant_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    return len(plant_contours), plant_contours
```

## 4. Image Preprocessing for Plant Images

### Background Removal
```python
def remove_background(image, method='color'):
    """Remove background from plant image"""
    if method == 'color':
        # Color-based segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
    elif method == 'grabcut':
        # GrabCut algorithm
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (50, 50, image.shape[1]-100, image.shape[0]-100)
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    result = image * mask[:, :, np.newaxis]
    return result, mask
```

### Image Enhancement
```python
def enhance_plant_image(image):
    """Enhance plant image for better analysis"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced
```

### Noise Reduction
```python
def denoise_plant_image(image):
    """Remove noise from plant image"""
    # Bilateral filter preserves edges
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    return denoised
```

### Normalization
```python
def normalize_plant_image(image, target_size=(224, 224)):
    """Normalize plant image for ML models"""
    # Resize
    resized = cv2.resize(image, target_size)
    
    # Normalize pixel values to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Standard normalization (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (normalized - mean) / std
    
    return normalized
```

## 5. Classical Computer Vision Techniques

### Feature Extraction

#### Color Features
```python
def extract_color_features(image, mask=None):
    """Extract color-based features"""
    if mask is not None:
        image = image[mask > 0]
    
    features = {
        'mean_r': np.mean(image[:, 0]),
        'mean_g': np.mean(image[:, 1]),
        'mean_b': np.mean(image[:, 2]),
        'std_r': np.std(image[:, 0]),
        'std_g': np.std(image[:, 1]),
        'std_b': np.std(image[:, 2]),
    }
    
    # Vegetation indices
    r, g, b = image[:, 0], image[:, 1], image[:, 2]
    features['exg'] = 2 * g - r - b  # Excess Green
    features['exgr'] = features['exg'] - (1.4 * r - g)  # Excess Green-Red
    
    return features
```

#### Texture Features
```python
from skimage.feature import graycomatrix, graycoprops

def extract_texture_features(image):
    """Extract texture features using GLCM"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Calculate GLCM
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances, angles, levels=256, symmetric=True, normed=True)
    
    # Extract properties
    features = {
        'contrast': np.mean(graycoprops(glcm, 'contrast')),
        'dissimilarity': np.mean(graycoprops(glcm, 'dissimilarity')),
        'homogeneity': np.mean(graycoprops(glcm, 'homogeneity')),
        'energy': np.mean(graycoprops(glcm, 'energy')),
        'correlation': np.mean(graycoprops(glcm, 'correlation')),
    }
    
    return features
```

#### Shape Features
```python
def extract_shape_features(contour):
    """Extract shape features from contour"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        return {}
    
    features = {
        'area': area,
        'perimeter': perimeter,
        'circularity': 4 * np.pi * area / (perimeter ** 2),
        'solidity': area / cv2.contourArea(cv2.convexHull(contour)),
    }
    
    # Bounding box features
    x, y, w, h = cv2.boundingRect(contour)
    features['aspect_ratio'] = w / h if h > 0 else 0
    features['extent'] = area / (w * h) if (w * h) > 0 else 0
    
    # Moments
    M = cv2.moments(contour)
    if M['m00'] != 0:
        features['centroid_x'] = M['m10'] / M['m00']
        features['centroid_y'] = M['m01'] / M['m00']
    
    return features
```

## 6. Dataset Preparation and Annotation

### Data Collection Strategy
- Diverse conditions (lighting, angles, growth stages)
- Multiple plant varieties
- Temporal sequences for growth analysis
- Quality control and validation

### Annotation Tools
- LabelMe
- CVAT
- VGG Image Annotator (VIA)
- Custom annotation pipelines

### Annotation Types
- Bounding boxes (object detection)
- Polygons (segmentation)
- Keypoints (landmark detection)
- Classification labels

### Data Organization
```
plant_phenotyping_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── annotations/
│   ├── train/
│   ├── val/
│   └── test/
├── metadata/
│   └── dataset_info.json
└── README.md
```

## 7. Evaluation Metrics for Phenotyping

### Classification Metrics
- Accuracy, Precision, Recall, F1-score
- Confusion matrix
- ROC-AUC

### Detection Metrics
- mAP (mean Average Precision)
- IoU (Intersection over Union)

### Segmentation Metrics
- Pixel accuracy
- IoU per class
- Dice coefficient
- Boundary accuracy

### Regression Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² score

```python
def calculate_phenotyping_metrics(predictions, ground_truth):
    """Calculate comprehensive metrics for phenotyping"""
    metrics = {}
    
    # Classification metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    metrics['accuracy'] = accuracy_score(ground_truth, predictions)
    metrics['precision'] = precision_score(ground_truth, predictions, average='weighted')
    metrics['recall'] = recall_score(ground_truth, predictions, average='weighted')
    metrics['f1'] = f1_score(ground_truth, predictions, average='weighted')
    
    return metrics
```

## 8. Practical Exercises

### Exercise 1: Leaf Area Calculation
1. Load a plant image
2. Segment leaves from background
3. Calculate total leaf area
4. Visualize results

### Exercise 2: Disease Detection
1. Create a dataset with healthy and diseased plants
2. Extract color and texture features
3. Train a simple classifier
4. Evaluate performance

### Exercise 3: Plant Counting
1. Process an image with multiple plants
2. Count individual plants
3. Calculate average plant size
4. Generate a report

## 9. Best Practices

1. **Consistent Imaging**: Standardize acquisition conditions
2. **Quality Control**: Validate image quality before processing
3. **Feature Engineering**: Combine multiple feature types
4. **Validation**: Use cross-validation for small datasets
5. **Documentation**: Document all preprocessing steps
6. **Reproducibility**: Save preprocessing parameters

## 10. Common Challenges and Solutions

### Challenge: Variable Lighting
**Solution**: Use color normalization, histogram equalization, or learn robust features

### Challenge: Overlapping Plants
**Solution**: Use instance segmentation or 3D reconstruction

### Challenge: Background Complexity
**Solution**: Controlled imaging conditions or advanced segmentation

### Challenge: Scale Variation
**Solution**: Multi-scale processing or scale-invariant features

## Next Steps

Proceed to [Module 2: Deep Learning for Plant Image Analysis](02-deep-learning-plant-analysis.md) to learn how to apply deep learning models for advanced plant phenotyping tasks.

