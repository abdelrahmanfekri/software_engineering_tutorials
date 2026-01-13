# Module 3: Advanced Plant Phenotyping Techniques

## Overview
This module covers advanced techniques for plant phenotyping including time-series analysis, 3D reconstruction, hyperspectral imaging, domain adaptation, and production deployment strategies.

## 1. Time-Series Analysis for Growth Monitoring

### Temporal Data Collection
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TemporalPhenotyping:
    """Analyze plant growth over time"""
    
    def __init__(self, plant_id):
        self.plant_id = plant_id
        self.measurements = []
    
    def add_measurement(self, date, height, leaf_area, disease_status):
        """Add measurement at specific date"""
        self.measurements.append({
            'date': date,
            'height': height,
            'leaf_area': leaf_area,
            'disease_status': disease_status
        })
    
    def calculate_growth_rate(self):
        """Calculate growth rate over time"""
        df = pd.DataFrame(self.measurements)
        df = df.sort_values('date')
        
        df['height_growth_rate'] = df['height'].diff() / df['date'].diff().dt.days
        df['leaf_area_growth_rate'] = df['leaf_area'].diff() / df['date'].diff().dt.days
        
        return df
```

### Growth Curve Fitting
```python
from scipy.optimize import curve_fit

def logistic_growth(t, K, r, t0):
    """Logistic growth model"""
    return K / (1 + np.exp(-r * (t - t0)))

def fit_growth_curve(dates, heights):
    """Fit logistic growth curve to height data"""
    # Convert dates to numeric
    t = np.array([(d - dates[0]).days for d in dates])
    
    # Initial parameters
    K_init = max(heights) * 1.2  # Carrying capacity
    r_init = 0.1  # Growth rate
    t0_init = len(dates) / 2  # Inflection point
    
    # Fit curve
    popt, _ = curve_fit(
        logistic_growth,
        t,
        heights,
        p0=[K_init, r_init, t0_init],
        maxfev=10000
    )
    
    return popt, logistic_growth(t, *popt)
```

## 2. 3D Reconstruction of Plants

### Multi-View Stereo
```python
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

class Plant3DReconstruction:
    """Reconstruct 3D structure from multiple views"""
    
    def __init__(self):
        self.camera_params = None
        self.images = []
        self.points_3d = []
    
    def calibrate_cameras(self, calibration_images):
        """Calibrate cameras using checkerboard"""
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Find corners
        objpoints = []
        imgpoints = []
        
        objp = np.zeros((6*9, 3), np.float32)
        objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        
        for img in calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
        
        # Calibrate
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        self.camera_params = {'mtx': mtx, 'dist': dist}
        return self.camera_params
    
    def reconstruct_3d(self, image1, image2, camera1_pose, camera2_pose):
        """Reconstruct 3D points from stereo pair"""
        # Feature detection and matching
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(image1, None)
        kp2, des2 = sift.detectAndCompute(image2, None)
        
        # Match features
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Filter matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        # Triangulate
        points_3d = cv2.triangulatePoints(
            camera1_pose, camera2_pose, pts1.T, pts2.T
        )
        points_3d = points_3d / points_3d[3]  # Homogeneous to 3D
        
        return points_3d[:3].T
```

## 3. Hyperspectral Image Analysis

### Hyperspectral Data Processing
```python
import spectral
import numpy as np

class HyperspectralAnalysis:
    """Analyze hyperspectral plant images"""
    
    def __init__(self, hyperspectral_cube):
        """
        Args:
            hyperspectral_cube: numpy array of shape (height, width, bands)
        """
        self.cube = hyperspectral_cube
        self.height, self.width, self.bands = hyperspectral_cube.shape
    
    def calculate_vegetation_indices(self):
        """Calculate various vegetation indices"""
        indices = {}
        
        # NDVI (Normalized Difference Vegetation Index)
        # Assuming red band at index 30, NIR at index 60
        red = self.cube[:, :, 30]
        nir = self.cube[:, :, 60]
        indices['ndvi'] = (nir - red) / (nir + red + 1e-10)
        
        # EVI (Enhanced Vegetation Index)
        blue = self.cube[:, :, 10]
        indices['evi'] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
        
        # PRI (Photochemical Reflectance Index)
        # Assuming 531nm and 570nm bands
        band_531 = self.cube[:, :, 20]  # Approximate
        band_570 = self.cube[:, :, 25]  # Approximate
        indices['pri'] = (band_531 - band_570) / (band_531 + band_570 + 1e-10)
        
        return indices
    
    def spectral_unmixing(self, endmembers):
        """Perform spectral unmixing"""
        # Reshape cube to 2D
        pixels = self.cube.reshape(-1, self.bands)
        
        # Unmix using least squares
        abundances = np.linalg.lstsq(endmembers.T, pixels.T, rcond=None)[0]
        abundances = abundances.T
        abundances = np.clip(abundances, 0, 1)
        
        # Normalize
        abundances = abundances / (abundances.sum(axis=1, keepdims=True) + 1e-10)
        
        # Reshape back
        abundances = abundances.reshape(self.height, self.width, -1)
        
        return abundances
```

## 4. Domain Adaptation

### Lab to Field Adaptation
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class DomainAdapter(nn.Module):
    """Domain adaptation network"""
    
    def __init__(self, feature_extractor, classifier, domain_classifier):
        super(DomainAdapter, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.domain_classifier = domain_classifier
    
    def forward(self, x, alpha=1.0):
        # Extract features
        features = self.feature_extractor(x)
        
        # Reverse gradient for domain classifier
        reverse_features = ReverseLayerF.apply(features, alpha)
        
        # Class predictions
        class_pred = self.classifier(features)
        
        # Domain predictions
        domain_pred = self.domain_classifier(reverse_features)
        
        return class_pred, domain_pred

class ReverseLayerF(torch.autograd.Function):
    """Gradient reversal layer"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def train_domain_adaptation(model, source_loader, target_loader, num_epochs=50):
    """Train domain adaptation model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        # Gradually increase domain adaptation weight
        p = epoch / num_epochs
        alpha = 2.0 / (1 + np.exp(-10 * p)) - 1
        
        # Train on source (labeled)
        for (src_x, src_y) in source_loader:
            class_pred, domain_pred = model(src_x, alpha)
            
            # Classification loss (source only)
            loss_class = criterion_class(class_pred, src_y)
            
            # Domain loss (source = 0)
            loss_domain_src = criterion_domain(domain_pred, torch.zeros(len(src_x), dtype=torch.long))
            
            # Total loss
            loss = loss_class + loss_domain_src
            loss.backward()
            optimizer.step()
        
        # Train on target (unlabeled)
        for tgt_x in target_loader:
            _, domain_pred = model(tgt_x, alpha)
            
            # Domain loss (target = 1)
            loss_domain_tgt = criterion_domain(domain_pred, torch.ones(len(tgt_x), dtype=torch.long))
            
            loss_domain_tgt.backward()
            optimizer.step()
```

## 5. Real-Time Inference Optimization

### Model Quantization
```python
import torch
import torch.quantization

def quantize_model(model, calibration_data):
    """Quantize model for faster inference"""
    model.eval()
    
    # Prepare for quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate
    for data in calibration_data:
        model(data)
    
    # Convert to quantized
    quantized_model = torch.quantization.convert(model, inplace=False)
    
    return quantized_model
```

### TensorRT Optimization
```python
# Convert PyTorch to ONNX
import torch.onnx

def export_to_onnx(model, dummy_input, onnx_path):
    """Export model to ONNX"""
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

# Then use TensorRT to optimize ONNX model
# trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```

## 6. Complete End-to-End Pipeline

```python
class PlantPhenotypingPipeline:
    """Complete phenotyping pipeline"""
    
    def __init__(self, model_path, config):
        self.model = self.load_model(model_path)
        self.config = config
        self.preprocessor = ImagePreprocessor(config)
        self.postprocessor = ResultPostprocessor(config)
    
    def process_image(self, image_path):
        """Process single image"""
        # 1. Load and preprocess
        image = self.preprocessor.load_image(image_path)
        processed = self.preprocessor.preprocess(image)
        
        # 2. Run inference
        with torch.no_grad():
            predictions = self.model(processed)
        
        # 3. Post-process
        results = self.postprocessor.process(predictions)
        
        return results
    
    def process_batch(self, image_paths):
        """Process batch of images"""
        results = []
        for image_path in image_paths:
            result = self.process_image(image_path)
            results.append(result)
        return results
    
    def process_video(self, video_path, frame_interval=30):
        """Process video for temporal analysis"""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        results = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                result = self.process_image(frame)
                result['frame_number'] = frame_count
                results.append(result)
            
            frame_count += 1
        
        cap.release()
        return results
```

## 7. Production Deployment Considerations

### Model Serving
```python
from flask import Flask, request, jsonify
import torch
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
model = torch.load('plant_classifier.pth')
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    # Decode image
    image_data = request.json['image']
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    
    # Preprocess
    processed = preprocess_image(image)
    
    # Predict
    with torch.no_grad():
        output = model(processed)
        prediction = torch.argmax(output, dim=1)
        probabilities = torch.softmax(output, dim=1)
    
    return jsonify({
        'prediction': prediction.item(),
        'probabilities': probabilities.tolist()
    })
```

### Caching Strategy
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_predict(image_hash, model_version):
    """Cache predictions for identical images"""
    # Prediction logic
    pass

def predict_with_cache(image):
    """Predict with caching"""
    image_hash = hashlib.md5(image.tobytes()).hexdigest()
    model_version = get_model_version()
    return cached_predict(image_hash, model_version)
```

## 8. Best Practices

1. **Standardize Imaging Conditions**: Consistent lighting, angles, backgrounds
2. **Calibrate Cameras**: Regular calibration for accurate measurements
3. **Handle Temporal Data**: Account for growth and seasonal variations
4. **Domain Adaptation**: Adapt models trained in lab to field conditions
5. **Optimize for Production**: Quantize and optimize models for speed
6. **Monitor Performance**: Track model accuracy and inference time
7. **Version Control**: Version models and preprocessing pipelines
8. **Documentation**: Document all parameters and assumptions

## Next Steps

Proceed to [Module 4: Data Engineering Fundamentals](04-data-engineering-fundamentals.md) to learn about data management and pipelines.

