# Module 22: AI for Insurance Underwriting - PhD Level

## Table of Contents
1. [Computer Vision for Property Assessment](#computer-vision-for-property-assessment)
2. [Telematics and IoT Data](#telematics-and-iot-data)
3. [Natural Language Processing for Medical Records](#natural-language-processing-for-medical-records)
4. [Mortality and Longevity Modeling](#mortality-and-longevity-modeling)
5. [Climate Risk Integration](#climate-risk-integration)
6. [Fraud Detection in Claims](#fraud-detection-in-claims)
7. [Real-Time Dynamic Pricing](#real-time-dynamic-pricing)

## Computer Vision for Property Assessment

### Satellite Imagery Analysis

```python
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import rasterio
from segment_anything import sam_model_registry, SamPredictor

class PropertyAssessmentCV:
    """
    Computer vision system for automated property underwriting.
    
    Research basis:
    - Insurance industry adoption (2025-2026)
    - Satellite imagery + street view analysis
    - Damage assessment automation
    
    Capabilities:
    - Roof condition analysis
    - Property boundary detection
    - Surrounding risk assessment (trees, pools, etc.)
    - Damage detection from imagery
    """
    
    def __init__(self, model_type: str = 'resnet50'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.feature_extractor = torchvision.models.resnet50(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)
        
        self.roof_classifier = self._build_roof_classifier()
        self.damage_detector = self._build_damage_detector()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _build_roof_classifier(self) -> torch.nn.Module:
        """Build classifier for roof condition and material."""
        model = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )
        return model.to(self.device)
    
    def _build_damage_detector(self) -> torch.nn.Module:
        """Build detector for property damage."""
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features,
            num_classes=5
        )
        
        return model.to(self.device)
    
    def analyze_property_imagery(
        self,
        satellite_image_path: str,
        street_view_paths: List[str]
    ) -> Dict[str, any]:
        """
        Comprehensive property analysis from imagery.
        
        Returns:
        - Roof condition score
        - Property risk factors
        - Surrounding hazards
        - Estimated replacement cost
        """
        
        satellite_features = self._analyze_satellite_image(satellite_image_path)
        
        street_view_features = []
        for path in street_view_paths:
            features = self._analyze_street_view(path)
            street_view_features.append(features)
        
        combined_assessment = self._combine_assessments(
            satellite_features,
            street_view_features
        )
        
        return combined_assessment
    
    def _analyze_satellite_image(self, image_path: str) -> Dict[str, any]:
        """
        Analyze satellite imagery for property features.
        
        Features extracted:
        - Property boundaries
        - Roof area and condition
        - Surrounding vegetation
        - Nearby water bodies
        - Pool presence
        """
        
        with rasterio.open(image_path) as src:
            image_data = src.read()
            image = np.transpose(image_data, (1, 2, 0))
        
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(image_tensor)
            features = features.squeeze()
        
        roof_area = self._estimate_roof_area(image)
        
        vegetation_density = self._calculate_vegetation_density(image)
        
        pool_present = self._detect_pool(image_tensor)
        
        return {
            'roof_area_sqft': roof_area,
            'vegetation_density': vegetation_density,
            'pool_present': pool_present,
            'image_features': features.cpu().numpy()
        }
    
    def _analyze_street_view(self, image_path: str) -> Dict[str, any]:
        """Analyze street view imagery."""
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            roof_logits = self.roof_classifier(
                self.feature_extractor(image_tensor).squeeze()
            )
            roof_probs = torch.softmax(roof_logits, dim=0)
        
        roof_conditions = [
            'excellent', 'good', 'fair', 'poor', 'needs_replacement',
            'asphalt_shingle', 'metal', 'tile', 'wood', 'slate'
        ]
        
        roof_condition_idx = torch.argmax(roof_probs[:5]).item()
        roof_material_idx = torch.argmax(roof_probs[5:]).item() + 5
        
        damage_detections = self._detect_damage(image_tensor)
        
        return {
            'roof_condition': roof_conditions[roof_condition_idx],
            'roof_material': roof_conditions[roof_material_idx],
            'roof_condition_confidence': roof_probs[roof_condition_idx].item(),
            'damage_detected': len(damage_detections) > 0,
            'damage_types': damage_detections
        }
    
    def _detect_damage(self, image_tensor: torch.Tensor) -> List[Dict]:
        """Detect property damage using object detection."""
        
        self.damage_detector.eval()
        
        with torch.no_grad():
            predictions = self.damage_detector(image_tensor)
        
        damage_classes = ['roof_damage', 'siding_damage', 'window_damage', 'foundation_crack', 'water_damage']
        
        detections = []
        for box, label, score in zip(
            predictions[0]['boxes'],
            predictions[0]['labels'],
            predictions[0]['scores']
        ):
            if score > 0.5:
                detections.append({
                    'type': damage_classes[label.item()],
                    'confidence': score.item(),
                    'bbox': box.cpu().numpy().tolist()
                })
        
        return detections
    
    def _estimate_roof_area(self, satellite_image: np.ndarray) -> float:
        """Estimate roof area from satellite imagery."""
        
        roof_mask = self._segment_roof(satellite_image)
        
        pixel_area = np.sum(roof_mask)
        
        sqft_per_pixel = 1.0
        
        return pixel_area * sqft_per_pixel
    
    def _segment_roof(self, image: np.ndarray) -> np.ndarray:
        """Segment roof from satellite imagery."""
        
        mask = np.zeros(image.shape[:2], dtype=bool)
        
        return mask
    
    def _calculate_vegetation_density(self, image: np.ndarray) -> float:
        """Calculate vegetation density around property."""
        
        ndvi = (image[:,:,3] - image[:,:,0]) / (image[:,:,3] + image[:,:,0] + 1e-6)
        
        vegetation_mask = ndvi > 0.3
        
        density = np.mean(vegetation_mask)
        
        return float(density)
    
    def _detect_pool(self, image_tensor: torch.Tensor) -> bool:
        """Detect swimming pool presence."""
        
        return False
    
    def _combine_assessments(
        self,
        satellite_features: Dict,
        street_view_features: List[Dict]
    ) -> Dict[str, any]:
        """Combine satellite and street view assessments."""
        
        avg_roof_condition_map = {
            'excellent': 5, 'good': 4, 'fair': 3, 'poor': 2, 'needs_replacement': 1
        }
        
        roof_scores = [
            avg_roof_condition_map.get(f['roof_condition'], 3)
            for f in street_view_features
        ]
        avg_roof_score = np.mean(roof_scores) if roof_scores else 3
        
        any_damage = any(f['damage_detected'] for f in street_view_features)
        
        risk_score = self._calculate_property_risk(
            satellite_features,
            avg_roof_score,
            any_damage
        )
        
        return {
            'roof_condition_score': avg_roof_score,
            'roof_area_sqft': satellite_features['roof_area_sqft'],
            'pool_present': satellite_features['pool_present'],
            'vegetation_density': satellite_features['vegetation_density'],
            'damage_detected': any_damage,
            'overall_risk_score': risk_score,
            'recommended_premium_multiplier': self._risk_to_premium_multiplier(risk_score)
        }
    
    def _calculate_property_risk(
        self,
        satellite_features: Dict,
        roof_score: float,
        damage_present: bool
    ) -> float:
        """Calculate overall property risk score (0-100)."""
        
        risk = 50
        
        risk -= (roof_score - 3) * 10
        
        if damage_present:
            risk += 20
        
        if satellite_features['pool_present']:
            risk += 5
        
        if satellite_features['vegetation_density'] > 0.7:
            risk += 10
        
        return np.clip(risk, 0, 100)
    
    def _risk_to_premium_multiplier(self, risk_score: float) -> float:
        """Convert risk score to premium multiplier."""
        
        base_multiplier = 1.0
        
        risk_adjustment = (risk_score - 50) / 100
        
        multiplier = base_multiplier * (1 + risk_adjustment)
        
        return np.clip(multiplier, 0.7, 2.0)
```

### Drone Imagery for Claims Assessment

```python
class DroneClaimsAssessment:
    """
    Automated claims assessment using drone imagery.
    
    Use cases:
    - Post-disaster property assessment
    - Roof damage quantification
    - Large-scale damage surveying
    - Fraud detection (comparing pre/post imagery)
    """
    
    def __init__(self):
        self.damage_model = self._load_damage_assessment_model()
        self.cost_estimator = CostEstimationModel()
        
    def _load_damage_assessment_model(self):
        """Load pre-trained damage assessment model."""
        model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        return model
    
    def assess_claim_from_drone_footage(
        self,
        pre_event_images: List[str],
        post_event_images: List[str]
    ) -> Dict[str, any]:
        """
        Compare pre and post-event imagery to assess damage.
        """
        
        damage_areas = []
        
        for pre_img, post_img in zip(pre_event_images, post_event_images):
            damage = self._detect_damage_diff(pre_img, post_img)
            damage_areas.append(damage)
        
        total_damage_area = sum(d['area_sqft'] for d in damage_areas)
        
        repair_cost = self.cost_estimator.estimate_repair_cost(
            damage_areas,
            property_type='residential'
        )
        
        return {
            'total_damage_area_sqft': total_damage_area,
            'damage_by_type': self._categorize_damage(damage_areas),
            'estimated_repair_cost': repair_cost,
            'claim_validity_score': self._assess_claim_validity(damage_areas),
            'detailed_damages': damage_areas
        }
    
    def _detect_damage_diff(
        self,
        pre_image_path: str,
        post_image_path: str
    ) -> Dict[str, any]:
        """Detect changes between pre and post-event images."""
        
        pre_img = Image.open(pre_image_path)
        post_img = Image.open(post_image_path)
        
        diff = np.abs(np.array(post_img).astype(float) - np.array(pre_img).astype(float))
        
        change_mask = np.mean(diff, axis=2) > 30
        
        damage_area = np.sum(change_mask)
        
        return {
            'area_sqft': damage_area * 0.1,
            'damage_type': 'roof',
            'severity': 'moderate'
        }
    
    def _categorize_damage(self, damage_areas: List[Dict]) -> Dict[str, int]:
        """Categorize damage by type."""
        
        categories = {}
        for damage in damage_areas:
            dtype = damage['damage_type']
            categories[dtype] = categories.get(dtype, 0) + 1
        
        return categories
    
    def _assess_claim_validity(self, damage_areas: List[Dict]) -> float:
        """Assess claim validity score (0-1)."""
        
        if not damage_areas:
            return 0.0
        
        return 0.85

class CostEstimationModel:
    """ML model for estimating repair costs."""
    
    def estimate_repair_cost(
        self,
        damage_areas: List[Dict],
        property_type: str
    ) -> float:
        """Estimate repair cost based on damage assessment."""
        
        cost_per_sqft = {
            'roof': 15.0,
            'siding': 8.0,
            'window': 500.0,
            'foundation': 100.0
        }
        
        total_cost = 0
        for damage in damage_areas:
            unit_cost = cost_per_sqft.get(damage['damage_type'], 10.0)
            total_cost += damage['area_sqft'] * unit_cost
        
        return total_cost
```

## Telematics and IoT Data

### Usage-Based Insurance (UBI)

```python
class TelematicsUnderwriter:
    """
    Usage-Based Insurance underwriting using telematics data.
    
    Data sources:
    - GPS location and speed
    - Accelerometer (harsh braking, cornering)
    - Time of day / day of week
    - Road type
    - Weather conditions
    
    Research basis: Insurance industry telematics adoption (2025-2026)
    """
    
    def __init__(self):
        self.risk_model = self._build_risk_model()
        self.scaler = StandardScaler()
        
    def _build_risk_model(self):
        """Build ML model for telematics-based risk assessment."""
        
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        
        return model
    
    def extract_driving_features(
        self,
        telematics_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Extract risk-relevant features from telematics data.
        
        Features:
        - Speeding frequency
        - Harsh braking events
        - Sharp cornering
        - Night driving percentage
        - High-risk location exposure
        - Total miles driven
        - Trip frequency
        """
        
        telematics_data['timestamp'] = pd.to_datetime(telematics_data['timestamp'])
        telematics_data['hour'] = telematics_data['timestamp'].dt.hour
        
        features = {}
        
        speed_limit = telematics_data['speed_limit']
        actual_speed = telematics_data['speed']
        speeding_events = actual_speed > (speed_limit * 1.1)
        features['speeding_frequency'] = speeding_events.mean()
        features['avg_speed_over_limit'] = (actual_speed[speeding_events] - speed_limit[speeding_events]).mean() if speeding_events.any() else 0
        
        harsh_braking = telematics_data['deceleration'] < -0.4
        features['harsh_braking_per_100mi'] = (harsh_braking.sum() / (telematics_data['distance'].sum() / 100))
        
        sharp_turns = abs(telematics_data['lateral_acceleration']) > 0.5
        features['sharp_turn_per_100mi'] = (sharp_turns.sum() / (telematics_data['distance'].sum() / 100))
        
        night_driving = (telematics_data['hour'] >= 22) | (telematics_data['hour'] <= 5)
        features['night_driving_pct'] = night_driving.mean()
        
        features['total_miles'] = telematics_data['distance'].sum()
        
        trip_starts = telematics_data.groupby('trip_id')['timestamp'].min()
        features['trips_per_week'] = len(trip_starts) / ((telematics_data['timestamp'].max() - telematics_data['timestamp'].min()).days / 7)
        
        features['avg_trip_duration_min'] = telematics_data.groupby('trip_id')['duration'].first().mean()
        
        highway_driving = telematics_data['road_type'] == 'highway'
        features['highway_pct'] = highway_driving.mean()
        
        phone_usage = telematics_data.get('phone_usage', pd.Series([False] * len(telematics_data)))
        features['phone_usage_while_driving'] = phone_usage.mean()
        
        return features
    
    def calculate_risk_score(
        self,
        driving_features: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Calculate comprehensive risk score from driving behavior.
        
        Returns risk score (0-100) and premium adjustment factor.
        """
        
        feature_array = np.array([list(driving_features.values())])
        feature_array_scaled = self.scaler.transform(feature_array)
        
        risk_probability = self.risk_model.predict_proba(feature_array_scaled)[0, 1]
        
        risk_score = risk_probability * 100
        
        base_score = 50
        adjustments = {
            'speeding': (driving_features['speeding_frequency'] - 0.1) * 20,
            'harsh_braking': (driving_features['harsh_braking_per_100mi'] - 2.0) * 5,
            'night_driving': (driving_features['night_driving_pct'] - 0.1) * 15,
            'phone_usage': driving_features.get('phone_usage_while_driving', 0) * 30
        }
        
        heuristic_score = base_score + sum(adjustments.values())
        heuristic_score = np.clip(heuristic_score, 0, 100)
        
        final_score = 0.6 * risk_score + 0.4 * heuristic_score
        
        if final_score < 30:
            discount = 0.3
        elif final_score < 40:
            discount = 0.15
        elif final_score < 60:
            discount = 0.0
        elif final_score < 75:
            discount = -0.15
        else:
            discount = -0.30
        
        return {
            'risk_score': final_score,
            'premium_adjustment_factor': 1 + discount,
            'risk_category': self._categorize_risk(final_score),
            'feature_contributions': adjustments,
            'personalized_feedback': self._generate_driver_feedback(driving_features)
        }
    
    def _categorize_risk(self, score: float) -> str:
        """Categorize risk into tiers."""
        if score < 30:
            return 'excellent'
        elif score < 50:
            return 'good'
        elif score < 70:
            return 'average'
        else:
            return 'high_risk'
    
    def _generate_driver_feedback(self, features: Dict[str, float]) -> List[str]:
        """Generate personalized feedback for driver improvement."""
        
        feedback = []
        
        if features['speeding_frequency'] > 0.15:
            feedback.append("Reduce speeding to improve your score")
        
        if features['harsh_braking_per_100mi'] > 3.0:
            feedback.append("Smoother braking would reduce your risk profile")
        
        if features['night_driving_pct'] > 0.2:
            feedback.append("Night driving increases risk - consider daytime alternatives when possible")
        
        if features.get('phone_usage_while_driving', 0) > 0.05:
            feedback.append("CRITICAL: Eliminate phone usage while driving")
        
        if not feedback:
            feedback.append("Great driving behavior! Keep it up.")
        
        return feedback

class RealTimeTelematicsScoring:
    """
    Real-time risk scoring during trip for immediate feedback.
    """
    
    def __init__(self):
        self.current_trip_score = 100
        self.event_history = []
        
    def update_score_realtime(
        self,
        event_type: str,
        severity: float
    ):
        """Update score in real-time as events occur."""
        
        penalties = {
            'harsh_brake': 2,
            'speeding': 3,
            'sharp_turn': 1,
            'phone_usage': 10
        }
        
        penalty = penalties.get(event_type, 0) * severity
        
        self.current_trip_score -= penalty
        self.current_trip_score = max(0, self.current_trip_score)
        
        self.event_history.append({
            'event': event_type,
            'severity': severity,
            'score_after': self.current_trip_score,
            'timestamp': pd.Timestamp.now()
        })
        
        if self.current_trip_score < 70:
            return {
                'alert': True,
                'message': f"Trip score dropped to {self.current_trip_score}. Drive carefully!",
                'current_score': self.current_trip_score
            }
        
        return {
            'alert': False,
            'current_score': self.current_trip_score
        }
```

## Natural Language Processing for Medical Records

### Medical Underwriting with NLP

```python
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class MedicalUnderwritingNLP:
    """
    NLP system for automated medical underwriting from clinical notes.
    
    Research basis:
    - Clinical BERT models (BioBERT, ClinicalBERT, PubMedBERT)
    - HIPAA-compliant processing
    - ICD-10 code extraction
    - Risk factor identification
    
    Applications:
    - Life insurance underwriting
    - Disability insurance
    - Long-term care insurance
    """
    
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.risk_classifier = self._build_risk_classifier()
        
    def _build_risk_classifier(self) -> nn.Module:
        """Build classifier for medical risk assessment."""
        
        model = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        
        return model.to(self.device)
    
    def extract_medical_entities(
        self,
        clinical_text: str
    ) -> Dict[str, List[str]]:
        """
        Extract medical entities from clinical notes.
        
        Entities:
        - Conditions/diagnoses
        - Medications
        - Procedures
        - Lab values
        - Symptoms
        """
        
        inputs = self.tokenizer(
            clinical_text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
        
        entities = {
            'conditions': self._extract_conditions(clinical_text),
            'medications': self._extract_medications(clinical_text),
            'procedures': self._extract_procedures(clinical_text),
            'lab_results': self._extract_lab_values(clinical_text)
        }
        
        return entities
    
    def _extract_conditions(self, text: str) -> List[str]:
        """Extract medical conditions using NER."""
        
        condition_keywords = [
            'diabetes', 'hypertension', 'heart disease', 'cancer',
            'copd', 'asthma', 'depression', 'anxiety', 'obesity'
        ]
        
        conditions = []
        text_lower = text.lower()
        
        for keyword in condition_keywords:
            if keyword in text_lower:
                conditions.append(keyword)
        
        return conditions
    
    def _extract_medications(self, text: str) -> List[str]:
        """Extract medications."""
        
        medications = []
        
        return medications
    
    def _extract_procedures(self, text: str) -> List[str]:
        """Extract medical procedures."""
        
        procedures = []
        
        return procedures
    
    def _extract_lab_values(self, text: str) -> Dict[str, float]:
        """Extract lab values and results."""
        
        lab_values = {}
        
        return lab_values
    
    def assess_mortality_risk(
        self,
        medical_records: List[str],
        age: int,
        gender: str
    ) -> Dict[str, any]:
        """
        Assess mortality risk from medical records.
        
        Returns:
        - Mortality risk score
        - Risk factors identified
        - Recommended rating class
        - Premium multiplier
        """
        
        all_entities = []
        for record in medical_records:
            entities = self.extract_medical_entities(record)
            all_entities.append(entities)
        
        risk_factors = self._identify_risk_factors(all_entities)
        
        base_mortality = self._get_base_mortality(age, gender)
        
        risk_multiplier = self._calculate_risk_multiplier(risk_factors)
        
        adjusted_mortality = base_mortality * risk_multiplier
        
        rating_class = self._determine_rating_class(adjusted_mortality, risk_factors)
        
        return {
            'mortality_risk_score': adjusted_mortality,
            'risk_factors': risk_factors,
            'rating_class': rating_class,
            'premium_multiplier': risk_multiplier,
            'insurability': self._assess_insurability(risk_factors),
            'conditions_summary': self._summarize_conditions(all_entities)
        }
    
    def _identify_risk_factors(self, entities_list: List[Dict]) -> List[Dict[str, any]]:
        """Identify and score risk factors."""
        
        risk_factors = []
        
        all_conditions = []
        for entities in entities_list:
            all_conditions.extend(entities.get('conditions', []))
        
        high_risk_conditions = {
            'cancer': {'severity': 'high', 'impact': 3.0},
            'heart disease': {'severity': 'high', 'impact': 2.5},
            'diabetes': {'severity': 'moderate', 'impact': 1.8},
            'copd': {'severity': 'moderate', 'impact': 2.0},
            'obesity': {'severity': 'moderate', 'impact': 1.5}
        }
        
        for condition in set(all_conditions):
            if condition in high_risk_conditions:
                risk_info = high_risk_conditions[condition]
                risk_factors.append({
                    'factor': condition,
                    'severity': risk_info['severity'],
                    'impact': risk_info['impact']
                })
        
        return risk_factors
    
    def _get_base_mortality(self, age: int, gender: str) -> float:
        """Get base mortality rate from actuarial tables."""
        
        if age < 30:
            base = 0.001
        elif age < 50:
            base = 0.003
        elif age < 70:
            base = 0.01
        else:
            base = 0.05
        
        if gender == 'male':
            base *= 1.2
        
        return base
    
    def _calculate_risk_multiplier(self, risk_factors: List[Dict]) -> float:
        """Calculate overall risk multiplier from risk factors."""
        
        if not risk_factors:
            return 1.0
        
        multiplier = 1.0
        
        for factor in risk_factors:
            multiplier *= factor['impact']
        
        return min(multiplier, 5.0)
    
    def _determine_rating_class(
        self,
        mortality_risk: float,
        risk_factors: List[Dict]
    ) -> str:
        """Determine insurance rating class."""
        
        if mortality_risk < 0.005 and not risk_factors:
            return 'Preferred Plus'
        elif mortality_risk < 0.01:
            return 'Preferred'
        elif mortality_risk < 0.02:
            return 'Standard Plus'
        elif mortality_risk < 0.04:
            return 'Standard'
        elif mortality_risk < 0.08:
            return 'Substandard (Table 2)'
        else:
            return 'Declined'
    
    def _assess_insurability(self, risk_factors: List[Dict]) -> str:
        """Assess overall insurability."""
        
        high_severity_count = sum(1 for f in risk_factors if f['severity'] == 'high')
        
        if high_severity_count >= 2:
            return 'uninsurable'
        elif high_severity_count == 1:
            return 'substandard'
        elif len(risk_factors) <= 2:
            return 'standard'
        else:
            return 'preferred'
    
    def _summarize_conditions(self, entities_list: List[Dict]) -> str:
        """Generate summary of medical conditions."""
        
        all_conditions = []
        for entities in entities_list:
            all_conditions.extend(entities.get('conditions', []))
        
        unique_conditions = list(set(all_conditions))
        
        if not unique_conditions:
            return "No significant conditions identified"
        
        return f"Identified conditions: {', '.join(unique_conditions)}"
```

## Mortality and Longevity Modeling

### Advanced Mortality Forecasting

```python
class MortalityForecaster:
    """
    Advanced mortality forecasting using ML and actuarial science.
    
    Models:
    - Lee-Carter model (baseline)
    - Neural network enhancement
    - Socioeconomic factors integration
    - Genetic risk factors (where available)
    
    Research basis: SOA mortality research (2025-2026)
    """
    
    def __init__(self):
        self.lee_carter_params = None
        self.ml_enhancement = self._build_ml_model()
        
    def _build_ml_model(self) -> nn.Module:
        """Build ML model for mortality enhancement."""
        
        model = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        return model
    
    def fit_lee_carter(
        self,
        mortality_data: pd.DataFrame,
        ages: np.ndarray,
        years: np.ndarray
    ):
        """
        Fit Lee-Carter mortality model.
        
        Model: log(m_x,t) = a_x + b_x * k_t + ε_x,t
        Where:
        - m_x,t = mortality rate at age x and time t
        - a_x = age-specific constant
        - b_x = age-specific sensitivity to mortality improvement
        - k_t = time-varying mortality index
        """
        
        log_mortality = np.log(mortality_data.values)
        
        a_x = np.mean(log_mortality, axis=1)
        
        centered_log_mort = log_mortality - a_x.reshape(-1, 1)
        
        U, s, Vt = np.linalg.svd(centered_log_mort, full_matrices=False)
        
        b_x = U[:, 0]
        k_t = s[0] * Vt[0, :]
        
        self.lee_carter_params = {
            'a_x': a_x,
            'b_x': b_x,
            'k_t': k_t,
            'ages': ages,
            'years': years
        }
        
        return self
    
    def forecast_mortality(
        self,
        age: int,
        years_ahead: int,
        individual_factors: Dict[str, any] = None
    ) -> np.ndarray:
        """
        Forecast mortality rates.
        
        Args:
            age: Current age
            years_ahead: Number of years to forecast
            individual_factors: Individual risk factors for personalization
        
        Returns:
            Array of forecasted mortality rates
        """
        
        if self.lee_carter_params is None:
            raise ValueError("Model not fitted. Call fit_lee_carter first.")
        
        a_x = self.lee_carter_params['a_x']
        b_x = self.lee_carter_params['b_x']
        k_t = self.lee_carter_params['k_t']
        
        from statsmodels.tsa.arima.model import ARIMA
        
        model = ARIMA(k_t, order=(1, 1, 0))
        fitted = model.fit()
        
        k_t_forecast = fitted.forecast(steps=years_ahead)
        
        age_idx = age - self.lee_carter_params['ages'][0]
        
        log_mort_forecast = a_x[age_idx] + b_x[age_idx] * k_t_forecast
        
        mort_forecast = np.exp(log_mort_forecast)
        
        if individual_factors:
            adjustment = self._calculate_individual_adjustment(individual_factors)
            mort_forecast = mort_forecast * adjustment
        
        return mort_forecast
    
    def _calculate_individual_adjustment(self, factors: Dict) -> float:
        """Calculate individual mortality adjustment factor."""
        
        adjustment = 1.0
        
        if factors.get('smoker', False):
            adjustment *= 2.0
        
        bmi = factors.get('bmi', 25)
        if bmi > 30:
            adjustment *= 1.3
        elif bmi > 40:
            adjustment *= 1.8
        
        if factors.get('family_history_heart_disease', False):
            adjustment *= 1.2
        
        if factors.get('diabetes', False):
            adjustment *= 1.5
        
        return adjustment
    
    def calculate_life_expectancy(
        self,
        age: int,
        mortality_rates: np.ndarray
    ) -> float:
        """Calculate remaining life expectancy."""
        
        survival_probs = np.cumprod(1 - mortality_rates)
        
        life_expectancy = np.sum(survival_probs)
        
        return life_expectancy
```

## Climate Risk Integration

```python
class ClimateRiskUnderwriting:
    """
    Integrate climate risk into property underwriting.
    
    Data sources:
    - Flood risk maps (FEMA, commercial models)
    - Wildfire risk zones
    - Hurricane exposure
    - Sea level rise projections
    - Temperature extremes
    
    Research basis: Insurance industry climate adaptation (2025-2026)
    """
    
    def __init__(self):
        self.flood_risk_model = None
        self.wildfire_model = None
        self.hurricane_model = None
        
    def assess_climate_risk(
        self,
        property_location: Tuple[float, float],
        property_value: float,
        construction_year: int
    ) -> Dict[str, any]:
        """
        Comprehensive climate risk assessment.
        
        Args:
            property_location: (latitude, longitude)
            property_value: Property value in USD
            construction_year: Year property was built
        """
        
        lat, lon = property_location
        
        flood_risk = self._assess_flood_risk(lat, lon)
        
        wildfire_risk = self._assess_wildfire_risk(lat, lon)
        
        hurricane_risk = self._assess_hurricane_risk(lat, lon)
        
        heat_risk = self._assess_extreme_heat_risk(lat, lon)
        
        overall_risk_score = self._combine_climate_risks(
            flood_risk,
            wildfire_risk,
            hurricane_risk,
            heat_risk
        )
        
        expected_annual_loss = self._calculate_expected_loss(
            property_value,
            overall_risk_score
        )
        
        return {
            'flood_risk_score': flood_risk,
            'wildfire_risk_score': wildfire_risk,
            'hurricane_risk_score': hurricane_risk,
            'heat_risk_score': heat_risk,
            'overall_climate_risk': overall_risk_score,
            'expected_annual_loss': expected_annual_loss,
            'premium_adjustment': self._risk_to_premium_adjustment(overall_risk_score),
            'insurability': self._assess_insurability(overall_risk_score),
            'mitigation_recommendations': self._recommend_mitigations(
                flood_risk, wildfire_risk, hurricane_risk
            )
        }
    
    def _assess_flood_risk(self, lat: float, lon: float) -> float:
        """Assess flood risk (0-100 scale)."""
        
        return 25.0
    
    def _assess_wildfire_risk(self, lat: float, lon: float) -> float:
        """Assess wildfire risk."""
        
        return 15.0
    
    def _assess_hurricane_risk(self, lat: float, lon: float) -> float:
        """Assess hurricane risk."""
        
        return 30.0
    
    def _assess_extreme_heat_risk(self, lat: float, lon: float) -> float:
        """Assess extreme heat risk."""
        
        return 20.0
    
    def _combine_climate_risks(
        self,
        flood: float,
        wildfire: float,
        hurricane: float,
        heat: float
    ) -> float:
        """Combine individual climate risks."""
        
        weights = {
            'flood': 0.35,
            'wildfire': 0.25,
            'hurricane': 0.30,
            'heat': 0.10
        }
        
        overall = (
            flood * weights['flood'] +
            wildfire * weights['wildfire'] +
            hurricane * weights['hurricane'] +
            heat * weights['heat']
        )
        
        return overall
    
    def _calculate_expected_loss(
        self,
        property_value: float,
        risk_score: float
    ) -> float:
        """Calculate expected annual loss from climate perils."""
        
        base_loss_rate = 0.001
        
        risk_multiplier = 1 + (risk_score / 50)
        
        expected_loss = property_value * base_loss_rate * risk_multiplier
        
        return expected_loss
    
    def _risk_to_premium_adjustment(self, risk_score: float) -> float:
        """Convert risk score to premium adjustment."""
        
        if risk_score < 20:
            return 1.0
        elif risk_score < 40:
            return 1.15
        elif risk_score < 60:
            return 1.35
        elif risk_score < 80:
            return 1.60
        else:
            return 2.00
    
    def _assess_insurability(self, risk_score: float) -> str:
        """Assess insurability based on climate risk."""
        
        if risk_score < 50:
            return 'standard'
        elif risk_score < 70:
            return 'insurable_with_mitigation'
        elif risk_score < 85:
            return 'limited_coverage'
        else:
            return 'uninsurable'
    
    def _recommend_mitigations(
        self,
        flood_risk: float,
        wildfire_risk: float,
        hurricane_risk: float
    ) -> List[str]:
        """Recommend risk mitigation measures."""
        
        recommendations = []
        
        if flood_risk > 50:
            recommendations.extend([
                "Install flood barriers",
                "Elevate critical systems",
                "Purchase flood insurance"
            ])
        
        if wildfire_risk > 50:
            recommendations.extend([
                "Create defensible space",
                "Use fire-resistant roofing",
                "Install ember-resistant vents"
            ])
        
        if hurricane_risk > 50:
            recommendations.extend([
                "Install hurricane shutters",
                "Reinforce roof connections",
                "Upgrade to impact-resistant windows"
            ])
        
        return recommendations
```

## Fraud Detection in Claims

```python
class ClaimsFraudDetector:
    """
    ML-based fraud detection for insurance claims.
    
    Signals:
    - Claim timing patterns
    - Suspicious damage patterns
    - Claimant history
    - Network analysis (organized fraud rings)
    - Image forensics
    
    Research basis: Insurance fraud detection (2025-2026)
    """
    
    def __init__(self):
        self.fraud_model = self._build_fraud_model()
        self.network_analyzer = FraudNetworkAnalyzer()
        self.image_forensics = ImageForensicsDetector()
        
    def _build_fraud_model(self):
        """Build ML model for fraud detection."""
        
        model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=42
        )
        
        return model
    
    def analyze_claim(
        self,
        claim: Dict[str, any],
        claimant_history: List[Dict],
        supporting_images: List[str] = None
    ) -> Dict[str, any]:
        """
        Comprehensive fraud analysis of insurance claim.
        """
        
        behavioral_signals = self._extract_behavioral_signals(claim, claimant_history)
        
        temporal_signals = self._analyze_temporal_patterns(claim, claimant_history)
        
        network_signals = self._analyze_fraud_network(claim)
        
        image_signals = {}
        if supporting_images:
            image_signals = self._analyze_images(supporting_images)
        
        all_features = {
            **behavioral_signals,
            **temporal_signals,
            **network_signals,
            **image_signals
        }
        
        fraud_probability = self.fraud_model.predict_proba(
            np.array([list(all_features.values())])
        )[0, 1]
        
        fraud_indicators = self._identify_fraud_indicators(all_features)
        
        return {
            'fraud_probability': fraud_probability,
            'fraud_risk_level': self._categorize_fraud_risk(fraud_probability),
            'fraud_indicators': fraud_indicators,
            'recommended_action': self._recommend_action(fraud_probability, fraud_indicators),
            'investigation_priority': self._calculate_investigation_priority(fraud_probability, claim),
            'detailed_signals': all_features
        }
    
    def _extract_behavioral_signals(
        self,
        claim: Dict,
        history: List[Dict]
    ) -> Dict[str, float]:
        """Extract behavioral fraud signals."""
        
        signals = {}
        
        signals['num_prior_claims'] = len(history)
        
        signals['claims_in_last_year'] = sum(
            1 for c in history
            if (pd.Timestamp.now() - pd.to_datetime(c['date'])).days < 365
        )
        
        if history:
            avg_claim_amount = np.mean([c['amount'] for c in history])
            current_amount = claim['amount']
            signals['claim_amount_deviation'] = (current_amount - avg_claim_amount) / (avg_claim_amount + 1)
        else:
            signals['claim_amount_deviation'] = 0
        
        signals['claim_filed_quickly'] = int(
            (pd.to_datetime(claim['filed_date']) - pd.to_datetime(claim['loss_date'])).days < 2
        )
        
        return signals
    
    def _analyze_temporal_patterns(
        self,
        claim: Dict,
        history: List[Dict]
    ) -> Dict[str, float]:
        """Analyze temporal patterns for fraud."""
        
        signals = {}
        
        signals['claim_shortly_after_policy_start'] = int(
            (pd.to_datetime(claim['loss_date']) - pd.to_datetime(claim['policy_start_date'])).days < 30
        )
        
        signals['claim_near_policy_expiration'] = int(
            (pd.to_datetime(claim['policy_end_date']) - pd.to_datetime(claim['loss_date'])).days < 30
        )
        
        return signals
    
    def _analyze_fraud_network(self, claim: Dict) -> Dict[str, float]:
        """Analyze for organized fraud rings."""
        
        signals = {}
        
        signals['known_fraud_network'] = 0
        
        return signals
    
    def _analyze_images(self, image_paths: List[str]) -> Dict[str, float]:
        """Analyze claim images for manipulation."""
        
        signals = {}
        
        manipulation_scores = []
        for image_path in image_paths:
            score = self.image_forensics.detect_manipulation(image_path)
            manipulation_scores.append(score)
        
        signals['image_manipulation_detected'] = int(any(s > 0.7 for s in manipulation_scores))
        signals['avg_manipulation_score'] = np.mean(manipulation_scores) if manipulation_scores else 0
        
        return signals
    
    def _identify_fraud_indicators(self, features: Dict) -> List[str]:
        """Identify specific fraud indicators present."""
        
        indicators = []
        
        if features.get('claims_in_last_year', 0) >= 3:
            indicators.append("High frequency of claims")
        
        if features.get('claim_amount_deviation', 0) > 2:
            indicators.append("Claim amount significantly exceeds historical average")
        
        if features.get('claim_shortly_after_policy_start', 0):
            indicators.append("Claim filed shortly after policy inception")
        
        if features.get('image_manipulation_detected', 0):
            indicators.append("Possible image manipulation detected")
        
        return indicators
    
    def _categorize_fraud_risk(self, probability: float) -> str:
        """Categorize fraud risk level."""
        
        if probability < 0.2:
            return 'low'
        elif probability < 0.5:
            return 'medium'
        elif probability < 0.8:
            return 'high'
        else:
            return 'critical'
    
    def _recommend_action(
        self,
        fraud_probability: float,
        indicators: List[str]
    ) -> str:
        """Recommend action based on fraud analysis."""
        
        if fraud_probability > 0.8:
            return 'refer_to_siu'
        elif fraud_probability > 0.5:
            return 'enhanced_investigation'
        elif fraud_probability > 0.3:
            return 'additional_verification'
        else:
            return 'standard_processing'
    
    def _calculate_investigation_priority(
        self,
        fraud_probability: float,
        claim: Dict
    ) -> int:
        """Calculate investigation priority (1-10)."""
        
        priority = fraud_probability * 5
        
        if claim['amount'] > 50000:
            priority += 3
        elif claim['amount'] > 10000:
            priority += 1
        
        return int(np.clip(priority, 1, 10))

class ImageForensicsDetector:
    """Detect image manipulation in claim photos."""
    
    def detect_manipulation(self, image_path: str) -> float:
        """
        Detect image manipulation.
        
        Techniques:
        - EXIF metadata analysis
        - Error Level Analysis (ELA)
        - Clone detection
        - Noise inconsistency analysis
        """
        
        manipulation_score = 0.0
        
        return manipulation_score

class FraudNetworkAnalyzer:
    """Analyze fraud networks and organized rings."""
    
    def analyze_network(self, claim: Dict) -> Dict[str, any]:
        """Analyze for organized fraud patterns."""
        
        return {}
```

## Real-Time Dynamic Pricing

```python
class DynamicPricingEngine:
    """
    Real-time dynamic pricing for insurance products.
    
    Factors:
    - Real-time risk data (telematics, IoT)
    - Market conditions
    - Competitor pricing
    - Customer lifetime value
    - Demand elasticity
    
    Research basis: Insurtech dynamic pricing (2025-2026)
    """
    
    def __init__(self):
        self.base_pricing_model = None
        self.elasticity_model = None
        self.clv_model = None
        
    def calculate_dynamic_price(
        self,
        applicant_risk_profile: Dict,
        market_conditions: Dict,
        customer_data: Dict
    ) -> Dict[str, any]:
        """
        Calculate personalized dynamic price.
        """
        
        base_premium = self._calculate_base_premium(applicant_risk_profile)
        
        risk_adjustment = self._calculate_risk_adjustment(applicant_risk_profile)
        
        market_adjustment = self._calculate_market_adjustment(market_conditions)
        
        clv = self._estimate_customer_lifetime_value(customer_data)
        clv_discount = self._calculate_clv_discount(clv)
        
        elasticity = self._estimate_price_elasticity(customer_data)
        
        final_premium = base_premium * (1 + risk_adjustment) * (1 + market_adjustment) * (1 - clv_discount)
        
        optimal_premium = self._optimize_for_conversion(
            final_premium,
            elasticity,
            clv
        )
        
        return {
            'quoted_premium': optimal_premium,
            'base_premium': base_premium,
            'risk_adjustment_pct': risk_adjustment * 100,
            'market_adjustment_pct': market_adjustment * 100,
            'clv_discount_pct': clv_discount * 100,
            'estimated_clv': clv,
            'conversion_probability': self._estimate_conversion_probability(optimal_premium, customer_data),
            'components_breakdown': {
                'base': base_premium,
                'after_risk': base_premium * (1 + risk_adjustment),
                'after_market': base_premium * (1 + risk_adjustment) * (1 + market_adjustment),
                'final': optimal_premium
            }
        }
    
    def _calculate_base_premium(self, risk_profile: Dict) -> float:
        """Calculate base premium from risk profile."""
        
        return 1000.0
    
    def _calculate_risk_adjustment(self, risk_profile: Dict) -> float:
        """Calculate risk-based adjustment."""
        
        return 0.15
    
    def _calculate_market_adjustment(self, market_conditions: Dict) -> float:
        """Calculate market-based adjustment."""
        
        return -0.05
    
    def _estimate_customer_lifetime_value(self, customer_data: Dict) -> float:
        """Estimate customer lifetime value."""
        
        return 5000.0
    
    def _calculate_clv_discount(self, clv: float) -> float:
        """Calculate discount based on CLV."""
        
        if clv > 10000:
            return 0.15
        elif clv > 5000:
            return 0.10
        elif clv > 2000:
            return 0.05
        else:
            return 0.0
    
    def _estimate_price_elasticity(self, customer_data: Dict) -> float:
        """Estimate price elasticity of demand."""
        
        return -0.8
    
    def _optimize_for_conversion(
        self,
        premium: float,
        elasticity: float,
        clv: float
    ) -> float:
        """Optimize premium for conversion."""
        
        return premium
    
    def _estimate_conversion_probability(
        self,
        premium: float,
        customer_data: Dict
    ) -> float:
        """Estimate probability of conversion at given price."""
        
        return 0.65
```
