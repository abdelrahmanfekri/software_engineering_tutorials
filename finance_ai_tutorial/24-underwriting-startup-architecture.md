# Module 24: Building an AI Underwriting Startup - Part I: Technical Architecture

## Table of Contents
1. [Startup Technical Stack](#startup-technical-stack)
2. [Microservices Architecture](#microservices-architecture)
3. [Real-Time Data Pipelines](#real-time-data-pipelines)
4. [ML Model Deployment](#ml-model-deployment)
5. [API Design and Integration](#api-design-and-integration)
6. [Scalability and Performance](#scalability-and-performance)
7. [Security and Compliance](#security-and-compliance)

## Startup Technical Stack

### Modern Underwriting Platform Stack

```python
"""
Complete technical stack for world-class AI underwriting startup (2026).

Backend:
- Python 3.11+ (FastAPI, Pydantic, asyncio)
- Go (for high-performance services)
- Node.js/TypeScript (real-time services)

ML/AI:
- PyTorch (primary ML framework)
- TensorFlow (production serving)
- Hugging Face Transformers (NLP)
- Ray (distributed computing)
- MLflow (experiment tracking)
- Feast (feature store)

Data:
- PostgreSQL (transactional)
- TimescaleDB (time-series)
- MongoDB (document store)
- Redis (caching, real-time)
- Apache Kafka (event streaming)
- Apache Airflow (orchestration)

Infrastructure:
- Kubernetes (container orchestration)
- Docker (containerization)
- Terraform (infrastructure as code)
- AWS/GCP/Azure (cloud providers)
- Istio (service mesh)

Monitoring:
- Prometheus (metrics)
- Grafana (visualization)
- ELK Stack (logging)
- Sentry (error tracking)
- DataDog (APM)

Security:
- HashiCorp Vault (secrets management)
- OAuth 2.0 / JWT (authentication)
- AWS KMS (encryption)
- Snyk (vulnerability scanning)
"""

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
import asyncio
import aioredis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import structlog
from prometheus_client import Counter, Histogram, Gauge
import time

logger = structlog.get_logger()

class UnderwritingPlatformConfig(BaseModel):
    """
    Configuration for underwriting platform.
    Follows 12-factor app methodology.
    """
    
    api_version: str = "v1"
    environment: str = Field(..., regex="^(development|staging|production)$")
    
    database_url: str
    redis_url: str
    kafka_bootstrap_servers: List[str]
    
    ml_model_registry_url: str
    feature_store_url: str
    
    max_concurrent_applications: int = 1000
    request_timeout_seconds: int = 30
    
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 100
    
    enable_circuit_breaker: bool = True
    circuit_breaker_failure_threshold: int = 5
    
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 30

class ApplicationRequest(BaseModel):
    """Request model for underwriting application."""
    
    applicant_id: str
    application_type: str = Field(..., regex="^(credit|insurance|takaful)$")
    requested_amount: float = Field(gt=0)
    
    applicant_data: Dict[str, Any]
    financial_data: Optional[Dict[str, Any]] = None
    alternative_data: Optional[Dict[str, Any]] = None
    
    priority: str = Field(default="normal", regex="^(low|normal|high|urgent)$")
    
    @validator('applicant_data')
    def validate_applicant_data(cls, v):
        required_fields = ['income', 'age', 'employment_status']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required field: {field}")
        return v

class ApplicationResponse(BaseModel):
    """Response model for underwriting decision."""
    
    application_id: str
    decision: str
    decision_score: float
    confidence: float
    
    risk_factors: List[Dict[str, Any]]
    recommended_terms: Optional[Dict[str, Any]]
    
    processing_time_ms: float
    model_version: str
    
    requires_manual_review: bool
    review_reasons: Optional[List[str]]
    
    compliance_flags: List[str] = []
    shariah_compliant: Optional[bool] = None

prometheus_requests = Counter(
    'underwriting_requests_total',
    'Total underwriting requests',
    ['application_type', 'decision']
)

prometheus_latency = Histogram(
    'underwriting_latency_seconds',
    'Underwriting request latency',
    ['application_type']
)

prometheus_active_applications = Gauge(
    'underwriting_active_applications',
    'Number of applications currently being processed'
)

app = FastAPI(
    title="AI Underwriting Platform",
    description="World-class AI-powered underwriting system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class UnderwritingEngine:
    """
    Core underwriting engine with async processing.
    """
    
    def __init__(self, config: UnderwritingPlatformConfig):
        self.config = config
        self.model_registry = None
        self.feature_store = None
        self.redis_client = None
        
    async def initialize(self):
        """Initialize async resources."""
        self.redis_client = await aioredis.create_redis_pool(self.config.redis_url)
        
        logger.info("underwriting_engine_initialized")
        
    async def process_application(
        self,
        request: ApplicationRequest,
        session: AsyncSession
    ) -> ApplicationResponse:
        """
        Process underwriting application asynchronously.
        
        Steps:
        1. Fetch features from feature store
        2. Load appropriate ML model
        3. Generate prediction
        4. Apply business rules
        5. Check compliance
        6. Return decision
        """
        
        start_time = time.time()
        
        try:
            prometheus_active_applications.inc()
            
            cached_result = await self._check_cache(request.applicant_id)
            if cached_result:
                logger.info("cache_hit", applicant_id=request.applicant_id)
                return cached_result
            
            features = await self._fetch_features(request)
            
            model = await self._load_model(request.application_type)
            
            prediction = await self._generate_prediction(model, features)
            
            business_rules_result = await self._apply_business_rules(
                request,
                prediction
            )
            
            compliance_check = await self._check_compliance(
                request,
                prediction,
                business_rules_result
            )
            
            response = self._construct_response(
                request,
                prediction,
                business_rules_result,
                compliance_check,
                start_time
            )
            
            await self._cache_result(request.applicant_id, response)
            
            await self._emit_events(request, response)
            
            prometheus_requests.labels(
                application_type=request.application_type,
                decision=response.decision
            ).inc()
            
            prometheus_latency.labels(
                application_type=request.application_type
            ).observe(response.processing_time_ms / 1000)
            
            return response
            
        except Exception as e:
            logger.error("application_processing_error", error=str(e))
            raise HTTPException(status_code=500, detail="Internal processing error")
            
        finally:
            prometheus_active_applications.dec()
    
    async def _check_cache(self, applicant_id: str) -> Optional[ApplicationResponse]:
        """Check Redis cache for recent decision."""
        cache_key = f"underwriting:{applicant_id}"
        
        cached = await self.redis_client.get(cache_key)
        if cached:
            return ApplicationResponse.parse_raw(cached)
        
        return None
    
    async def _fetch_features(self, request: ApplicationRequest) -> Dict[str, Any]:
        """Fetch features from feature store."""
        
        features = {
            **request.applicant_data,
            **(request.financial_data or {}),
            **(request.alternative_data or {})
        }
        
        return features
    
    async def _load_model(self, application_type: str):
        """Load appropriate ML model from registry."""
        
        return None
    
    async def _generate_prediction(self, model, features: Dict) -> Dict[str, Any]:
        """Generate ML prediction."""
        
        return {
            'score': 0.75,
            'probability': 0.85,
            'confidence': 0.90
        }
    
    async def _apply_business_rules(
        self,
        request: ApplicationRequest,
        prediction: Dict
    ) -> Dict[str, Any]:
        """Apply business rules and policy constraints."""
        
        return {
            'passes_business_rules': True,
            'triggered_rules': []
        }
    
    async def _check_compliance(
        self,
        request: ApplicationRequest,
        prediction: Dict,
        business_rules: Dict
    ) -> Dict[str, Any]:
        """Check regulatory compliance."""
        
        return {
            'compliant': True,
            'flags': []
        }
    
    def _construct_response(
        self,
        request: ApplicationRequest,
        prediction: Dict,
        business_rules: Dict,
        compliance: Dict,
        start_time: float
    ) -> ApplicationResponse:
        """Construct final response."""
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return ApplicationResponse(
            application_id=request.applicant_id,
            decision="approve" if prediction['score'] > 0.5 else "decline",
            decision_score=prediction['score'],
            confidence=prediction['confidence'],
            risk_factors=[],
            recommended_terms={},
            processing_time_ms=processing_time_ms,
            model_version="1.0.0",
            requires_manual_review=prediction['confidence'] < 0.7,
            review_reasons=[] if prediction['confidence'] >= 0.7 else ["Low confidence"],
            compliance_flags=compliance.get('flags', [])
        )
    
    async def _cache_result(self, applicant_id: str, response: ApplicationResponse):
        """Cache result in Redis."""
        cache_key = f"underwriting:{applicant_id}"
        await self.redis_client.setex(
            cache_key,
            3600,
            response.json()
        )
    
    async def _emit_events(self, request: ApplicationRequest, response: ApplicationResponse):
        """Emit events to Kafka for downstream processing."""
        
        pass

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    config = UnderwritingPlatformConfig(
        environment="production",
        database_url="postgresql+asyncpg://user:pass@localhost/underwriting",
        redis_url="redis://localhost",
        kafka_bootstrap_servers=["localhost:9092"],
        ml_model_registry_url="http://mlflow:5000",
        feature_store_url="http://feast:6565",
        jwt_secret_key="your-secret-key"
    )
    
    app.state.engine = UnderwritingEngine(config)
    await app.state.engine.initialize()
    
    logger.info("application_started", environment=config.environment)

@app.post("/api/v1/underwrite", response_model=ApplicationResponse)
async def underwrite_application(
    request: ApplicationRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(oauth2_scheme)
) -> ApplicationResponse:
    """
    Main underwriting endpoint.
    
    Process an underwriting application and return decision.
    """
    
    logger.info(
        "underwriting_request_received",
        applicant_id=request.applicant_id,
        application_type=request.application_type
    )
    
    response = await app.state.engine.process_application(request, None)
    
    background_tasks.add_task(log_audit_trail, request, response)
    
    return response

async def log_audit_trail(request: ApplicationRequest, response: ApplicationResponse):
    """Log audit trail for compliance."""
    
    logger.info(
        "audit_trail",
        application_id=response.application_id,
        decision=response.decision,
        score=response.decision_score
    )

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancer."""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest
    return generate_latest()
```

## Microservices Architecture

### Service Decomposition

```python
"""
Microservices architecture for scalable underwriting platform.

Services:
1. Application Ingestion Service (handles incoming applications)
2. Data Enrichment Service (fetches alternative data)
3. Feature Engineering Service (computes features)
4. ML Inference Service (model predictions)
5. Decision Engine Service (business rules + ML)
6. Compliance Service (regulatory checks)
7. Notification Service (alerts, emails)
8. Audit Service (compliance logging)
9. Analytics Service (dashboards, reporting)
10. Model Training Service (continuous learning)
"""

from dataclasses import dataclass
from enum import Enum
import grpc
from concurrent import futures

class ServiceDiscovery:
    """
    Service discovery and registration using Consul/Eureka.
    """
    
    def __init__(self, registry_url: str):
        self.registry_url = registry_url
        self.services = {}
        
    def register_service(
        self,
        service_name: str,
        host: str,
        port: int,
        health_check_url: str
    ):
        """Register a service with the registry."""
        
        self.services[service_name] = {
            'host': host,
            'port': port,
            'health_check_url': health_check_url
        }
        
        logger.info(
            "service_registered",
            service_name=service_name,
            host=host,
            port=port
        )
    
    def discover_service(self, service_name: str) -> Dict[str, Any]:
        """Discover a service by name."""
        
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not found")
        
        return self.services[service_name]
    
    async def health_check(self, service_name: str) -> bool:
        """Check health of a service."""
        
        return True

class DataEnrichmentService:
    """
    Microservice for enriching applications with alternative data.
    
    Data sources:
    - Credit bureaus
    - Open banking APIs
    - Telco payment data
    - Social media (with consent)
    - Public records
    """
    
    def __init__(self):
        self.data_providers = {}
        
    async def enrich_application(
        self,
        applicant_id: str,
        applicant_data: Dict
    ) -> Dict[str, Any]:
        """
        Enrich application with alternative data sources.
        
        Returns enriched data dictionary.
        """
        
        enriched_data = {}
        
        credit_bureau_task = self._fetch_credit_bureau_data(applicant_id)
        bank_data_task = self._fetch_bank_account_data(applicant_id)
        telco_data_task = self._fetch_telco_data(applicant_id)
        
        results = await asyncio.gather(
            credit_bureau_task,
            bank_data_task,
            telco_data_task,
            return_exceptions=True
        )
        
        for result in results:
            if isinstance(result, Exception):
                logger.warning("data_enrichment_failed", error=str(result))
                continue
            enriched_data.update(result)
        
        return enriched_data
    
    async def _fetch_credit_bureau_data(self, applicant_id: str) -> Dict:
        """Fetch data from credit bureau."""
        
        return {
            'credit_score': 720,
            'accounts': 5,
            'delinquencies': 0
        }
    
    async def _fetch_bank_account_data(self, applicant_id: str) -> Dict:
        """Fetch bank account data via Open Banking API."""
        
        return {
            'avg_balance': 5000,
            'monthly_income': 6000,
            'nsf_count': 0
        }
    
    async def _fetch_telco_data(self, applicant_id: str) -> Dict:
        """Fetch telecommunications payment data."""
        
        return {
            'telco_on_time_rate': 0.95,
            'telco_tenure_months': 36
        }

class FeatureEngineeringService:
    """
    Microservice for computing ML features from raw data.
    
    Uses feature store (Feast) for consistency.
    """
    
    def __init__(self, feature_store_url: str):
        self.feature_store_url = feature_store_url
        
    async def compute_features(
        self,
        applicant_data: Dict,
        enriched_data: Dict
    ) -> Dict[str, float]:
        """
        Compute ML features from raw data.
        
        Features organized by feature groups:
        - Demographic features
        - Financial features
        - Behavioral features
        - Alternative data features
        """
        
        features = {}
        
        features.update(self._compute_demographic_features(applicant_data))
        features.update(self._compute_financial_features(applicant_data, enriched_data))
        features.update(self._compute_behavioral_features(enriched_data))
        features.update(self._compute_alternative_data_features(enriched_data))
        
        await self._push_to_feature_store(features)
        
        return features
    
    def _compute_demographic_features(self, data: Dict) -> Dict[str, float]:
        """Compute demographic features."""
        return {
            'age': data.get('age', 0),
            'years_at_current_address': data.get('years_at_address', 0),
            'employment_tenure_months': data.get('employment_tenure', 0)
        }
    
    def _compute_financial_features(self, applicant_data: Dict, enriched_data: Dict) -> Dict[str, float]:
        """Compute financial features."""
        income = applicant_data.get('income', 0)
        debt = applicant_data.get('debt', 0)
        
        return {
            'debt_to_income': debt / (income + 1),
            'credit_score': enriched_data.get('credit_score', 0),
            'avg_balance': enriched_data.get('avg_balance', 0)
        }
    
    def _compute_behavioral_features(self, enriched_data: Dict) -> Dict[str, float]:
        """Compute behavioral features."""
        return {
            'nsf_count': enriched_data.get('nsf_count', 0),
            'telco_on_time_rate': enriched_data.get('telco_on_time_rate', 0)
        }
    
    def _compute_alternative_data_features(self, enriched_data: Dict) -> Dict[str, float]:
        """Compute features from alternative data."""
        return {}
    
    async def _push_to_feature_store(self, features: Dict):
        """Push features to feature store for consistency."""
        pass

class MLInferenceService:
    """
    High-performance ML inference service.
    
    Uses:
    - TensorFlow Serving / TorchServe
    - Model versioning
    - A/B testing
    - Shadow mode deployment
    """
    
    def __init__(self, model_registry_url: str):
        self.model_registry_url = model_registry_url
        self.loaded_models = {}
        
    async def predict(
        self,
        features: Dict[str, float],
        model_name: str,
        model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate ML prediction.
        
        Supports:
        - Multi-model inference
        - Ensemble predictions
        - Confidence scoring
        """
        
        model = await self._load_model(model_name, model_version)
        
        feature_vector = self._prepare_features(features)
        
        prediction = await self._run_inference(model, feature_vector)
        
        return {
            'score': prediction['score'],
            'probability': prediction['probability'],
            'confidence': prediction['confidence'],
            'model_version': model['version']
        }
    
    async def _load_model(self, model_name: str, version: Optional[str]):
        """Load model from registry."""
        
        cache_key = f"{model_name}:{version or 'latest'}"
        
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        model = {}
        
        self.loaded_models[cache_key] = model
        
        return model
    
    def _prepare_features(self, features: Dict) -> Any:
        """Prepare features for inference."""
        return features
    
    async def _run_inference(self, model, feature_vector) -> Dict:
        """Run inference."""
        
        return {
            'score': 0.75,
            'probability': 0.85,
            'confidence': 0.90
        }

class DecisionEngineService:
    """
    Combine ML predictions with business rules.
    
    Implements:
    - Rule engine (Drools-style)
    - ML + rules hybrid decisions
    - Override mechanisms
    - Challenger models
    """
    
    def __init__(self):
        self.rules = []
        
    async def make_decision(
        self,
        ml_prediction: Dict,
        applicant_data: Dict,
        features: Dict
    ) -> Dict[str, Any]:
        """
        Make final underwriting decision.
        
        Process:
        1. Check hard rules (knock-outs)
        2. Apply ML prediction
        3. Apply soft rules (adjustments)
        4. Determine final decision
        """
        
        hard_rules_result = await self._check_hard_rules(applicant_data, features)
        
        if not hard_rules_result['passed']:
            return {
                'decision': 'decline',
                'reason': 'policy_violation',
                'violated_rules': hard_rules_result['violations']
            }
        
        ml_decision = 'approve' if ml_prediction['score'] > 0.5 else 'decline'
        
        soft_rules_adjustments = await self._apply_soft_rules(
            ml_decision,
            ml_prediction,
            features
        )
        
        final_decision = self._determine_final_decision(
            ml_decision,
            ml_prediction,
            soft_rules_adjustments
        )
        
        return final_decision
    
    async def _check_hard_rules(self, applicant_data: Dict, features: Dict) -> Dict:
        """Check mandatory business rules."""
        
        violations = []
        
        if applicant_data.get('age', 0) < 18:
            violations.append("Applicant under minimum age")
        
        if features.get('credit_score', 0) < 500:
            violations.append("Credit score below minimum threshold")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations
        }
    
    async def _apply_soft_rules(
        self,
        ml_decision: str,
        ml_prediction: Dict,
        features: Dict
    ) -> Dict:
        """Apply soft rules that adjust decision."""
        
        adjustments = []
        
        return {'adjustments': adjustments}
    
    def _determine_final_decision(
        self,
        ml_decision: str,
        ml_prediction: Dict,
        adjustments: Dict
    ) -> Dict:
        """Determine final decision."""
        
        return {
            'decision': ml_decision,
            'decision_score': ml_prediction['score'],
            'confidence': ml_prediction['confidence'],
            'requires_manual_review': ml_prediction['confidence'] < 0.7
        }

class ComplianceService:
    """
    Regulatory compliance checking service.
    
    Checks:
    - Fair lending (ECOA, FCRA)
    - Adverse action notices
    - Model explainability requirements
    - Data privacy (GDPR, CCPA)
    - Industry-specific regulations
    """
    
    def __init__(self):
        self.compliance_rules = []
        
    async def check_compliance(
        self,
        application: ApplicationRequest,
        decision: Dict,
        features: Dict
    ) -> Dict[str, Any]:
        """
        Comprehensive compliance check.
        """
        
        fair_lending_check = await self._check_fair_lending(features, decision)
        
        privacy_check = await self._check_data_privacy(application)
        
        explainability_check = await self._check_explainability(decision)
        
        all_compliant = (
            fair_lending_check['compliant'] and
            privacy_check['compliant'] and
            explainability_check['compliant']
        )
        
        return {
            'compliant': all_compliant,
            'fair_lending': fair_lending_check,
            'privacy': privacy_check,
            'explainability': explainability_check,
            'flags': self._collect_flags([
                fair_lending_check,
                privacy_check,
                explainability_check
            ])
        }
    
    async def _check_fair_lending(self, features: Dict, decision: Dict) -> Dict:
        """Check fair lending compliance."""
        
        return {
            'compliant': True,
            'disparate_impact_ratio': 0.85
        }
    
    async def _check_data_privacy(self, application: ApplicationRequest) -> Dict:
        """Check data privacy compliance."""
        
        return {
            'compliant': True,
            'consent_obtained': True
        }
    
    async def _check_explainability(self, decision: Dict) -> Dict:
        """Check explainability requirements."""
        
        return {
            'compliant': True,
            'explanation_provided': True
        }
    
    def _collect_flags(self, checks: List[Dict]) -> List[str]:
        """Collect compliance flags."""
        
        flags = []
        
        for check in checks:
            if not check['compliant']:
                flags.append(f"Non-compliant: {check}")
        
        return flags
```

## Real-Time Data Pipelines

### Streaming Architecture

```python
"""
Real-time data pipeline using Apache Kafka.

Pipeline stages:
1. Application ingestion → Kafka topic
2. Data enrichment → Kafka topic
3. Feature computation → Kafka topic
4. ML inference → Kafka topic
5. Decision storage → Database
"""

from confluent_kafka import Producer, Consumer
from confluent_kafka.admin import AdminClient, NewTopic
import json
from datetime import datetime

class KafkaDataPipeline:
    """
    Real-time data pipeline for underwriting.
    """
    
    def __init__(self, bootstrap_servers: List[str]):
        self.bootstrap_servers = bootstrap_servers
        self.producer = self._create_producer()
        self.topics = self._define_topics()
        
    def _create_producer(self) -> Producer:
        """Create Kafka producer."""
        
        config = {
            'bootstrap.servers': ','.join(self.bootstrap_servers),
            'client.id': 'underwriting-producer'
        }
        
        return Producer(config)
    
    def _define_topics(self) -> Dict[str, str]:
        """Define Kafka topics."""
        
        return {
            'applications_received': 'underwriting.applications.received',
            'data_enriched': 'underwriting.data.enriched',
            'features_computed': 'underwriting.features.computed',
            'predictions_generated': 'underwriting.predictions.generated',
            'decisions_made': 'underwriting.decisions.made'
        }
    
    async def publish_application_received(
        self,
        application: ApplicationRequest
    ):
        """Publish application received event."""
        
        event = {
            'event_type': 'application_received',
            'timestamp': datetime.utcnow().isoformat(),
            'application_id': application.applicant_id,
            'application_type': application.application_type,
            'data': application.dict()
        }
        
        self.producer.produce(
            self.topics['applications_received'],
            key=application.applicant_id.encode('utf-8'),
            value=json.dumps(event).encode('utf-8')
        )
        
        self.producer.flush()
        
        logger.info(
            "kafka_event_published",
            topic=self.topics['applications_received'],
            application_id=application.applicant_id
        )
    
    async def publish_decision_made(
        self,
        application_id: str,
        decision: ApplicationResponse
    ):
        """Publish decision made event."""
        
        event = {
            'event_type': 'decision_made',
            'timestamp': datetime.utcnow().isoformat(),
            'application_id': application_id,
            'decision': decision.decision,
            'score': decision.decision_score,
            'data': decision.dict()
        }
        
        self.producer.produce(
            self.topics['decisions_made'],
            key=application_id.encode('utf-8'),
            value=json.dumps(event).encode('utf-8')
        )
        
        self.producer.flush()

class StreamProcessor:
    """
    Stream processor for real-time feature computation.
    
    Uses Apache Flink / Kafka Streams for stateful processing.
    """
    
    def __init__(self, bootstrap_servers: List[str]):
        self.bootstrap_servers = bootstrap_servers
        self.consumer = self._create_consumer()
        
    def _create_consumer(self) -> Consumer:
        """Create Kafka consumer."""
        
        config = {
            'bootstrap.servers': ','.join(self.bootstrap_servers),
            'group.id': 'underwriting-stream-processor',
            'auto.offset.reset': 'earliest'
        }
        
        return Consumer(config)
    
    async def process_stream(self):
        """Process events from Kafka stream."""
        
        self.consumer.subscribe(['underwriting.data.enriched'])
        
        while True:
            msg = self.consumer.poll(timeout=1.0)
            
            if msg is None:
                continue
            
            if msg.error():
                logger.error("kafka_consumer_error", error=msg.error())
                continue
            
            event = json.loads(msg.value().decode('utf-8'))
            
            await self._process_event(event)
    
    async def _process_event(self, event: Dict):
        """Process individual event."""
        
        logger.info("processing_event", event_type=event.get('event_type'))
```

## ML Model Deployment

### Production Model Serving

```python
"""
Production ML model serving infrastructure.

Components:
- Model registry (MLflow)
- Model versioning
- A/B testing framework
- Shadow mode deployment
- Model monitoring
"""

import mlflow
import mlflow.pyfunc
from typing import Optional

class ModelRegistry:
    """
    Centralized model registry using MLflow.
    """
    
    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
        
    def register_model(
        self,
        model_name: str,
        model_path: str,
        model_version: str,
        metrics: Dict[str, float],
        tags: Dict[str, str]
    ):
        """Register new model version."""
        
        with mlflow.start_run():
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            for tag_name, tag_value in tags.items():
                mlflow.set_tag(tag_name, tag_value)
            
            mlflow.sklearn.log_model(
                sk_model=None,
                artifact_path="model",
                registered_model_name=model_name
            )
        
        logger.info(
            "model_registered",
            model_name=model_name,
            version=model_version
        )
    
    def promote_model_to_production(
        self,
        model_name: str,
        version: str
    ):
        """Promote model version to production."""
        
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        
        logger.info(
            "model_promoted_to_production",
            model_name=model_name,
            version=version
        )
    
    def load_production_model(self, model_name: str):
        """Load production model."""
        
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        
        return model

class ABTestingFramework:
    """
    A/B testing framework for model deployment.
    
    Allows gradual rollout of new models.
    """
    
    def __init__(self):
        self.experiments = {}
        
    def create_experiment(
        self,
        experiment_name: str,
        control_model_version: str,
        treatment_model_version: str,
        traffic_split: float = 0.1
    ):
        """
        Create A/B test experiment.
        
        Args:
            traffic_split: Fraction of traffic to route to treatment (0-1)
        """
        
        self.experiments[experiment_name] = {
            'control': control_model_version,
            'treatment': treatment_model_version,
            'traffic_split': traffic_split,
            'metrics': {'control': [], 'treatment': []}
        }
        
        logger.info(
            "ab_test_created",
            experiment=experiment_name,
            traffic_split=traffic_split
        )
    
    def route_request(
        self,
        experiment_name: str,
        request_id: str
    ) -> str:
        """
        Route request to control or treatment.
        
        Uses consistent hashing for stable assignment.
        """
        
        experiment = self.experiments[experiment_name]
        
        hash_value = hash(request_id) % 100
        
        if hash_value < experiment['traffic_split'] * 100:
            return experiment['treatment']
        else:
            return experiment['control']
    
    def record_metric(
        self,
        experiment_name: str,
        variant: str,
        metric_name: str,
        metric_value: float
    ):
        """Record metric for experiment analysis."""
        
        self.experiments[experiment_name]['metrics'][variant].append({
            'name': metric_name,
            'value': metric_value,
            'timestamp': datetime.utcnow()
        })
    
    def analyze_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """Analyze A/B test results."""
        
        experiment = self.experiments[experiment_name]
        
        return {
            'experiment': experiment_name,
            'control_performance': self._calculate_metrics(experiment['metrics']['control']),
            'treatment_performance': self._calculate_metrics(experiment['metrics']['treatment']),
            'statistical_significance': self._test_significance(experiment)
        }
    
    def _calculate_metrics(self, metrics: List[Dict]) -> Dict:
        """Calculate aggregate metrics."""
        return {}
    
    def _test_significance(self, experiment: Dict) -> Dict:
        """Test statistical significance of results."""
        return {}
```

This module provides the foundation for building a production-ready AI underwriting platform with modern architecture. The next modules will cover additional startup building blocks.
