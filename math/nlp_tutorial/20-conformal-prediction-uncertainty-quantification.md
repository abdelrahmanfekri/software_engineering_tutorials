# NLP Tutorial Module 20: Conformal Prediction and Uncertainty Quantification in NLP

## Learning Objectives
By the end of this module, you will be able to:
- Understand conformal prediction theory and its guarantees
- Implement conformal prediction for NLP tasks
- Quantify uncertainty in language model predictions
- Build calibrated confidence sets for text classification
- Apply conformal prediction to sequence labeling and generation
- Use uncertainty quantification for reliable AI systems

## Introduction to Conformal Prediction

Conformal prediction is a framework for uncertainty quantification that provides statistically valid confidence sets for predictions with finite-sample coverage guarantees, without making distributional assumptions.

### Why Uncertainty Quantification in NLP?

1. **Reliability**: Know when models are confident vs uncertain
2. **Safety**: Avoid acting on unreliable predictions
3. **Human-AI Collaboration**: Help humans focus on uncertain cases
4. **Model Debugging**: Identify systematic failures
5. **Active Learning**: Select most informative examples

### Key Concepts

- **Coverage Guarantee**: P(Y ∈ C(X)) ≥ 1 - α
- **Conformal Score**: Measure of nonconformity
- **Calibration Set**: Held-out data for calibration
- **Prediction Set**: Set of plausible labels

## Mathematical Foundation

### Conformal Prediction Framework

```python
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Callable, Dict, Any
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict

class ConformalPredictor:
    """
    Base class for conformal prediction
    
    Provides finite-sample coverage guarantee:
    P(y_test ∈ C(x_test)) ≥ 1 - α
    """
    
    def __init__(self, model, alpha: float = 0.1):
        """
        Initialize conformal predictor
        
        Args:
            model: Pre-trained model
            alpha: Significance level (1-alpha is coverage level)
        """
        self.model = model
        self.alpha = alpha
        self.calibration_scores = None
        self.q_hat = None  # Conformal quantile
    
    def calibrate(self, cal_x: torch.Tensor, cal_y: torch.Tensor):
        """
        Calibrate conformal predictor on calibration set
        
        Args:
            cal_x: Calibration features
            cal_y: Calibration labels
        """
        # Compute nonconformity scores on calibration set
        scores = self.compute_nonconformity_scores(cal_x, cal_y)
        self.calibration_scores = scores
        
        # Compute quantile
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(scores, q_level)
        
        print(f"Calibrated with {n} samples")
        print(f"Quantile at level {q_level:.4f}: {self.q_hat:.4f}")
    
    def compute_nonconformity_scores(self, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """
        Compute nonconformity scores
        Override in subclasses for specific tasks
        """
        raise NotImplementedError
    
    def predict_with_confidence(self, x: torch.Tensor) -> Tuple[Any, Any]:
        """
        Make prediction with confidence set
        """
        raise NotImplementedError

class ClassificationConformalPredictor(ConformalPredictor):
    """Conformal prediction for classification tasks"""
    
    def compute_nonconformity_scores(self, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """
        Compute nonconformity scores using softmax probabilities
        Score: 1 - P(y_true | x)
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
        
        # Get probability of true class
        batch_size = x.size(0)
        true_class_probs = probs[torch.arange(batch_size), y].cpu().numpy()
        
        # Nonconformity score: 1 - p(y_true)
        scores = 1 - true_class_probs
        
        return scores
    
    def predict_with_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        Predict with conformal prediction set
        
        Returns:
            Point prediction and prediction set
        """
        if self.q_hat is None:
            raise ValueError("Must calibrate before prediction")
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        # Point prediction
        point_pred = torch.argmax(logits, dim=-1)
        
        # Prediction set: all classes with score ≤ q_hat
        # Score for class c: 1 - p(c)
        prediction_sets = []
        for prob_dist in probs:
            pred_set = np.where(1 - prob_dist <= self.q_hat)[0].tolist()
            prediction_sets.append(pred_set)
        
        return point_pred, prediction_sets
    
    def evaluate_coverage(self, test_x: torch.Tensor, test_y: torch.Tensor) -> Dict:
        """Evaluate empirical coverage on test set"""
        _, prediction_sets = self.predict_with_confidence(test_x)
        test_y_np = test_y.cpu().numpy()
        
        # Check coverage
        covered = [test_y_np[i] in pred_set for i, pred_set in enumerate(prediction_sets)]
        coverage = np.mean(covered)
        
        # Average set size
        avg_set_size = np.mean([len(pred_set) for pred_set in prediction_sets])
        
        # Singleton sets (certain predictions)
        singleton_rate = np.mean([len(pred_set) == 1 for pred_set in prediction_sets])
        
        return {
            'coverage': coverage,
            'target_coverage': 1 - self.alpha,
            'avg_set_size': avg_set_size,
            'singleton_rate': singleton_rate,
            'num_samples': len(test_y)
        }

# Example: Text Classification with Conformal Prediction

class SimpleTextClassifier(nn.Module):
    """Simple text classifier for demonstration"""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        pooled = embedded.mean(dim=1)  # (batch_size, embed_dim)
        logits = self.fc(pooled)  # (batch_size, num_classes)
        return logits

# Create synthetic data
def create_synthetic_text_data(num_samples: int = 1000, vocab_size: int = 100, 
                               seq_len: int = 20, num_classes: int = 5):
    """Create synthetic text classification data"""
    X = torch.randint(0, vocab_size, (num_samples, seq_len))
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y

# Generate data
X, y = create_synthetic_text_data(num_samples=1000)

# Split into train, calibration, and test
train_size = 600
cal_size = 200
test_size = 200

X_train, y_train = X[:train_size], y[:train_size]
X_cal, y_cal = X[train_size:train_size+cal_size], y[train_size:train_size+cal_size]
X_test, y_test = X[train_size+cal_size:], y[train_size+cal_size:]

print(f"Train: {len(X_train)}, Calibration: {len(X_cal)}, Test: {len(X_test)}")

# Train model (simplified training)
model = SimpleTextClassifier(vocab_size=100, embed_dim=50, num_classes=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Quick training
model.train()
for epoch in range(50):
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Apply conformal prediction
cp_predictor = ClassificationConformalPredictor(model, alpha=0.1)
cp_predictor.calibrate(X_cal, y_cal)

# Evaluate on test set
metrics = cp_predictor.evaluate_coverage(X_test, y_test)
print("\nConformal Prediction Metrics:")
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# Make predictions with confidence sets
sample_x = X_test[:5]
sample_y = y_test[:5]
point_preds, pred_sets = cp_predictor.predict_with_confidence(sample_x)

print("\nSample Predictions:")
for i in range(len(sample_x)):
    print(f"  True: {sample_y[i].item()}, Predicted: {point_preds[i].item()}, "
          f"Confidence Set: {pred_sets[i]}")
```

## Adaptive Conformal Prediction

### Locally Adaptive Conformal Prediction

```python
class AdaptiveConformalPredictor(ConformalPredictor):
    """
    Adaptive conformal prediction with difficulty-based weighting
    Provides smaller prediction sets for easier examples
    """
    
    def __init__(self, model, alpha: float = 0.1):
        super().__init__(model, alpha)
        self.difficulty_scores = None
    
    def compute_difficulty_score(self, x: torch.Tensor) -> np.ndarray:
        """
        Compute difficulty/uncertainty score for each example
        Higher score = more difficult/uncertain
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
        
        # Use entropy as difficulty measure
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy.cpu().numpy()
    
    def calibrate(self, cal_x: torch.Tensor, cal_y: torch.Tensor):
        """Calibrate with difficulty-based weighting"""
        # Compute nonconformity scores
        scores = self.compute_nonconformity_scores(cal_x, cal_y)
        
        # Compute difficulty scores
        difficulty = self.compute_difficulty_score(cal_x)
        
        # Normalize scores by difficulty
        # Easier examples (low entropy) get lower scores
        normalized_scores = scores / (1 + difficulty)
        
        self.calibration_scores = normalized_scores
        self.difficulty_scores = difficulty
        
        # Compute quantile
        n = len(normalized_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(normalized_scores, q_level)
        
        print(f"Adaptive calibration completed with {n} samples")
    
    def predict_with_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """Predict with adaptive prediction sets"""
        if self.q_hat is None:
            raise ValueError("Must calibrate before prediction")
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        # Compute difficulty for test examples
        test_difficulty = self.compute_difficulty_score(x)
        
        # Point prediction
        point_pred = torch.argmax(logits, dim=-1)
        
        # Adaptive prediction sets
        prediction_sets = []
        for i, prob_dist in enumerate(probs):
            # Adjust threshold based on difficulty
            adjusted_threshold = self.q_hat * (1 + test_difficulty[i])
            pred_set = np.where(1 - prob_dist <= adjusted_threshold)[0].tolist()
            prediction_sets.append(pred_set)
        
        return point_pred, prediction_sets

# Apply adaptive conformal prediction
adaptive_cp = AdaptiveConformalPredictor(model, alpha=0.1)
adaptive_cp.calibrate(X_cal, y_cal)

# Evaluate
adaptive_metrics = adaptive_cp.evaluate_coverage(X_test, y_test)
print("\nAdaptive Conformal Prediction Metrics:")
for key, value in adaptive_metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# Compare standard vs adaptive
print("\nComparison:")
print(f"Standard CP avg set size: {metrics['avg_set_size']:.2f}")
print(f"Adaptive CP avg set size: {adaptive_metrics['avg_set_size']:.2f}")
```

## Conformal Prediction for Sequence Labeling

### Token-Level Conformal Prediction

```python
class SequenceLabelingConformalPredictor:
    """Conformal prediction for sequence labeling (NER, POS tagging)"""
    
    def __init__(self, model, alpha: float = 0.1):
        self.model = model
        self.alpha = alpha
        self.q_hats = None  # Quantile per position
    
    def calibrate(self, cal_x: torch.Tensor, cal_y: torch.Tensor, 
                  cal_lengths: torch.Tensor):
        """
        Calibrate for sequence labeling
        
        Args:
            cal_x: (batch_size, max_seq_len) token IDs
            cal_y: (batch_size, max_seq_len) true labels
            cal_lengths: (batch_size,) actual sequence lengths
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(cal_x)  # (batch_size, max_seq_len, num_classes)
            probs = torch.softmax(logits, dim=-1)
        
        # Collect scores for each position
        position_scores = defaultdict(list)
        
        for i in range(len(cal_x)):
            seq_len = cal_lengths[i].item()
            for pos in range(seq_len):
                true_label = cal_y[i, pos].item()
                true_prob = probs[i, pos, true_label].item()
                score = 1 - true_prob
                position_scores[pos].append(score)
        
        # Compute quantile for each position
        self.q_hats = {}
        for pos, scores in position_scores.items():
            n = len(scores)
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            self.q_hats[pos] = np.quantile(scores, q_level)
        
        print(f"Calibrated {len(self.q_hats)} positions")
    
    def predict_with_confidence(self, x: torch.Tensor, lengths: torch.Tensor):
        """Predict sequence labels with confidence sets"""
        if self.q_hats is None:
            raise ValueError("Must calibrate before prediction")
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        # Get point predictions
        point_preds = torch.argmax(logits, dim=-1)
        
        # Get prediction sets for each token
        prediction_sets = []
        for i in range(len(x)):
            seq_len = lengths[i].item()
            seq_pred_sets = []
            
            for pos in range(seq_len):
                # Use quantile for this position (or max if not calibrated)
                q_hat = self.q_hats.get(pos, max(self.q_hats.values()))
                prob_dist = probs[i, pos]
                pred_set = np.where(1 - prob_dist <= q_hat)[0].tolist()
                seq_pred_sets.append(pred_set)
            
            prediction_sets.append(seq_pred_sets)
        
        return point_preds, prediction_sets
    
    def evaluate_coverage(self, test_x: torch.Tensor, test_y: torch.Tensor,
                         test_lengths: torch.Tensor) -> Dict:
        """Evaluate token-level coverage"""
        _, prediction_sets = self.predict_with_confidence(test_x, test_lengths)
        
        total_tokens = 0
        covered_tokens = 0
        total_set_size = 0
        
        for i in range(len(test_x)):
            seq_len = test_lengths[i].item()
            for pos in range(seq_len):
                true_label = test_y[i, pos].item()
                pred_set = prediction_sets[i][pos]
                
                total_tokens += 1
                if true_label in pred_set:
                    covered_tokens += 1
                total_set_size += len(pred_set)
        
        return {
            'token_coverage': covered_tokens / total_tokens,
            'target_coverage': 1 - self.alpha,
            'avg_set_size': total_set_size / total_tokens,
            'num_tokens': total_tokens
        }
```

## Conformal Prediction for Text Generation

### Generation with Uncertainty Quantification

```python
class GenerationConformalPredictor:
    """Conformal prediction for text generation"""
    
    def __init__(self, model, tokenizer, alpha: float = 0.1):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.token_q_hats = {}  # Quantiles per token position
    
    def calibrate_generation(self, cal_prompts: List[str], 
                            cal_continuations: List[str]):
        """
        Calibrate generation model
        
        Args:
            cal_prompts: List of prompt texts
            cal_continuations: List of corresponding true continuations
        """
        position_scores = defaultdict(list)
        
        for prompt, continuation in zip(cal_prompts, cal_continuations):
            # Tokenize
            prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            continuation_ids = self.tokenizer.encode(continuation, return_tensors='pt')
            
            # Get model probabilities for continuation
            with torch.no_grad():
                outputs = self.model(
                    input_ids=torch.cat([prompt_ids, continuation_ids], dim=1)
                )
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            # Compute scores for each token in continuation
            prompt_len = prompt_ids.size(1)
            for pos, token_id in enumerate(continuation_ids[0]):
                token_prob = probs[0, prompt_len + pos - 1, token_id].item()
                score = -np.log(token_prob + 1e-10)  # Negative log-likelihood
                position_scores[pos].append(score)
        
        # Compute quantiles
        for pos, scores in position_scores.items():
            n = len(scores)
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            self.token_q_hats[pos] = np.quantile(scores, q_level)
        
        print(f"Calibrated generation for {len(self.token_q_hats)} positions")
    
    def generate_with_confidence(self, prompt: str, max_length: int = 50) -> Dict:
        """
        Generate text with token-level confidence
        
        Returns:
            Dictionary with generated text and confidence scores
        """
        if not self.token_q_hats:
            raise ValueError("Must calibrate before generation")
        
        prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        generated_ids = prompt_ids.clone()
        
        token_confidences = []
        uncertain_positions = []
        
        for pos in range(max_length):
            # Get next token distribution
            with torch.no_grad():
                outputs = self.model(input_ids=generated_ids)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Compute confidence
            next_token_prob = probs[0, next_token].item()
            score = -np.log(next_token_prob + 1e-10)
            
            # Check against calibrated quantile
            q_hat = self.token_q_hats.get(pos, max(self.token_q_hats.values()))
            is_confident = score <= q_hat
            
            token_confidences.append({
                'token': self.tokenizer.decode([next_token[0].item()]),
                'probability': next_token_prob,
                'score': score,
                'threshold': q_hat,
                'confident': is_confident
            })
            
            if not is_confident:
                uncertain_positions.append(pos)
            
            # Stop if EOS token
            if next_token[0].item() == self.tokenizer.eos_token_id:
                break
        
        generated_text = self.tokenizer.decode(
            generated_ids[0][prompt_ids.size(1):],
            skip_special_tokens=True
        )
        
        return {
            'prompt': prompt,
            'generated_text': generated_text,
            'token_confidences': token_confidences,
            'uncertain_positions': uncertain_positions,
            'overall_confidence': np.mean([t['confident'] for t in token_confidences])
        }
```

## Calibration Diagnostics and Visualization

### Calibration Analysis Tools

```python
class CalibrationDiagnostics:
    """Tools for analyzing calibration quality"""
    
    @staticmethod
    def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, 
                                 n_bins: int = 10):
        """
        Plot reliability diagram (calibration curve)
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins
        """
        # Bin predictions
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
            if mask.sum() > 0:
                bin_acc = y_true[mask].mean()
                bin_conf = y_prob[mask].mean()
                bin_accuracies.append(bin_acc)
                bin_confidences.append(bin_conf)
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(bin_centers[i])
                bin_counts.append(0)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Reliability diagram
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.plot(bin_confidences, bin_accuracies, 'o-', label='Model')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Reliability Diagram')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histogram of predictions
        ax2.bar(bin_centers, bin_counts, width=bins[1]-bins[0], alpha=0.7)
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Predictions')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Compute ECE (Expected Calibration Error)
        ece = 0
        total_samples = sum(bin_counts)
        for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
            ece += (count / total_samples) * abs(acc - conf)
        
        print(f"Expected Calibration Error (ECE): {ece:.4f}")
        return ece
    
    @staticmethod
    def plot_coverage_by_alpha(predictor, test_x: torch.Tensor, 
                              test_y: torch.Tensor, alphas: List[float]):
        """
        Plot empirical coverage vs target coverage for different alpha values
        """
        coverages = []
        avg_set_sizes = []
        
        for alpha in alphas:
            predictor.alpha = alpha
            predictor.calibrate(X_cal, y_cal)  # Re-calibrate
            metrics = predictor.evaluate_coverage(test_x, test_y)
            coverages.append(metrics['coverage'])
            avg_set_sizes.append(metrics['avg_set_size'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Coverage plot
        target_coverages = [1 - alpha for alpha in alphas]
        ax1.plot(target_coverages, target_coverages, 'k--', label='Target')
        ax1.plot(target_coverages, coverages, 'o-', label='Empirical')
        ax1.set_xlabel('Target Coverage (1 - α)')
        ax1.set_ylabel('Empirical Coverage')
        ax1.set_title('Coverage Calibration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Set size vs coverage
        ax2.plot(coverages, avg_set_sizes, 'o-')
        ax2.set_xlabel('Empirical Coverage')
        ax2.set_ylabel('Average Prediction Set Size')
        ax2.set_title('Efficiency vs Coverage Trade-off')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def analyze_prediction_sets(prediction_sets: List[List[int]], 
                               true_labels: np.ndarray):
        """Analyze properties of prediction sets"""
        set_sizes = [len(ps) for ps in prediction_sets]
        
        # Coverage
        coverage = np.mean([true_labels[i] in prediction_sets[i] 
                          for i in range(len(prediction_sets))])
        
        # Statistics
        stats = {
            'coverage': coverage,
            'avg_set_size': np.mean(set_sizes),
            'median_set_size': np.median(set_sizes),
            'min_set_size': np.min(set_sizes),
            'max_set_size': np.max(set_sizes),
            'singleton_rate': np.mean([s == 1 for s in set_sizes]),
            'empty_rate': np.mean([s == 0 for s in set_sizes])
        }
        
        # Plot distribution
        plt.figure(figsize=(10, 6))
        plt.hist(set_sizes, bins=range(max(set_sizes) + 2), alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Set Size')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Set Sizes')
        plt.axvline(stats['avg_set_size'], color='r', linestyle='--', 
                   label=f"Mean: {stats['avg_set_size']:.2f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return stats

# Example: Analyze calibration
diagnostics = CalibrationDiagnostics()

# Get predictions
model.eval()
with torch.no_grad():
    test_logits = model(X_test)
    test_probs = torch.softmax(test_logits, dim=-1)
    test_pred = torch.argmax(test_logits, dim=-1)

# For binary case, convert to binary
y_true_binary = (y_test == 0).cpu().numpy()
y_prob_binary = test_probs[:, 0].cpu().numpy()

# Plot reliability diagram
ece = diagnostics.plot_reliability_diagram(y_true_binary, y_prob_binary, n_bins=10)

# Analyze prediction sets
_, pred_sets = cp_predictor.predict_with_confidence(X_test)
stats = diagnostics.analyze_prediction_sets(pred_sets, y_test.cpu().numpy())
print("\nPrediction Set Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value:.4f}")
```

## Advanced Applications

### Risk-Controlled Prediction Sets

```python
class RiskControlledPredictor:
    """
    Conformal prediction with risk control
    Controls false negative rate or other risk metrics
    """
    
    def __init__(self, model, risk_metric: str = 'fnr', target_risk: float = 0.05):
        """
        Initialize risk-controlled predictor
        
        Args:
            model: Base model
            risk_metric: Type of risk to control ('fnr', 'fpr', 'custom')
            target_risk: Target risk level
        """
        self.model = model
        self.risk_metric = risk_metric
        self.target_risk = target_risk
        self.threshold = None
    
    def calibrate(self, cal_x: torch.Tensor, cal_y: torch.Tensor):
        """Calibrate to achieve target risk"""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(cal_x)
            probs = torch.softmax(logits, dim=-1)
        
        # Compute scores
        scores = 1 - probs[torch.arange(len(cal_y)), cal_y].cpu().numpy()
        
        # Find threshold that achieves target risk
        # Binary search over possible thresholds
        thresholds = np.sort(scores)
        best_threshold = thresholds[-1]
        
        for threshold in thresholds:
            # Compute empirical risk at this threshold
            pred_sets = [np.where(1 - probs[i].cpu().numpy() <= threshold)[0] 
                        for i in range(len(cal_y))]
            
            if self.risk_metric == 'fnr':
                # False negative rate: how often true label not in set
                risk = np.mean([cal_y[i].item() not in pred_sets[i] 
                              for i in range(len(cal_y))])
            elif self.risk_metric == 'fpr':
                # False positive rate: average extra labels in set
                risk = np.mean([(len(pred_sets[i]) - 1) / (len(probs[i]) - 1) 
                              for i in range(len(cal_y))])
            
            if risk <= self.target_risk:
                best_threshold = threshold
                break
        
        self.threshold = best_threshold
        print(f"Calibrated threshold: {self.threshold:.4f}")
    
    def predict(self, x: torch.Tensor) -> List[List[int]]:
        """Predict with risk-controlled sets"""
        if self.threshold is None:
            raise ValueError("Must calibrate before prediction")
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        prediction_sets = []
        for prob_dist in probs:
            pred_set = np.where(1 - prob_dist <= self.threshold)[0].tolist()
            prediction_sets.append(pred_set)
        
        return prediction_sets
```

## Practical Exercises

### Exercise 1: Sentiment Analysis with Conformal Prediction
Build a sentiment analysis system that:
- Provides confidence sets for predictions
- Flags uncertain examples for human review
- Achieves 90% coverage guarantee
- Minimizes average set size

### Exercise 2: Named Entity Recognition with Uncertainty
Implement NER with token-level uncertainty:
- Calibrate per entity type
- Visualize uncertain spans
- Compare adaptive vs standard conformal prediction

### Exercise 3: Text Generation with Confidence
Create a text generation system that:
- Provides token-level confidence scores
- Highlights uncertain generations
- Allows confidence-controlled generation

## Assessment Questions

1. **What does the coverage guarantee in conformal prediction mean?**
   - P(Y ∈ C(X)) ≥ 1 - α holds for any distribution
   - Finite-sample guarantee
   - No distributional assumptions needed
   - Requires exchangeability

2. **How does conformal prediction differ from calibration?**
   - CP provides set predictions, not just probabilities
   - CP has finite-sample guarantees
   - CP doesn't require distributional assumptions
   - Calibration adjusts probabilities

3. **When should you use adaptive vs standard conformal prediction?**
   - Adaptive when difficulty varies across examples
   - Standard for uniform coverage
   - Adaptive reduces set sizes for easy examples
   - Trade-off between efficiency and simplicity

## Key Takeaways

- Conformal prediction provides statistically valid uncertainty quantification
- Coverage guarantee holds regardless of model quality
- Prediction sets trade off coverage and efficiency
- Adaptive methods improve efficiency for varying difficulty
- Applicable to classification, regression, and structured prediction
- Critical for deploying reliable ML systems
- Enables human-AI collaboration through uncertainty awareness

## Next Steps

We've completed the advanced NLP tutorial modules! Next, we'll create comprehensive tutorials for applying these techniques to finance, including financial text analysis, trading systems, and agentic AI for financial analysis.

