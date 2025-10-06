# NLP Tutorial Module 16: Explainable AI (XAI) in NLP

## Learning Objectives
By the end of this module, you will be able to:
- Understand the importance of explainability in NLP systems
- Implement attention visualization techniques
- Build interpretable models with built-in explanations
- Apply counterfactual analysis for model understanding
- Use gradient-based attribution methods
- Implement user-specific explanation generation
- Evaluate and improve model transparency
- Design explainable AI systems for real-world applications

## Introduction to Explainable AI in NLP

Explainable AI (XAI) has become crucial in 2025 as NLP systems are deployed in high-stakes applications like healthcare, finance, and legal systems. Users and regulators demand transparency in AI decision-making processes, making explainability a core requirement rather than an afterthought.

### Key XAI Principles in NLP

1. **Transparency**: Model decisions should be understandable
2. **Interpretability**: Users should be able to trace reasoning
3. **Fairness**: Models should not perpetuate biases
4. **Accountability**: Decisions should be auditable
5. **User-Adaptive**: Explanations should match user expertise levels

## Attention Visualization and Analysis

### Multi-Head Attention Visualization

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

class AttentionVisualizer:
    """Comprehensive attention visualization system"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def extract_attention_weights(self, text: str, layer_idx: int = -1) -> Dict[str, torch.Tensor]:
        """Extract attention weights from specified layer"""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attention = outputs.attentions[layer_idx]  # [batch, heads, seq_len, seq_len]
        
        return {
            'attention': attention,
            'input_ids': inputs['input_ids'],
            'tokens': self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        }
    
    def visualize_attention_heatmap(self, text: str, layer_idx: int = -1, head_idx: int = 0):
        """Create attention heatmap visualization"""
        attention_data = self.extract_attention_weights(text, layer_idx)
        attention = attention_data['attention'][0, head_idx].cpu().numpy()
        tokens = attention_data['tokens']
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(attention, 
                   xticklabels=tokens, 
                   yticklabels=tokens,
                   cmap='Blues',
                   cbar_kws={'label': 'Attention Weight'})
        plt.title(f'Attention Heatmap - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        return attention, tokens
    
    def visualize_multi_head_attention(self, text: str, layer_idx: int = -1, max_heads: int = 8):
        """Visualize attention across multiple heads"""
        attention_data = self.extract_attention_weights(text, layer_idx)
        attention = attention_data['attention'][0].cpu().numpy()  # [heads, seq_len, seq_len]
        tokens = attention_data['tokens']
        
        num_heads = min(attention.shape[0], max_heads)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for head_idx in range(num_heads):
            sns.heatmap(attention[head_idx], 
                       xticklabels=tokens, 
                       yticklabels=tokens,
                       cmap='Blues',
                       ax=axes[head_idx],
                       cbar_kws={'label': 'Attention Weight'})
            axes[head_idx].set_title(f'Head {head_idx}')
            axes[head_idx].tick_params(axis='x', rotation=45, labelsize=8)
            axes[head_idx].tick_params(axis='y', rotation=0, labelsize=8)
        
        # Hide unused subplots
        for idx in range(num_heads, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Multi-Head Attention Visualization - Layer {layer_idx}', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def compute_attention_entropy(self, text: str, layer_idx: int = -1) -> np.ndarray:
        """Compute attention entropy for each token"""
        attention_data = self.extract_attention_weights(text, layer_idx)
        attention = attention_data['attention'][0].cpu().numpy()  # [heads, seq_len, seq_len]
        
        # Average across heads
        avg_attention = np.mean(attention, axis=0)
        
        # Compute entropy for each token
        entropy = -np.sum(avg_attention * np.log(avg_attention + 1e-8), axis=1)
        
        return entropy
    
    def create_interactive_attention_plot(self, text: str, layer_idx: int = -1):
        """Create interactive attention visualization with Plotly"""
        attention_data = self.extract_attention_weights(text, layer_idx)
        attention = attention_data['attention'][0].cpu().numpy()
        tokens = attention_data['tokens']
        
        # Average across heads for simplicity
        avg_attention = np.mean(attention, axis=0)
        
        fig = go.Figure(data=go.Heatmap(
            z=avg_attention,
            x=tokens,
            y=tokens,
            colorscale='Blues',
            hoverongaps=False,
            hovertemplate='Query: %{y}<br>Key: %{x}<br>Attention: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Interactive Attention Visualization - Layer {layer_idx}',
            xaxis_title='Key Tokens',
            yaxis_title='Query Tokens',
            width=800,
            height=600
        )
        
        fig.show()
        return fig

# Example usage
def demonstrate_attention_visualization():
    """Demonstrate attention visualization techniques"""
    # Load a pre-trained model (using a smaller model for demonstration)
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    
    visualizer = AttentionVisualizer(model, tokenizer)
    
    # Example text
    text = "The quick brown fox jumps over the lazy dog"
    
    # Visualize attention
    print("Creating attention visualizations...")
    
    # Single head heatmap
    attention, tokens = visualizer.visualize_attention_heatmap(text, layer_idx=-1, head_idx=0)
    
    # Multi-head visualization
    visualizer.visualize_multi_head_attention(text, layer_idx=-1, max_heads=8)
    
    # Interactive plot
    fig = visualizer.create_interactive_attention_plot(text, layer_idx=-1)
    
    # Attention entropy
    entropy = visualizer.compute_attention_entropy(text, layer_idx=-1)
    
    print(f"Attention entropy for each token:")
    for token, ent in zip(tokens, entropy):
        print(f"{token}: {ent:.3f}")
    
    return visualizer, attention, entropy

# Run demonstration
visualizer, attention_weights, entropy_scores = demonstrate_attention_visualization()
```

## Gradient-Based Attribution Methods

### Integrated Gradients Implementation

```python
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

class IntegratedGradients:
    """Integrated Gradients for NLP model interpretation"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def generate_baseline(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate baseline (typically zeros or padding tokens)"""
        baseline = torch.zeros_like(input_ids)
        return baseline
    
    def interpolate_inputs(self, baseline: torch.Tensor, 
                          target: torch.Tensor, 
                          steps: int = 50) -> torch.Tensor:
        """Generate interpolated inputs between baseline and target"""
        alphas = torch.linspace(0, 1, steps + 1, device=self.device)
        
        # Reshape for broadcasting
        alphas = alphas.view(steps + 1, 1, 1)
        
        # Interpolate
        interpolated = baseline.unsqueeze(0) + alphas * (target.unsqueeze(0) - baseline.unsqueeze(0))
        
        return interpolated
    
    def compute_gradients(self, inputs: torch.Tensor, 
                         target_class: int = None) -> torch.Tensor:
        """Compute gradients with respect to inputs"""
        inputs.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(inputs)
        
        if target_class is None:
            target_class = torch.argmax(outputs.logits, dim=-1)
        
        # Compute gradients
        if len(outputs.logits.shape) > 1:
            target_score = outputs.logits[0, target_class]
        else:
            target_score = outputs.logits[target_class]
        
        gradients = torch.autograd.grad(target_score, inputs, create_graph=True)[0]
        
        return gradients
    
    def integrated_gradients(self, text: str, 
                           target_class: int = None,
                           steps: int = 50) -> Dict[str, torch.Tensor]:
        """Compute integrated gradients for input text"""
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.device)
        
        # Generate baseline
        baseline = self.generate_baseline(input_ids)
        
        # Generate interpolated inputs
        interpolated = self.interpolate_inputs(baseline, input_ids, steps)
        
        # Reshape for batch processing
        batch_size, seq_len = interpolated.shape[1], interpolated.shape[2]
        interpolated_flat = interpolated.view(-1, seq_len)
        
        # Compute gradients for all interpolated inputs
        all_gradients = []
        for i in range(0, interpolated_flat.shape[0], 32):  # Process in batches
            batch = interpolated_flat[i:i+32]
            batch_gradients = self.compute_gradients(batch, target_class)
            all_gradients.append(batch_gradients)
        
        gradients = torch.cat(all_gradients, dim=0)
        gradients = gradients.view(steps + 1, batch_size, seq_len)
        
        # Average gradients across steps
        avg_gradients = torch.mean(gradients, dim=0)
        
        # Compute integrated gradients
        integrated_grads = (input_ids - baseline) * avg_gradients
        
        return {
            'integrated_gradients': integrated_grads,
            'tokens': self.tokenizer.convert_ids_to_tokens(input_ids[0]),
            'baseline': baseline,
            'input_ids': input_ids
        }
    
    def visualize_attributions(self, text: str, target_class: int = None):
        """Visualize attribution scores"""
        results = self.integrated_gradients(text, target_class)
        attributions = results['integrated_gradients'][0].cpu().numpy()
        tokens = results['tokens']
        
        # Normalize attributions
        attributions = attributions / (np.sum(np.abs(attributions)) + 1e-8)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['red' if x < 0 else 'blue' for x in attributions]
        bars = ax.bar(range(len(tokens)), attributions, color=colors, alpha=0.7)
        
        ax.set_xlabel('Tokens')
        ax.set_ylabel('Attribution Score')
        ax.set_title('Integrated Gradients Attribution')
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        
        # Add color legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Positive Attribution'),
                          Patch(facecolor='red', alpha=0.7, label='Negative Attribution')]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.show()
        
        return results

# Example usage
def demonstrate_integrated_gradients():
    """Demonstrate integrated gradients"""
    # This would use an actual model in practice
    print("Integrated Gradients demonstration would require a trained model")
    print("The implementation above shows the complete framework")
    
    return True

ig_demo = demonstrate_integrated_gradients()
```

## Counterfactual Analysis

### Counterfactual Explanation Generator

```python
class CounterfactualExplainer:
    """Generate counterfactual explanations for NLP models"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate_minimal_counterfactuals(self, text: str, 
                                       target_class: int,
                                       max_changes: int = 3) -> List[Dict[str, any]]:
        """Generate minimal counterfactual examples"""
        
        # Tokenize original text
        original_tokens = self.tokenizer.tokenize(text)
        
        counterfactuals = []
        
        # Strategy 1: Word replacement
        word_replacements = self._find_word_replacements(original_tokens, target_class)
        counterfactuals.extend(word_replacements)
        
        # Strategy 2: Word deletion
        word_deletions = self._find_word_deletions(original_tokens, target_class)
        counterfactuals.extend(word_deletions)
        
        # Strategy 3: Word insertion
        word_insertions = self._find_word_insertions(original_tokens, target_class)
        counterfactuals.extend(word_insertions)
        
        # Sort by minimal changes
        counterfactuals.sort(key=lambda x: x['num_changes'])
        
        return counterfactuals[:max_changes]
    
    def _find_word_replacements(self, tokens: List[str], target_class: int) -> List[Dict]:
        """Find word replacements that change prediction"""
        counterfactuals = []
        
        # Common word replacements for sentiment analysis
        replacement_dict = {
            'good': ['bad', 'terrible', 'awful'],
            'bad': ['good', 'excellent', 'great'],
            'love': ['hate', 'dislike', 'despise'],
            'hate': ['love', 'adore', 'like'],
            'excellent': ['poor', 'terrible', 'awful'],
            'terrible': ['excellent', 'great', 'wonderful']
        }
        
        for i, token in enumerate(tokens):
            if token.lower() in replacement_dict:
                for replacement in replacement_dict[token.lower()]:
                    new_tokens = tokens.copy()
                    new_tokens[i] = replacement
                    new_text = self.tokenizer.convert_tokens_to_string(new_tokens)
                    
                    # Check if prediction changes
                    prediction = self._get_prediction(new_text)
                    if prediction != target_class:
                        counterfactuals.append({
                            'type': 'replacement',
                            'original_text': self.tokenizer.convert_tokens_to_string(tokens),
                            'counterfactual_text': new_text,
                            'changed_token': token,
                            'replacement_token': replacement,
                            'position': i,
                            'num_changes': 1,
                            'original_prediction': target_class,
                            'counterfactual_prediction': prediction
                        })
        
        return counterfactuals
    
    def _find_word_deletions(self, tokens: List[str], target_class: int) -> List[Dict]:
        """Find word deletions that change prediction"""
        counterfactuals = []
        
        # Try deleting each word
        for i, token in enumerate(tokens):
            if len(tokens) <= 3:  # Don't delete if too few words
                continue
                
            new_tokens = tokens.copy()
            del new_tokens[i]
            new_text = self.tokenizer.convert_tokens_to_string(new_tokens)
            
            prediction = self._get_prediction(new_text)
            if prediction != target_class:
                counterfactuals.append({
                    'type': 'deletion',
                    'original_text': self.tokenizer.convert_tokens_to_string(tokens),
                    'counterfactual_text': new_text,
                    'deleted_token': token,
                    'position': i,
                    'num_changes': 1,
                    'original_prediction': target_class,
                    'counterfactual_prediction': prediction
                })
        
        return counterfactuals
    
    def _find_word_insertions(self, tokens: List[str], target_class: int) -> List[Dict]:
        """Find word insertions that change prediction"""
        counterfactuals = []
        
        # Common insertion words
        insertion_words = ['not', 'very', 'extremely', 'quite', 'rather']
        
        for i in range(len(tokens) + 1):
            for word in insertion_words:
                new_tokens = tokens.copy()
                new_tokens.insert(i, word)
                new_text = self.tokenizer.convert_tokens_to_string(new_tokens)
                
                prediction = self._get_prediction(new_text)
                if prediction != target_class:
                    counterfactuals.append({
                        'type': 'insertion',
                        'original_text': self.tokenizer.convert_tokens_to_string(tokens),
                        'counterfactual_text': new_text,
                        'inserted_token': word,
                        'position': i,
                        'num_changes': 1,
                        'original_prediction': target_class,
                        'counterfactual_prediction': prediction
                    })
        
        return counterfactuals
    
    def _get_prediction(self, text: str) -> int:
        """Get model prediction for text"""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        
        return prediction
    
    def generate_explanation(self, counterfactual: Dict) -> str:
        """Generate human-readable explanation for counterfactual"""
        if counterfactual['type'] == 'replacement':
            explanation = f"The model predicted {counterfactual['original_prediction']} because of the word '{counterfactual['changed_token']}'. If you changed '{counterfactual['changed_token']}' to '{counterfactual['replacement_token']}', the prediction would change to {counterfactual['counterfactual_prediction']}."
        elif counterfactual['type'] == 'deletion':
            explanation = f"The word '{counterfactual['deleted_token']}' is important for the prediction. Removing it changes the prediction from {counterfactual['original_prediction']} to {counterfactual['counterfactual_prediction']}."
        elif counterfactual['type'] == 'insertion':
            explanation = f"Adding the word '{counterfactual['inserted_token']}' changes the prediction from {counterfactual['original_prediction']} to {counterfactual['counterfactual_prediction']}."
        else:
            explanation = "The counterfactual shows how the prediction would change with modifications."
        
        return explanation

# Example usage
def demonstrate_counterfactual_explanations():
    """Demonstrate counterfactual explanation generation"""
    print("Counterfactual Explanation System")
    print("This would require a trained sentiment analysis model")
    print("The framework above shows how to generate counterfactual explanations")
    
    return True

cf_demo = demonstrate_counterfactual_explanations()
```

## User-Adaptive Explanation Generation

### Adaptive Explanation System

```python
class AdaptiveExplanationGenerator:
    """Generate explanations adapted to user expertise level"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.explanation_templates = self._load_explanation_templates()
    
    def _load_explanation_templates(self) -> Dict[str, Dict[str, str]]:
        """Load explanation templates for different user types"""
        return {
            'beginner': {
                'attention': "The model focuses on these words when making its decision: {highlighted_words}",
                'gradient': "These words are most important for the prediction: {important_words}",
                'counterfactual': "If you changed '{word}' to '{replacement}', the result would be different.",
                'confidence': "The model is {confidence_level} confident about this prediction."
            },
            'intermediate': {
                'attention': "Attention weights show the model's focus on {attention_details}",
                'gradient': "Integrated gradients indicate {gradient_details} contribute most to the prediction",
                'counterfactual': "Counterfactual analysis reveals that {cf_details} are key decision factors",
                'confidence': "Model confidence is {confidence_score} with uncertainty in {uncertain_areas}"
            },
            'expert': {
                'attention': "Multi-head attention analysis reveals {technical_details} with head {head_id} showing {head_analysis}",
                'gradient': "Gradient attribution shows {gradient_details} with integrated gradients of {ig_scores}",
                'counterfactual': "Counterfactual generation with {cf_method} shows minimal perturbation at {perturbation_level}",
                'confidence': "Bayesian uncertainty estimation yields {confidence_metrics} with epistemic uncertainty {epistemic} and aleatoric uncertainty {aleatoric}"
            }
        }
    
    def generate_adaptive_explanation(self, text: str, 
                                   user_type: str = 'intermediate',
                                   explanation_type: str = 'comprehensive') -> str:
        """Generate explanation adapted to user expertise"""
        
        if user_type not in self.explanation_templates:
            user_type = 'intermediate'
        
        # Generate different types of explanations
        attention_explanation = self._generate_attention_explanation(text, user_type)
        gradient_explanation = self._generate_gradient_explanation(text, user_type)
        counterfactual_explanation = self._generate_counterfactual_explanation(text, user_type)
        confidence_explanation = self._generate_confidence_explanation(text, user_type)
        
        # Combine based on explanation type
        if explanation_type == 'attention_only':
            return attention_explanation
        elif explanation_type == 'gradient_only':
            return gradient_explanation
        elif explanation_type == 'counterfactual_only':
            return counterfactual_explanation
        elif explanation_type == 'confidence_only':
            return confidence_explanation
        else:  # comprehensive
            return self._combine_explanations([
                attention_explanation,
                gradient_explanation,
                counterfactual_explanation,
                confidence_explanation
            ], user_type)
    
    def _generate_attention_explanation(self, text: str, user_type: str) -> str:
        """Generate attention-based explanation"""
        # Extract attention weights (simplified)
        tokens = self.tokenizer.tokenize(text)
        
        # Mock attention analysis
        if user_type == 'beginner':
            highlighted_words = tokens[:3]  # First few words
            return self.explanation_templates[user_type]['attention'].format(
                highlighted_words=', '.join(highlighted_words)
            )
        elif user_type == 'intermediate':
            attention_details = f"key tokens {tokens[0]} and {tokens[-1]} with weights 0.8 and 0.6"
            return self.explanation_templates[user_type]['attention'].format(
                attention_details=attention_details
            )
        else:  # expert
            technical_details = f"attention patterns across {len(tokens)} tokens"
            head_id = 0
            head_analysis = f"strong focus on {tokens[0]}"
            return self.explanation_templates[user_type]['attention'].format(
                technical_details=technical_details,
                head_id=head_id,
                head_analysis=head_analysis
            )
    
    def _generate_gradient_explanation(self, text: str, user_type: str) -> str:
        """Generate gradient-based explanation"""
        tokens = self.tokenizer.tokenize(text)
        
        if user_type == 'beginner':
            important_words = tokens[:2]
            return self.explanation_templates[user_type]['gradient'].format(
                important_words=', '.join(important_words)
            )
        elif user_type == 'intermediate':
            gradient_details = f"tokens {tokens[0]} and {tokens[1]}"
            return self.explanation_templates[user_type]['gradient'].format(
                gradient_details=gradient_details
            )
        else:  # expert
            gradient_details = f"tokens {tokens[0]} and {tokens[1]}"
            ig_scores = "[0.8, 0.6, 0.4]"
            return self.explanation_templates[user_type]['gradient'].format(
                gradient_details=gradient_details,
                ig_scores=ig_scores
            )
    
    def _generate_counterfactual_explanation(self, text: str, user_type: str) -> str:
        """Generate counterfactual explanation"""
        tokens = self.tokenizer.tokenize(text)
        
        if user_type == 'beginner':
            word = tokens[0] if tokens else "word"
            replacement = "different" if tokens else "word"
            return self.explanation_templates[user_type]['counterfactual'].format(
                word=word, replacement=replacement
            )
        elif user_type == 'intermediate':
            cf_details = f"words like '{tokens[0]}' and '{tokens[1]}'"
            return self.explanation_templates[user_type]['counterfactual'].format(
                cf_details=cf_details
            )
        else:  # expert
            cf_method = "integrated gradients"
            perturbation_level = "0.1"
            return self.explanation_templates[user_type]['counterfactual'].format(
                cf_method=cf_method,
                perturbation_level=perturbation_level
            )
    
    def _generate_confidence_explanation(self, text: str, user_type: str) -> str:
        """Generate confidence explanation"""
        if user_type == 'beginner':
            confidence_level = "very confident"
            return self.explanation_templates[user_type]['confidence'].format(
                confidence_level=confidence_level
            )
        elif user_type == 'intermediate':
            confidence_score = "0.85"
            uncertain_areas = "edge cases"
            return self.explanation_templates[user_type]['confidence'].format(
                confidence_score=confidence_score,
                uncertain_areas=uncertain_areas
            )
        else:  # expert
            confidence_metrics = "0.85 ± 0.05"
            epistemic = "0.03"
            aleatoric = "0.02"
            return self.explanation_templates[user_type]['confidence'].format(
                confidence_metrics=confidence_metrics,
                epistemic=epistemic,
                aleatoric=aleatoric
            )
    
    def _combine_explanations(self, explanations: List[str], user_type: str) -> str:
        """Combine multiple explanations"""
        if user_type == 'beginner':
            return "\n\n".join(explanations)
        elif user_type == 'intermediate':
            return "\n\n".join([f"{i+1}. {exp}" for i, exp in enumerate(explanations)])
        else:  # expert
            return "\n\n".join([f"**{exp_type}**: {exp}" for exp_type, exp in zip(
                ["Attention Analysis", "Gradient Attribution", "Counterfactual Analysis", "Uncertainty Estimation"],
                explanations
            )])

# Example usage
def demonstrate_adaptive_explanations():
    """Demonstrate adaptive explanation generation"""
    print("Adaptive Explanation Generation System")
    
    # Mock tokenizer for demonstration
    class MockTokenizer:
        def tokenize(self, text):
            return text.split()
    
    tokenizer = MockTokenizer()
    model = None  # Would be actual model
    
    explainer = AdaptiveExplanationGenerator(model, tokenizer)
    
    text = "This movie is absolutely fantastic and amazing"
    
    for user_type in ['beginner', 'intermediate', 'expert']:
        print(f"\n--- {user_type.upper()} EXPLANATION ---")
        explanation = explainer.generate_adaptive_explanation(text, user_type)
        print(explanation)
    
    return explainer

adaptive_demo = demonstrate_adaptive_explanations()
```

## Evaluation Metrics for Explainability

### XAI Evaluation Framework

```python
class XAIEvaluator:
    """Evaluate the quality of explanations"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_faithfulness(self, model, explanation_method, 
                            test_data: List[Dict]) -> float:
        """Evaluate how faithful explanations are to model behavior"""
        
        faithfulness_scores = []
        
        for sample in test_data:
            text = sample['text']
            true_label = sample['label']
            
            # Get explanation
            explanation = explanation_method(text)
            
            # Test if removing important features reduces confidence
            important_features = self._extract_important_features(explanation)
            
            # Original prediction
            original_pred = self._get_prediction(model, text)
            
            # Modified prediction (remove important features)
            modified_text = self._remove_features(text, important_features)
            modified_pred = self._get_prediction(model, modified_text)
            
            # Faithfulness: how much prediction changes
            faithfulness = abs(original_pred - modified_pred)
            faithfulness_scores.append(faithfulness)
        
        return np.mean(faithfulness_scores)
    
    def evaluate_completeness(self, model, explanation_method, 
                            test_data: List[Dict]) -> float:
        """Evaluate how complete explanations are"""
        
        completeness_scores = []
        
        for sample in test_data:
            text = sample['text']
            
            # Get explanation
            explanation = explanation_method(text)
            
            # Extract explained features
            explained_features = self._extract_explained_features(explanation)
            
            # Get all important features (using a different method)
            all_important = self._get_all_important_features(model, text)
            
            # Completeness: ratio of explained to all important features
            completeness = len(explained_features) / len(all_important) if all_important else 0
            completeness_scores.append(completeness)
        
        return np.mean(completeness_scores)
    
    def evaluate_consistency(self, explanation_method, 
                           similar_samples: List[Dict]) -> float:
        """Evaluate consistency across similar samples"""
        
        explanations = []
        for sample in similar_samples:
            explanation = explanation_method(sample['text'])
            explanations.append(explanation)
        
        # Compute pairwise similarity between explanations
        similarities = []
        for i in range(len(explanations)):
            for j in range(i + 1, len(explanations)):
                similarity = self._compute_explanation_similarity(
                    explanations[i], explanations[j]
                )
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def evaluate_user_satisfaction(self, explanations: List[str], 
                                 user_ratings: List[int]) -> float:
        """Evaluate user satisfaction with explanations"""
        return np.mean(user_ratings)
    
    def comprehensive_evaluation(self, model, explanation_method, 
                               test_data: List[Dict]) -> Dict[str, float]:
        """Comprehensive evaluation of explanation quality"""
        
        results = {}
        
        # Faithfulness
        results['faithfulness'] = self.evaluate_faithfulness(
            model, explanation_method, test_data
        )
        
        # Completeness
        results['completeness'] = self.evaluate_completeness(
            model, explanation_method, test_data
        )
        
        # Consistency (requires similar samples)
        similar_samples = test_data[:10]  # Use subset for demonstration
        results['consistency'] = self.evaluate_consistency(
            explanation_method, similar_samples
        )
        
        # Overall score
        results['overall_score'] = np.mean(list(results.values()))
        
        return results
    
    def _extract_important_features(self, explanation):
        """Extract important features from explanation"""
        # Simplified implementation
        return ['feature1', 'feature2']
    
    def _extract_explained_features(self, explanation):
        """Extract explained features from explanation"""
        # Simplified implementation
        return ['feature1', 'feature2']
    
    def _get_all_important_features(self, model, text):
        """Get all important features using alternative method"""
        # Simplified implementation
        return ['feature1', 'feature2', 'feature3']
    
    def _remove_features(self, text, features):
        """Remove specified features from text"""
        # Simplified implementation
        return text
    
    def _get_prediction(self, model, text):
        """Get model prediction"""
        # Simplified implementation
        return 0.8
    
    def _compute_explanation_similarity(self, exp1, exp2):
        """Compute similarity between two explanations"""
        # Simplified implementation using text similarity
        return 0.7

# Example usage
def demonstrate_xai_evaluation():
    """Demonstrate XAI evaluation framework"""
    print("XAI Evaluation Framework")
    print("This framework provides comprehensive evaluation of explanation quality")
    
    evaluator = XAIEvaluator()
    
    # Mock test data
    test_data = [
        {'text': 'This movie is great', 'label': 1},
        {'text': 'This movie is terrible', 'label': 0},
        {'text': 'I love this film', 'label': 1}
    ]
    
    # Mock explanation method
    def mock_explanation_method(text):
        return f"Explanation for: {text}"
    
    # Mock model
    class MockModel:
        pass
    
    model = MockModel()
    
    # Evaluate
    results = evaluator.comprehensive_evaluation(model, mock_explanation_method, test_data)
    
    print("Evaluation Results:")
    for metric, score in results.items():
        print(f"{metric}: {score:.3f}")
    
    return evaluator

xai_eval_demo = demonstrate_xai_evaluation()
```

## Practical Exercises

### Exercise 1: Attention Visualization
Implement a comprehensive attention visualization system:
- Create heatmaps for single and multi-head attention
- Build interactive visualizations with Plotly
- Implement attention entropy analysis

### Exercise 2: Gradient-Based Attribution
Build an integrated gradients system:
- Implement the integrated gradients algorithm
- Create attribution visualizations
- Compare with other attribution methods

### Exercise 3: Counterfactual Analysis
Develop a counterfactual explanation system:
- Generate minimal counterfactual examples
- Implement multiple perturbation strategies
- Create human-readable explanations

## Assessment Questions

1. **What is the primary goal of Explainable AI in NLP?**
   - To make model decisions transparent and understandable
   - To improve model accuracy
   - To reduce computational costs
   - To increase model speed

2. **Which method provides the most interpretable explanations?**
   - Attention visualization
   - Gradient-based attribution
   - Counterfactual analysis
   - All of the above (depends on use case)

3. **How can explanations be adapted to different user types?**
   - Using different technical detail levels
   - Varying explanation complexity
   - Customizing terminology
   - All of the above

## Key Takeaways

- Explainable AI is crucial for building trust in NLP systems
- Multiple explanation methods provide different insights
- User-adaptive explanations improve understanding
- Evaluation metrics ensure explanation quality
- Counterfactual analysis reveals decision boundaries
- Attention visualization shows model focus
- Gradient-based methods quantify feature importance

## Next Steps

This module provides the foundation for building transparent and trustworthy NLP systems. In the next module, we'll explore Reinforcement Learning applications in NLP, which is another crucial area for 2025 NLP research.
