# NLP Tutorial Module 14: Multimodal NLP and Applications (PhD Level)

## Learning Objectives
By the end of this module, you will be able to:
- Understand multimodal learning theory and cross-modal architectures
- Implement vision-language models (CLIP, DALL-E, Flamingo)
- Build audio-text models for speech and music
- Create video understanding systems
- Implement cross-modal retrieval and generation
- Apply multimodal models to real-world applications
- Understand deployment strategies and optimization techniques
- Design end-to-end production NLP systems

## Multimodal Learning Foundations

### Theory

Multimodal learning combines information from multiple modalities (text, vision, audio, video) to create richer representations.

**Key Challenges:**
1. **Alignment**: Matching concepts across modalities
2. **Fusion**: Combining information effectively
3. **Translation**: Generating one modality from another
4. **Representation**: Learning joint embeddings

### Mathematical Framework

For modalities $\mathcal{M}_1, \mathcal{M}_2$, learn joint representation:
$$f: \mathcal{M}_1 \times \mathcal{M}_2 \rightarrow \mathbb{R}^d$$

**Contrastive Learning**: Maximize similarity of matched pairs:
$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(f_1(m_1), f_2(m_2)) / \tau)}{\sum_{m_2'} \exp(\text{sim}(f_1(m_1), f_2(m_2')) / \tau)}$$

## Vision-Language Models

### CLIP (Contrastive Language-Image Pre-training)

CLIP learns joint vision-language representations through contrastive learning.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

class VisionEncoder(nn.Module):
    """Vision encoder (simplified ViT)"""
    def __init__(self, image_size: int = 224, patch_size: int = 16, 
                 embed_dim: int = 768, num_layers: int = 12, num_heads: int = 12):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection head
        self.projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch_size, 3, image_size, image_size)
        Returns:
            embeddings: (batch_size, embed_dim)
        """
        batch_size = images.shape[0]
        
        # Patch embedding
        x = self.patch_embed(images)  # (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Use CLS token
        x = x[:, 0]
        
        # Project
        x = self.projection(x)
        
        # L2 normalize
        x = F.normalize(x, p=2, dim=-1)
        
        return x

class TextEncoder(nn.Module):
    """Text encoder (simplified Transformer)"""
    def __init__(self, vocab_size: int = 50000, max_length: int = 77,
                 embed_dim: int = 512, num_layers: int = 12, num_heads: int = 8):
        super().__init__()
        
        self.max_length = max_length
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_length, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection head
        self.projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, text: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            text: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length)
        Returns:
            embeddings: (batch_size, embed_dim)
        """
        # Token embedding
        x = self.token_embed(text)
        
        # Add positional embedding
        x = x + self.pos_embed[:, :text.shape[1], :]
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to float mask for transformer
            mask = (1.0 - attention_mask.float()) * -10000.0
        else:
            mask = None
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=(attention_mask == 0) if attention_mask is not None else None)
        
        # Use EOS token (last non-padding token)
        if attention_mask is not None:
            # Get last non-padding token position
            seq_lengths = attention_mask.sum(dim=1) - 1
            x = x[torch.arange(x.shape[0]), seq_lengths]
        else:
            x = x[:, -1]
        
        # Project
        x = self.projection(x)
        
        # L2 normalize
        x = F.normalize(x, p=2, dim=-1)
        
        return x

class CLIP(nn.Module):
    """CLIP model for vision-language learning"""
    def __init__(self, vision_encoder: VisionEncoder, text_encoder: TextEncoder,
                 temperature: float = 0.07):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        
    def forward(self, images: torch.Tensor, text: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (batch_size, 3, image_size, image_size)
            text: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length)
        Returns:
            image_embeddings: (batch_size, embed_dim)
            text_embeddings: (batch_size, embed_dim)
        """
        # Encode images and text
        image_embeddings = self.vision_encoder(images)
        text_embeddings = self.text_encoder(text, attention_mask)
        
        return image_embeddings, text_embeddings
    
    def compute_loss(self, image_embeddings: torch.Tensor, 
                    text_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss"""
        # Compute similarity matrix
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeddings @ text_embeddings.t()
        logits_per_text = logits_per_image.t()
        
        # Create labels (diagonal is positive pair)
        batch_size = image_embeddings.shape[0]
        labels = torch.arange(batch_size, device=image_embeddings.device)
        
        # Compute cross-entropy loss
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        
        loss = (loss_i + loss_t) / 2
        
        return loss

class CLIPTrainer:
    """Trainer for CLIP model"""
    def __init__(self, model: CLIP, learning_rate: float = 1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
    def train_step(self, images: torch.Tensor, text: torch.Tensor,
                   attention_mask: Optional[torch.Tensor] = None) -> float:
        """Single training step"""
        self.model.train()
        
        # Forward pass
        image_embeddings, text_embeddings = self.model(images, text, attention_mask)
        
        # Compute loss
        loss = self.model.compute_loss(image_embeddings, text_embeddings)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def zero_shot_classify(self, image: torch.Tensor, text_descriptions: list) -> int:
        """Zero-shot classification"""
        self.model.eval()
        
        with torch.no_grad():
            # Encode image
            image_embedding = self.vision_encoder(image.unsqueeze(0))
            
            # Encode all text descriptions
            text_embeddings = []
            for text in text_descriptions:
                # In practice, use proper tokenizer
                text_tokens = torch.tensor([[1, 2, 3]])  # Placeholder
                text_emb = self.text_encoder(text_tokens)
                text_embeddings.append(text_emb)
            
            text_embeddings = torch.cat(text_embeddings, dim=0)
            
            # Compute similarities
            similarities = image_embedding @ text_embeddings.t()
            
            # Return index of most similar
            return similarities.argmax().item()

# Example usage
vision_encoder = VisionEncoder()
text_encoder = TextEncoder()
clip_model = CLIP(vision_encoder, text_encoder)

# Training
trainer = CLIPTrainer(clip_model)

# Simulate batch
batch_size = 32
images = torch.randn(batch_size, 3, 224, 224)
text = torch.randint(0, 50000, (batch_size, 77))
attention_mask = torch.ones(batch_size, 77)

loss = trainer.train_step(images, text, attention_mask)
print(f"Training loss: {loss:.4f}")
```

### Image Generation with Diffusion Models

```python
class DiffusionModel(nn.Module):
    """Text-conditioned diffusion model for image generation"""
    def __init__(self, image_size: int = 256, text_embed_dim: int = 512):
        super().__init__()
        
        self.image_size = image_size
        
        # U-Net architecture for denoising
        self.unet = self._build_unet(text_embed_dim)
        
        # Noise schedule
        self.register_buffer('betas', self._cosine_beta_schedule(1000))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
    def _build_unet(self, text_embed_dim: int) -> nn.Module:
        """Build U-Net for denoising"""
        # Simplified U-Net
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )
    
    def _cosine_beta_schedule(self, timesteps: int) -> torch.Tensor:
        """Cosine noise schedule"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, 
                text_embed: torch.Tensor) -> torch.Tensor:
        """Predict noise"""
        # In practice, incorporate text_embed through cross-attention
        noise_pred = self.unet(x)
        return noise_pred
    
    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to image at timestep t"""
        noise = torch.randn_like(x)
        
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod[t])
        
        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1)
        
        noisy_image = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
        
        return noisy_image, noise
    
    @torch.no_grad()
    def sample(self, text_embed: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Generate images from text"""
        # Start from random noise
        x = torch.randn(num_samples, 3, self.image_size, self.image_size, 
                       device=text_embed.device)
        
        # Denoise step by step
        for t in reversed(range(len(self.betas))):
            t_batch = torch.full((num_samples,), t, device=x.device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.forward(x, t_batch, text_embed)
            
            # Remove noise
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            # Compute x_{t-1}
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * noise_pred
            ) + torch.sqrt(beta_t) * noise
        
        return x

# Example usage
diffusion_model = DiffusionModel()

# Training
images = torch.randn(4, 3, 256, 256)
text_embeddings = torch.randn(4, 512)
timesteps = torch.randint(0, 1000, (4,))

noisy_images, noise = diffusion_model.add_noise(images, timesteps)
noise_pred = diffusion_model(noisy_images, timesteps, text_embeddings)
loss = F.mse_loss(noise_pred, noise)

# Sampling
generated_images = diffusion_model.sample(text_embeddings, num_samples=4)
```

## Audio-Text Models

### Speech Recognition and Synthesis

```python
class AudioTextModel(nn.Module):
    """Audio-text multimodal model"""
    def __init__(self, audio_dim: int = 80, text_vocab_size: int = 50000,
                 hidden_dim: int = 512, num_layers: int = 6):
        super().__init__()
        
        # Audio encoder (for spectrograms)
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(audio_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Text encoder
        self.text_embed = nn.Embedding(text_vocab_size, hidden_dim)
        
        # Cross-modal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.text_head = nn.Linear(hidden_dim, text_vocab_size)  # Speech recognition
        self.audio_head = nn.Linear(hidden_dim, audio_dim)  # Speech synthesis
        
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to hidden states
        
        Args:
            audio: (batch_size, audio_dim, time_steps)
        """
        x = self.audio_encoder(audio)
        x = x.transpose(1, 2)  # (batch_size, time_steps, hidden_dim)
        return x
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text to hidden states"""
        return self.text_embed(text)
    
    def speech_recognition(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert speech to text"""
        # Encode audio
        hidden = self.encode_audio(audio)
        
        # Transform
        hidden = self.transformer(hidden)
        
        # Decode to text
        logits = self.text_head(hidden)
        
        return logits
    
    def text_to_speech(self, text: torch.Tensor) -> torch.Tensor:
        """Convert text to speech"""
        # Encode text
        hidden = self.encode_text(text)
        
        # Transform
        hidden = self.transformer(hidden)
        
        # Decode to audio
        audio_features = self.audio_head(hidden)
        audio_features = audio_features.transpose(1, 2)
        
        return audio_features

# Example usage
model = AudioTextModel()

# Speech recognition
audio = torch.randn(2, 80, 100)  # Batch of spectrograms
text_logits = model.speech_recognition(audio)
print(f"Text logits shape: {text_logits.shape}")

# Text to speech
text = torch.randint(0, 50000, (2, 50))
audio_features = model.text_to_speech(text)
print(f"Audio features shape: {audio_features.shape}")
```

## Video Understanding

```python
class VideoLanguageModel(nn.Module):
    """Video-language understanding model"""
    def __init__(self, frame_size: int = 224, text_vocab_size: int = 50000,
                 hidden_dim: int = 768, num_frames: int = 16):
        super().__init__()
        
        self.num_frames = num_frames
        
        # Frame encoder (CNN + Transformer)
        self.frame_encoder = VisionEncoder(image_size=frame_size, embed_dim=hidden_dim)
        
        # Temporal encoder
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=4
        )
        
        # Text encoder
        self.text_encoder = TextEncoder(vocab_size=text_vocab_size, embed_dim=hidden_dim)
        
        # Cross-modal fusion
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=4
        )
        
        # Task heads
        self.video_text_matching = nn.Linear(hidden_dim, 1)
        self.caption_head = nn.Linear(hidden_dim, text_vocab_size)
        
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video frames
        
        Args:
            video: (batch_size, num_frames, 3, height, width)
        """
        batch_size, num_frames, c, h, w = video.shape
        
        # Encode each frame
        frames = video.view(batch_size * num_frames, c, h, w)
        frame_embeddings = self.frame_encoder(frames)
        frame_embeddings = frame_embeddings.view(batch_size, num_frames, -1)
        
        # Temporal encoding
        video_embedding = self.temporal_encoder(frame_embeddings)
        
        return video_embedding
    
    def video_text_match(self, video: torch.Tensor, text: torch.Tensor,
                        attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute video-text matching score"""
        # Encode modalities
        video_emb = self.encode_video(video)
        text_emb = self.text_encoder(text, attention_mask)
        
        # Expand text for fusion
        text_emb = text_emb.unsqueeze(1).expand(-1, video_emb.shape[1], -1)
        
        # Fuse
        combined = torch.cat([video_emb, text_emb], dim=1)
        fused = self.fusion(combined)
        
        # Pool and classify
        pooled = fused.mean(dim=1)
        score = self.video_text_matching(pooled)
        
        return score
    
    def generate_caption(self, video: torch.Tensor, max_length: int = 50) -> torch.Tensor:
        """Generate caption for video"""
        video_emb = self.encode_video(video)
        
        # Autoregressive generation
        batch_size = video.shape[0]
        generated = torch.zeros(batch_size, max_length, dtype=torch.long, device=video.device)
        
        for i in range(max_length):
            # Encode current text
            if i == 0:
                text_emb = torch.zeros(batch_size, 1, video_emb.shape[-1], device=video.device)
            else:
                text_tokens = generated[:, :i]
                text_emb = self.text_encoder.token_embed(text_tokens)
            
            # Fuse with video
            combined = torch.cat([video_emb, text_emb], dim=1)
            fused = self.fusion(combined)
            
            # Predict next token
            logits = self.caption_head(fused[:, -1, :])
            next_token = logits.argmax(dim=-1)
            generated[:, i] = next_token
        
        return generated

# Example usage
video_model = VideoLanguageModel()

# Video-text matching
video = torch.randn(2, 16, 3, 224, 224)
text = torch.randint(0, 50000, (2, 77))
attention_mask = torch.ones(2, 77)

match_score = video_model.video_text_match(video, text, attention_mask)
print(f"Match score: {match_score.shape}")

# Video captioning
captions = video_model.generate_caption(video)
print(f"Generated captions shape: {captions.shape}")
```

## Cross-Modal Retrieval

```python
class CrossModalRetrieval:
    """Cross-modal retrieval system"""
    def __init__(self, model: CLIP):
        self.model = model
        self.image_database = []
        self.text_database = []
        self.image_embeddings = None
        self.text_embeddings = None
        
    def index_images(self, images: list):
        """Index images into database"""
        self.model.eval()
        
        embeddings = []
        with torch.no_grad():
            for image in images:
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image)
                image = image.unsqueeze(0)
                
                emb = self.model.vision_encoder(image)
                embeddings.append(emb)
        
        self.image_database = images
        self.image_embeddings = torch.cat(embeddings, dim=0)
        
    def index_texts(self, texts: list):
        """Index texts into database"""
        self.model.eval()
        
        embeddings = []
        with torch.no_grad():
            for text in texts:
                # In practice, use proper tokenizer
                text_tokens = torch.randint(0, 50000, (1, 77))
                attention_mask = torch.ones(1, 77)
                
                emb = self.model.text_encoder(text_tokens, attention_mask)
                embeddings.append(emb)
        
        self.text_database = texts
        self.text_embeddings = torch.cat(embeddings, dim=0)
    
    def search_images_by_text(self, query_text: str, top_k: int = 5) -> list:
        """Search images using text query"""
        self.model.eval()
        
        with torch.no_grad():
            # Encode query
            query_tokens = torch.randint(0, 50000, (1, 77))
            attention_mask = torch.ones(1, 77)
            query_emb = self.model.text_encoder(query_tokens, attention_mask)
            
            # Compute similarities
            similarities = query_emb @ self.image_embeddings.t()
            
            # Get top-k
            top_k_indices = similarities.topk(top_k).indices.squeeze().tolist()
            
            if isinstance(top_k_indices, int):
                top_k_indices = [top_k_indices]
            
            return [self.image_database[i] for i in top_k_indices]
    
    def search_texts_by_image(self, query_image: torch.Tensor, top_k: int = 5) -> list:
        """Search texts using image query"""
        self.model.eval()
        
        with torch.no_grad():
            # Encode query
            query_emb = self.model.vision_encoder(query_image.unsqueeze(0))
            
            # Compute similarities
            similarities = query_emb @ self.text_embeddings.t()
            
            # Get top-k
            top_k_indices = similarities.topk(top_k).indices.squeeze().tolist()
            
            if isinstance(top_k_indices, int):
                top_k_indices = [top_k_indices]
            
            return [self.text_database[i] for i in top_k_indices]
    
    def search_images_by_image(self, query_image: torch.Tensor, top_k: int = 5) -> list:
        """Search similar images"""
        self.model.eval()
        
        with torch.no_grad():
            # Encode query
            query_emb = self.model.vision_encoder(query_image.unsqueeze(0))
            
            # Compute similarities
            similarities = query_emb @ self.image_embeddings.t()
            
            # Get top-k (excluding query itself)
            top_k_indices = similarities.topk(top_k + 1).indices.squeeze().tolist()[1:]
            
            if isinstance(top_k_indices, int):
                top_k_indices = [top_k_indices]
            
            return [self.image_database[i] for i in top_k_indices]

# Example usage
clip_model = CLIP(VisionEncoder(), TextEncoder())
retrieval = CrossModalRetrieval(clip_model)

# Index database
images = [torch.randn(3, 224, 224) for _ in range(100)]
texts = [f"Description {i}" for i in range(100)]

retrieval.index_images(images)
retrieval.index_texts(texts)

# Search
results = retrieval.search_images_by_text("A cat sitting on a couch", top_k=5)
print(f"Found {len(results)} images")
```

## Production Deployment

### Model Optimization

```python
class ModelOptimizer:
    """Optimize models for production"""
    
    @staticmethod
    def quantize_model(model: nn.Module, calibration_data: torch.utils.data.DataLoader):
        """Quantize model to INT8"""
        import torch.quantization as quantization
        
        # Prepare model
        model.eval()
        model.qconfig = quantization.get_default_qconfig('fbgemm')
        quantization.prepare(model, inplace=True)
        
        # Calibrate
        with torch.no_grad():
            for batch in calibration_data:
                model(batch)
        
        # Convert
        quantization.convert(model, inplace=True)
        
        return model
    
    @staticmethod
    def prune_model(model: nn.Module, amount: float = 0.3):
        """Prune model weights"""
        import torch.nn.utils.prune as prune
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')
        
        return model
    
    @staticmethod
    def compile_model(model: nn.Module):
        """Compile model with TorchScript"""
        model.eval()
        
        # Create example input
        example_input = torch.randn(1, 3, 224, 224)
        
        # Trace model
        traced_model = torch.jit.trace(model, example_input)
        traced_model = torch.jit.optimize_for_inference(traced_model)
        
        return traced_model
    
    @staticmethod
    def export_onnx(model: nn.Module, output_path: str):
        """Export to ONNX format"""
        model.eval()
        
        example_input = torch.randn(1, 3, 224, 224)
        
        torch.onnx.export(
            model,
            example_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

# Example usage
optimizer = ModelOptimizer()

# Quantization
# quantized_model = optimizer.quantize_model(model, calibration_loader)

# Pruning
# pruned_model = optimizer.prune_model(model, amount=0.3)

# Compilation
# compiled_model = optimizer.compile_model(model)

# ONNX export
# optimizer.export_onnx(model, "model.onnx")
```

### Serving Infrastructure

```python
from dataclasses import dataclass
from typing import Any, List, Dict
import asyncio
import time

@dataclass
class InferenceRequest:
    """Request for inference"""
    id: str
    input_data: Any
    timestamp: float

@dataclass
class InferenceResponse:
    """Response from inference"""
    id: str
    output: Any
    latency: float

class ModelServer:
    """Production model server"""
    def __init__(self, model: nn.Module, batch_size: int = 32, 
                 max_batch_delay: float = 0.05):
        self.model = model
        self.batch_size = batch_size
        self.max_batch_delay = max_batch_delay
        
        self.request_queue = asyncio.Queue()
        self.response_futures = {}
        
        # Start batch processor
        self.processor_task = None
        
    async def start(self):
        """Start server"""
        self.processor_task = asyncio.create_task(self._batch_processor())
        
    async def stop(self):
        """Stop server"""
        if self.processor_task:
            self.processor_task.cancel()
            await self.processor_task
    
    async def predict(self, input_data: Any) -> Any:
        """Make prediction"""
        # Create request
        request_id = f"req_{time.time()}"
        request = InferenceRequest(
            id=request_id,
            input_data=input_data,
            timestamp=time.time()
        )
        
        # Create future for response
        future = asyncio.Future()
        self.response_futures[request_id] = future
        
        # Add to queue
        await self.request_queue.put(request)
        
        # Wait for response
        response = await future
        
        return response.output
    
    async def _batch_processor(self):
        """Process requests in batches"""
        while True:
            try:
                # Collect batch
                batch_requests = []
                deadline = time.time() + self.max_batch_delay
                
                while len(batch_requests) < self.batch_size:
                    timeout = max(0, deadline - time.time())
                    
                    try:
                        request = await asyncio.wait_for(
                            self.request_queue.get(), 
                            timeout=timeout
                        )
                        batch_requests.append(request)
                    except asyncio.TimeoutError:
                        break
                
                if not batch_requests:
                    continue
                
                # Run inference
                responses = await self._run_batch_inference(batch_requests)
                
                # Send responses
                for response in responses:
                    future = self.response_futures.pop(response.id)
                    future.set_result(response)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in batch processor: {e}")
    
    async def _run_batch_inference(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Run inference on batch"""
        # Prepare batch
        inputs = [req.input_data for req in requests]
        
        # Run model
        start_time = time.time()
        
        with torch.no_grad():
            # Stack inputs
            batched_input = torch.stack(inputs)
            
            # Run inference
            outputs = self.model(batched_input)
            
        latency = time.time() - start_time
        
        # Create responses
        responses = []
        for i, request in enumerate(requests):
            response = InferenceResponse(
                id=request.id,
                output=outputs[i],
                latency=latency / len(requests)
            )
            responses.append(response)
        
        return responses

class LoadBalancer:
    """Load balancer for multiple model servers"""
    def __init__(self, servers: List[ModelServer]):
        self.servers = servers
        self.current_server = 0
        
    async def predict(self, input_data: Any) -> Any:
        """Distribute prediction across servers"""
        # Round-robin
        server = self.servers[self.current_server]
        self.current_server = (self.current_server + 1) % len(self.servers)
        
        return await server.predict(input_data)

# Example usage (conceptual)
# server = ModelServer(model, batch_size=32)
# await server.start()
# 
# result = await server.predict(input_data)
```

## Real-World Applications

### 1. Visual Question Answering

```python
class VQASystem:
    """Visual Question Answering system"""
    def __init__(self, vision_model, text_model):
        self.vision_model = vision_model
        self.text_model = text_model
        
    def answer_question(self, image: torch.Tensor, question: str) -> str:
        """Answer question about image"""
        # Encode image
        image_features = self.vision_model.encode_audio(image)
        
        # Encode question
        question_tokens = self._tokenize(question)
        question_features = self.text_model.encode_text(question_tokens)
        
        # Fuse and generate answer
        answer = self._generate_answer(image_features, question_features)
        
        return answer
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text"""
        # In practice, use proper tokenizer
        return torch.randint(0, 50000, (1, 50))
    
    def _generate_answer(self, image_features, question_features):
        """Generate answer"""
        # In practice, use proper decoder
        return "The answer is 42"
```

### 2. Content Moderation

```python
class ContentModerationSystem:
    """Multimodal content moderation"""
    def __init__(self, model):
        self.model = model
        self.safety_threshold = 0.8
        
    def moderate_content(self, image: Optional[torch.Tensor] = None,
                        text: Optional[str] = None,
                        video: Optional[torch.Tensor] = None) -> Dict:
        """Moderate multimodal content"""
        scores = {}
        
        # Analyze each modality
        if image is not None:
            scores['image'] = self._analyze_image(image)
        
        if text is not None:
            scores['text'] = self._analyze_text(text)
        
        if video is not None:
            scores['video'] = self._analyze_video(video)
        
        # Aggregate scores
        overall_score = np.mean(list(scores.values()))
        is_safe = overall_score < self.safety_threshold
        
        return {
            'is_safe': is_safe,
            'overall_score': overall_score,
            'modality_scores': scores,
            'flags': self._get_flags(scores)
        }
    
    def _analyze_image(self, image: torch.Tensor) -> float:
        """Analyze image safety"""
        # In practice, use safety classifier
        return np.random.random()
    
    def _analyze_text(self, text: str) -> float:
        """Analyze text safety"""
        # In practice, use toxicity classifier
        return np.random.random()
    
    def _analyze_video(self, video: torch.Tensor) -> float:
        """Analyze video safety"""
        # In practice, use video safety classifier
        return np.random.random()
    
    def _get_flags(self, scores: Dict) -> List[str]:
        """Get safety flags"""
        flags = []
        for modality, score in scores.items():
            if score > self.safety_threshold:
                flags.append(f"unsafe_{modality}")
        return flags
```

## Practical Exercises

### Exercise 1: Build CLIP from Scratch
Implement full CLIP model:
- Vision and text encoders
- Contrastive learning
- Zero-shot classification
- Image retrieval

### Exercise 2: Multimodal Generation
Create text-to-image system:
- Implement diffusion model
- Add text conditioning
- Train on dataset
- Evaluate quality

### Exercise 3: Video Understanding
Build video QA system:
- Temporal modeling
- Cross-modal attention
- Question answering
- Action recognition

### Exercise 4: Production Deployment
Deploy multimodal system:
- Model optimization
- Batch inference
- API server
- Monitoring

## Research Directions

### Open Problems:
1. **Efficient Training**: Reducing computational cost
2. **Cross-Modal Alignment**: Better representation learning
3. **Long-Form Video**: Handling extended sequences
4. **Zero-Shot Transfer**: Generalizing to new tasks
5. **Safety and Fairness**: Ensuring responsible AI

## Key Papers

1. Radford et al. (2021) - CLIP
2. Ramesh et al. (2021) - DALL-E
3. Alayrac et al. (2022) - Flamingo
4. Rombach et al. (2022) - Stable Diffusion
5. Jia et al. (2021) - ALIGN
6. Li et al. (2022) - BLIP
7. Wang et al. (2022) - Image as a Foreign Language
8. Reed et al. (2022) - Generalist Agent (Gato)

## Key Takeaways

- Multimodal learning combines complementary information from multiple modalities
- Contrastive learning is powerful for cross-modal alignment
- Diffusion models enable high-quality generation
- Production deployment requires optimization and infrastructure
- Real-world applications span vision, audio, and video domains
- Safety and fairness are critical for deployment

## Conclusion

Congratulations! You've completed this comprehensive PhD-level NLP tutorial covering:
- Text processing and mathematical foundations
- Traditional and neural NLP models
- Word embeddings and recurrent architectures
- Attention mechanisms and Transformers
- BERT and pre-trained models
- Advanced architectures and prompting
- LLM training and reasoning
- Model evaluation and alignment
- Multimodal learning and applications

You now have the knowledge to tackle cutting-edge NLP research and build production systems!

