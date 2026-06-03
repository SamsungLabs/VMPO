import torch
import torch.nn as nn
from typing import Optional
from diffusers.models.embeddings import TimestepEmbedding, Timesteps


class ConditionalFPsiSD3(nn.Module):
    """
    A model based on SD3Transformer2DModel that takes only timestep and text conditioning
    (encoder_hidden_states and pooled_projections) as input and outputs a scalar value.
    
    This is useful for learning a value function or baseline that depends only on 
    the prompt and timestep, but not on the latent image.
    """
    
    def __init__(
        self,
        # Text conditioning dimensions (from SD3)
        caption_projection_dim: int = 1536,  # joint_attention_dim in SD3
        pooled_projection_dim: int = 2048,   # from SD3
        
        # Time embedding parameters
        num_attention_heads: int = 24,       # from SD3 config
        attention_head_dim: int = 64,        # from SD3 config
        
        # Architecture parameters
        num_layers: int = 4,                 # Reduced from SD3's 24 layers
        
        # Optional customization
        hidden_dim: Optional[int] = None,
        output_dim: int = 1,
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = num_attention_heads * attention_head_dim
        
        self.hidden_dim = hidden_dim
        
        # Time embedding (similar to SD3)
        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.time_embedding = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=hidden_dim,
            act_fn="silu",
        )
        
        # Project pooled text embeddings (similar to SD3's time_text_embed)
        self.pooled_text_proj = nn.Sequential(
            nn.Linear(pooled_projection_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Project caption embeddings to hidden dimension
        self.caption_proj = nn.Linear(caption_projection_dim, hidden_dim)
        
        # Transformer blocks (simplified from SD3's JointTransformerBlock)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
            )
            for _ in range(num_layers)
        ])
        
        # Layer norm
        self.norm_out = nn.LayerNorm(hidden_dim, elementwise_affine=True, eps=1e-6)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embedding.linear_1.weight, std=0.02)
        nn.init.normal_(self.time_embedding.linear_2.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.scale_shift_table[1].weight, 0)
            nn.init.constant_(block.scale_shift_table[1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.output_proj[-1].weight, 0)  # Last linear layer
        nn.init.constant_(self.output_proj[-1].bias, 0)

    
    def forward(
        self,
        timestep: torch.Tensor,                    # (batch_size,)
        encoder_hidden_states: torch.Tensor,       # (batch_size, seq_len, caption_projection_dim)
        pooled_projections: torch.Tensor,          # (batch_size, pooled_projection_dim)
    ) -> torch.Tensor:
        """
        Args:
            timestep: Timestep tensor of shape (batch_size,)
            encoder_hidden_states: Text encoder outputs (batch_size, seq_len, 1536)
            pooled_projections: Pooled text embeddings (batch_size, 2048)
            
        Returns:
            Scalar output of shape (batch_size,) or (batch_size, output_dim)
        """
        batch_size = timestep.shape[0]
        
        # Embed timestep
        t_emb = self.time_proj(timestep)                     # (batch_size, 256)
        t_emb = t_emb.to(dtype=self.time_embedding.linear_1.weight.dtype)
        time_embed = self.time_embedding(t_emb)              # (batch_size, hidden_dim)
        
        # Project pooled text embeddings and add to time embedding
        pooled_embed = self.pooled_text_proj(pooled_projections)  # (batch_size, hidden_dim)
        conditioning = time_embed + pooled_embed             # (batch_size, hidden_dim)
        
        # Project caption embeddings
        caption_tokens = self.caption_proj(encoder_hidden_states)  # (batch_size, seq_len, hidden_dim)

        # Pass through transformer blocks
        hidden_states = caption_tokens
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                conditioning=conditioning,
            )
        
        # Global average pooling over sequence dimension
        hidden_states = hidden_states.mean(dim=1)            # (batch_size, hidden_dim)

        # Normalize and project to output
        hidden_states = self.norm_out(hidden_states)
        output = self.output_proj(hidden_states)             # (batch_size, output_dim)
        
        # Squeeze if output_dim is 1
        if output.shape[-1] == 1:
            output = output.squeeze(-1)                      # (batch_size,)
        
        return output


class TransformerBlock(nn.Module):
    """
    Simplified transformer block with self-attention and conditioning.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        # Self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        
        # Conditioning modulation (like AdaLN)
        self.scale_shift_table = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        
        # Feedforward network
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,     # (batch_size, seq_len, hidden_dim)
        conditioning: torch.Tensor,      # (batch_size, hidden_dim)
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Get modulation parameters from conditioning
        scale_shift = self.scale_shift_table(conditioning)   # (batch_size, 6 * hidden_dim)
        scale_shift = scale_shift.unsqueeze(1)               # (batch_size, 1, 6 * hidden_dim)
        scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = scale_shift.chunk(6, dim=-1)
        
        # Self-attention with adaptive layer norm
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = hidden_states * (1 + scale_msa) + shift_msa
        hidden_states, _ = self.attn(hidden_states, hidden_states, hidden_states)
        hidden_states = gate_msa * hidden_states + residual
        
        # Feedforward with adaptive layer norm
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = hidden_states * (1 + scale_mlp) + shift_mlp
        hidden_states = self.ff(hidden_states)
        hidden_states = gate_mlp * hidden_states + residual
        
        return hidden_states


# Example usage
if __name__ == "__main__":
    # python -m flow_grpo.f_psi
    # Create model
    model = ConditionalFPsiSD3(
        caption_projection_dim=4096,
        pooled_projection_dim=2048,
        num_attention_heads=32,
        attention_head_dim=128,
        num_layers=4,
        output_dim=1,
    )
    
    # Example inputs
    batch_size = 4
    seq_len = 205
    
    timestep = torch.randn(batch_size)
    encoder_hidden_states = torch.randn(batch_size, seq_len, 4096)
    pooled_projections = torch.randn(batch_size, 2048)
    print(timestep)
    print(encoder_hidden_states.shape)
    print(pooled_projections.shape)
    
    # Forward pass
    output = model(
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
    )
    
    print(f"Output shape: {output.shape}")  # Should be (batch_size,)
    print(f"Output values: {output}")