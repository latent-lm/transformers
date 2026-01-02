import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FlowMatchingModel(nn.Module):
    """
    Flow Matching model for mapping latent hidden states to semantic space.
    Uses conditional flow matching with optimal transport paths.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        semantic_dim: int,
        time_embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Dimension of input hidden states from language model
            semantic_dim: Dimension of the target semantic space
            time_embed_dim: Dimension for time step embeddings
            num_layers: Number of transformer layers in the flow network
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.semantic_dim = semantic_dim
        self.time_embed_dim = time_embed_dim
        
        # Time embedding network (sinusoidal + MLP)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim)
        )
        
        # Project hidden state to semantic dimension
        self.hidden_proj = nn.Linear(hidden_dim, semantic_dim)
        
        # Flow velocity network (predicts dx/dt)
        self.velocity_net = VelocityNetwork(
            semantic_dim=semantic_dim,
            time_embed_dim=time_embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
    def get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal time embeddings.
        
        Args:
            t: Time steps, shape (batch_size,)
            
        Returns:
            Time embeddings, shape (batch_size, time_embed_dim)
        """
        half_dim = self.time_embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_mlp(emb)
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict velocity field v_t(x_t | condition).
        
        Args:
            x_t: Current point in flow, shape (batch_size, semantic_dim)
            t: Time steps in [0, 1], shape (batch_size,)
            condition: Conditioning hidden states, shape (batch_size, hidden_dim)
            
        Returns:
            Predicted velocity, shape (batch_size, semantic_dim)
        """
        # Get time embeddings
        t_emb = self.get_time_embedding(t)
        
        # Project condition to semantic space
        cond_proj = self.hidden_proj(condition)
        
        # Predict velocity
        v_t = self.velocity_net(x_t, t_emb, cond_proj)
        
        return v_t
    
    def sample(
        self,
        condition: torch.Tensor,
        num_steps: int = 100,
        method: str = 'euler'
    ) -> torch.Tensor:
        """
        Sample from the flow by integrating from t=0 to t=1.
        
        Args:
            condition: Conditioning hidden states, shape (batch_size, hidden_dim)
            num_steps: Number of integration steps
            method: Integration method ('euler' or 'midpoint')
            
        Returns:
            Final semantic space points, shape (batch_size, semantic_dim)
        """
        original_size = condition.size()
        hidden_size = original_size[-1]
        condition = condition.reshape(-1, hidden_size)

        batch_size = condition.shape[0]
        device = condition.device
        
        # Start from Gaussian noise
        x = torch.randn(batch_size, self.semantic_dim, device=device)
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            
            if method == 'euler':
                # Euler method
                v = self.forward(x, t, condition)
                x = x + v * dt
            elif method == 'midpoint':
                # Midpoint method (more accurate)
                v1 = self.forward(x, t, condition)
                x_mid = x + v1 * (dt / 2)
                t_mid = t + dt / 2
                v2 = self.forward(x_mid, t_mid, condition)
                x = x + v2 * dt
            else:
                raise ValueError(f"Unknown method: {method}")
        
        x = x.reshape(original_size)
        return x
    
    # TODO: Modify this
    def compute_loss(
        self,
        condition: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flow matching loss using conditional optimal transport.
        
        Args:
            condition: Conditioning hidden states, shape (batch_size, hidden_dim)
            target: Target points in semantic space, shape (batch_size, semantic_dim)
            
        Returns:
            Flow matching loss
        """
        batch_size = condition.shape[0]
        device = condition.device
        
        # Sample random time steps
        t = torch.rand(batch_size, device=device)
        
        # Sample noise
        x_0 = torch.randn_like(target)
        
        # Linear interpolation path (optimal transport for Gaussian)
        x_t = (1 - t[:, None]) * x_0 + t[:, None] * target
        
        # Target velocity (derivative of interpolation path)
        u_t = target - x_0
        
        # Predict velocity
        v_t = self.forward(x_t, t, condition)
        
        # MSE loss between predicted and target velocity
        loss = F.mse_loss(v_t, u_t)
        
        return loss


class VelocityNetwork(nn.Module):
    """
    Neural network for predicting velocity field in flow matching.
    Uses transformer architecture with time and condition modulation.
    """
    
    def __init__(
        self,
        semantic_dim: int,
        time_embed_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.semantic_dim = semantic_dim
        
        # Input projection
        self.input_proj = nn.Linear(semantic_dim, semantic_dim)
        
        # Transformer layers with adaptive layer norm
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=semantic_dim,
                num_heads=num_heads,
                mlp_ratio=4,
                dropout=dropout,
                time_embed_dim=time_embed_dim
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, semantic_dim)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Current points, shape (batch_size, semantic_dim)
            t_emb: Time embeddings, shape (batch_size, time_embed_dim)
            condition: Projected conditions, shape (batch_size, semantic_dim)
            
        Returns:
            Velocity prediction, shape (batch_size, semantic_dim)
        """
        # Combine input with condition
        h = self.input_proj(x) + condition
        
        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, t_emb)
        
        # Project to output
        v = self.output_proj(h)
        
        return v


class TransformerBlock(nn.Module):
    """
    Transformer block with adaptive layer normalization based on time.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        time_embed_dim: int = 256
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )
        
        # Adaptive layer norm parameters (scale and shift from time)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, dim * 4)
        )
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input, shape (batch_size, semantic_dim)
            t_emb: Time embeddings, shape (batch_size, time_embed_dim)
            
        Returns:
            Output, shape (batch_size, semantic_dim)
        """
        # Get adaptive parameters
        ada_params = self.adaLN(t_emb)
        scale1, shift1, scale2, shift2 = ada_params.chunk(4, dim=-1)
        
        # Self-attention with adaptive norm
        x_norm = self.norm1(x) * (1 + scale1) + shift1
        x_attn, _ = self.attn(
            x_norm.unsqueeze(1),
            x_norm.unsqueeze(1),
            x_norm.unsqueeze(1)
        )
        x = x + x_attn.squeeze(1)
        
        # MLP with adaptive norm
        x_norm = self.norm2(x) * (1 + scale2) + shift2
        x = x + self.mlp(x_norm)
        
        return x


# Example usage
if __name__ == "__main__":
    # Model parameters
    hidden_dim = 768  # From language model
    semantic_dim = 512  # Target semantic space
    batch_size = 4
    
    # Initialize model
    flow_model = FlowMatchingModel(
        hidden_dim=hidden_dim,
        semantic_dim=semantic_dim,
        time_embed_dim=256,
        num_layers=4,
        num_heads=8
    )
    
    # Example: Training step
    condition = torch.randn(batch_size, hidden_dim)  # From language model
    target = torch.randn(batch_size, semantic_dim)  # Target semantic embeddings
    
    loss = flow_model.compute_loss(condition, target)
    print(f"Training loss: {loss.item():.4f}")
    
    # Example: Sampling
    flow_model.eval()
    with torch.no_grad():
        samples = flow_model.sample(condition, num_steps=50, method='euler')
        print(f"Sampled semantic points shape: {samples.shape}")