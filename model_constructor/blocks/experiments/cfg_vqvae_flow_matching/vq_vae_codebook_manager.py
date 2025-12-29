import torch
import torch.nn as nn

class VQCodebookManager(nn.Module):
    def __init__(self, num_q_vectors, vec_dim):
        super().__init__()
        self.vq_codebook = nn.Embedding(num_q_vectors, vec_dim)

        # Optional: Initialize weights to be uniform for better initial convergence
        self.vq_codebook.weight.data.uniform_(-1.0 / num_q_vectors, 1.0 / num_q_vectors)
    
    def forward(self, continuous_vec: torch.Tensor):
        """
        Parameters: 
            continuous_vec: (batch, features) or (batch, 1, features) shape
        
        Returns:
            q_vec: same shape as continuous_vec, where each last-dim vector is
                   replaced by its nearest codebook embedding (L2 distance).
        """
        if continuous_vec.dim() < 2:
            raise ValueError(
                f"continuous_vec must have shape (..., features). Got {tuple(continuous_vec.shape)}"
            )
        d = continuous_vec.size(-1)
        if d != self.vq_codebook.embedding_dim:
            raise ValueError(
                f"Last dim of continuous_vec ({d}) must match embedding_dim "
                f"({self.vq_codebook.embedding_dim})."
            )
        
        og_shape = continuous_vec.shape
        x = continuous_vec.reshape(-1, d)

        with torch.no_grad():
            # Compute distances in float32 for numerical stability
            x_f = x.float()  # (N, D)
            w_f = self.vq_codebook.weight.float()  # (K, D)

            # Squared L2 distance
            x2 = (x_f * x_f).sum(dim=1, keepdim=True)          # (N, 1)
            w2 = (w_f * w_f).sum(dim=1).unsqueeze(0)           # (1, K)
            dist = x2 + w2 - 2.0 * (x_f @ w_f.t())             # (N, K)

            indices = dist.argmin(dim=1)                        # (N,)

        q = self.vq_codebook(indices)  # (N, D), grad -> vq_codebook.weight
        q = q.view(*og_shape[:-1], d) # Reshape back to match input

        if q.dtype != continuous_vec.dtype:
            q = q.to(dtype=continuous_vec.dtype)

        return q