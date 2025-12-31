import torch
import torch.nn as nn
import einops

class VQCodebookManager(nn.Module):
    def __init__(self, num_q_vectors, vec_dim):
        super().__init__()
        self.vq_codebook = nn.Embedding(num_q_vectors, vec_dim)

        # Optional: Initialize weights to be uniform for better initial convergence
        self.vq_codebook.weight.data.uniform_(-1.0 / num_q_vectors, 1.0 / num_q_vectors)
    
    def forward(self, continuous_vec: torch.Tensor, train: bool=True):
        """
        Parameters: 
            continuous_vec: (batch, features) or (batch, num_vec, features) shape
        
        Returns:
            dictionary of
                q_vec: same shape as continuous_vec, where each vector is
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
        
        # 2. Capture original shape and Flatten
        #    Input: (B, T, D) or (B, D) -> Flatten to (B*T, D) or (B, D)
        original_shape = continuous_vec.shape
        x_flat = continuous_vec.reshape(-1, d)

        # 3. Calculate Distances (Batch processing)
        with torch.no_grad():
            x_f = x_flat.float()
            w_f = self.vq_codebook.weight.float()

            # L2 Distance: ||x - w||^2 = ||x||^2 + ||w||^2 - 2xw
            # Shapes: x_f (N, D), w_f (K, D)
            x2 = (x_f * x_f).sum(dim=1, keepdim=True)       # (N, 1)
            w2 = (w_f * w_f).sum(dim=1).unsqueeze(0)        # (1, K)
            
            # (N, 1) + (1, K) - (N, K) -> (N, K)
            dist = x2 + w2 - 2.0 * (x_f @ w_f.t())
            
            # Get indices of nearest neighbors
            indices = dist.argmin(dim=1)                    # (N,)

        # 4. Quantize
        #    Look up the codebook vectors.
        #    Note: Gradients flow from Loss -> q -> vq_codebook.weight
        q_flat = self.vq_codebook(indices)                  # (N, D)

        # 5. Reshape back to original input shape
        #    (N, D) -> (Batch, Num_Vec, Features)
        q = q_flat.view(original_shape)

        # 6. Ensure dtype matches input
        if q.dtype != continuous_vec.dtype:
            q = q.to(dtype=continuous_vec.dtype)
        
        min_dist = 0.0
        with torch.no_grad():
            if train:
                dists = torch.cdist(self.vq_codebook.weight, self.vq_codebook.weight, p=2)
                dists.fill_diagonal_(float('inf'))
                min_dist = dists.min().item()
        

        return {
            'q': q,
            'codebook_min_dist': min_dist
        }

    def get_min_pairwise_dist(self):
        """
        Returns:
            Minimum distance between distinct vectors in the codebook.
        """
        min_dist = 0.0
        with torch.no_grad():
            dists = torch.cdist(self.vq_codebook.weight, self.vq_codebook.weight, p=2)

            dists.fill_diagonal_(float('inf'))

            min_dist = dists.min().item() # .item() moves the value to CPU

        return min_dist
