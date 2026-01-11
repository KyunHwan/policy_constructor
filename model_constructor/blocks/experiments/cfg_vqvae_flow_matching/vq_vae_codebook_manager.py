import torch
import torch.nn as nn
import einops

import torch.nn.functional as F
import torch.distributed as distributed

class VQCodebookManager(nn.Module):
    def __init__(self, num_q_vectors, vec_dim):
        super().__init__()
        self.num_q_vectors = num_q_vectors
        self.vec_dim = vec_dim

        self.vq_codebook = nn.Embedding(num_q_vectors, vec_dim)

        # Optional: Initialize weights to be uniform for better initial convergence
        with torch.no_grad():
            self.vq_codebook.weight.zero_()
            #self.vq_codebook.weight.data.uniform_(-1.0 / num_q_vectors, 1.0 / num_q_vectors)


    def forward(self, continuous_vec: torch.Tensor, train: bool, replacement: bool):
        """
        Parameters: 
            continuous_vec: (batch, features) or (batch, num_vec, features) shape
        
        Returns:
            dictionary of
                q_vec: same shape as continuous_vec, where each vector is
                    replaced by its nearest codebook embedding (L2 distance).
        """
        q = None
        if not replacement:
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
        max_dist = 0.0
        num_replaced = 0
        dead_indices = None
        with torch.no_grad():
            if train:
                dists = torch.cdist(self.vq_codebook.weight, self.vq_codebook.weight, p=2)
                max_dist = dists.max().item()
                dists.fill_diagonal_(float('inf'))
                min_dist = dists.min().item()
                if replacement:
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
                    """
                    Replaces codebook vectors that have usage < threshold.
                    DDP Safe: Ensures all ranks agree on the new vectors.
                    """
                    # 1. Count usage in the local batch
                    # indices shape: (Batch * Sequence_Length)
                    counts = torch.bincount(indices, minlength=self.num_q_vectors)

                    # 2. Synchronize counts across all GPUs (DDP)
                    if distributed.is_initialized():
                        distributed.all_reduce(counts, op=distributed.ReduceOp.SUM)

                    # should come after the distributed sync
                    threshold = max(int(torch.sum(counts) / self.num_q_vectors), 1)

                    # 3. Identify dead indices
                    # dead_mask = counts < threshold
                    dead_indices = torch.nonzero(counts < threshold).squeeze(-1)
                    live_indices = torch.nonzero(counts >= threshold).squeeze(-1)
                    num_dead = dead_indices.numel() # number of dead 

                    # Only proceed if there are dead vectors and at least one live vector to copy from
                    if num_dead > 0 and live_indices.numel() > 0:
                        num_replaced = num_dead
                        # 4. Select replacement candidates
                        # We create a container for replacements on the correct device
                        replacements = torch.empty(
                            (num_dead, self.vec_dim), 
                            device=self.vq_codebook.weight.device, 
                            dtype=self.vq_codebook.weight.dtype
                        )

                        # To ensure all DDP ranks stay in sync, Rank 0 generates the replacements
                        # and broadcasts them to everyone else.
                        if not distributed.is_initialized() or distributed.get_rank() == 0:
                            # A. Get the counts of the live vectors to use as weights
                            live_counts = counts[live_indices]
                            
                            # B. Convert counts to probabilities (high usage = high chance of being split)
                            # Using float() for division safety
                            probs = live_counts.float() / live_counts.sum()
                            
                            # C. Sample indices from the live set based on these probabilities
                            # We need 'num_dead' source vectors. 'replacement=True' allows picking the same popular vector multiple times.
                            source_indices_idx = torch.multinomial(probs, num_dead, replacement=True)
                            
                            # Map back to the actual codebook indices
                            source_indices = live_indices[source_indices_idx]
                            
                            # D. Fetch the actual high-usage vectors
                            best_vectors = self.vq_codebook.weight.data[source_indices]
                            
                            # E. PERTURBATION (Very Important!)
                            # We cannot just copy the vectors; we must add noise so the new vector 
                            # can diverge from the original one during training.
                            # 0.02 is a common perturbation scale, but this depends on your data norm.
                            noise_scale = 0.02 
                            
                            # If your codebook vectors are normalized (L2), use orthogonal noise or re-normalize after.
                            # Here we assume standard un-normalized VQ:
                            noise = torch.randn_like(best_vectors) * noise_scale
                            replacements = best_vectors + noise

                        # 5. Broadcast replacements to all ranks
                        if distributed.is_initialized():
                            # Broadcast the calculated replacements from Rank 0 to all other ranks
                            distributed.broadcast(replacements, src=0)
                        
                        # 6. Perform the update
                        self.vq_codebook.weight.data[dead_indices] = replacements
                    
                        # Optional: Reset the optimizer state for these indices would happen outside this class
                        # But updating the weight in-place is the critical first step.


        return {
            'q': q,
            'codebook_min_dist': min_dist,
            'codebook_max_dist': max_dist,
            'num_vecs_replaced': num_replaced,
            'dead_indices': dead_indices
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
