import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization for stability in dynamic learning rules.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class HebbianMLP(nn.Module):
    """
    Hebbian Fast-Weight MLP Layer with Contrastive Gating and Sparsification.
    
    This layer combines static (learned) weights with dynamic (plastic) fast-weights.
    It incorporates a Novelty/Contrastive Filter to bypass updates on familiar
    or low-significance tokens, solving memory saturation (noise) and CPU lag.
    """
    def __init__(
        self,
        d_in,
        d_out,
        learning_rule="oja_multidimensional",
        eta=0.01,
        decay=0.001,
        max_norm=1.0,
        use_frobenius_normalization=True,
        novelty_threshold=0.3,       # Min novelty required to trigger an update (0.0 to 1.0)
        sparsity_update_rate=0.05    # Stochastic update rate for active distractors to reduce lag
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.learning_rule = learning_rule
        self.eta = eta
        self.decay = decay
        self.max_norm = max_norm
        self.use_frobenius_normalization = use_frobenius_normalization
        self.novelty_threshold = novelty_threshold
        self.sparsity_update_rate = sparsity_update_rate
        
        # Static weights initialized as a standard linear layer
        self.W_static = nn.Parameter(torch.randn(d_out, d_in) * (2.0 / (d_in + d_out)) ** 0.5)
        self.bias_static = nn.Parameter(torch.zeros(d_out))
        
        # Input and output normalization layers to guarantee dynamic range stability
        self.in_norm = RMSNorm(d_in)
        self.out_norm = RMSNorm(d_out)
        
        # Plastic fast-weights, initialized per-batch in the forward pass
        self.W_plastic = None

    def reset_memory(self, batch_size, device):
        """
        Resets the dynamic fast-weights to zero at the start of a sequence.
        """
        self.W_plastic = torch.zeros(batch_size, self.d_out, self.d_in, device=device)

    def forward(self, x, update_gate=None, y_target=None):
        """
        Forward pass with Contrastive Gating and Sparsified dynamic updates.
        """
        B, d_in = x.shape
        device = x.device
        
        # Auto-initialize plastic weights if not set
        if self.W_plastic is None or self.W_plastic.shape[0] != B or self.W_plastic.device != device:
            self.reset_memory(B, device)
            
        # 1. Normalize the inputs for stable dynamics
        x_norm = self.in_norm(x)  # [B, d_in]
        
        # 2. Compute static projection
        y_static = F.linear(x_norm, self.W_static, self.bias_static)  # [B, d_out]
        
        # 3. Compute dynamic plastic projection
        x_col = x_norm.unsqueeze(-1)  # [B, d_in, 1]
        y_plastic = torch.bmm(self.W_plastic, x_col).squeeze(-1)  # [B, d_out]
        
        y_raw = y_static + y_plastic
        y = self.out_norm(y_raw)  # [B, d_out]
        
        # 4. Contrastive Gating and Sparsified dynamic update
        if update_gate is not None:
            # Fast scalar or vector checking to completely bypass updates if gate is zero
            if isinstance(update_gate, torch.Tensor):
                max_gate = torch.max(update_gate).item()
            else:
                max_gate = float(update_gate)
                
            if max_gate > 0:
                # If it's a weak background update (e.g. distractor phase where gate is small),
                # apply Sparsified stochastic updates and Contrastive Novelty Gating to reduce lag and memory saturation.
                is_distractor_phase = (max_gate < 0.5)
                
                # A: Stochastic Sparsification Bypass
                if is_distractor_phase:
                    # Only update on a small fraction of distractor steps to save CPU/GPU cycles
                    if torch.rand(1).item() > self.sparsity_update_rate:
                        return y  # COMPASSIONATE BYPASS: Skip entire update block
                
                # B: Contrastive Novelty Filtering
                # If there's no explicit target (unsupervised/self-association), check if the input is novel.
                # If the plastic projection is already highly aligned with the output, the memory is already familiar.
                if y_target is None and is_distractor_phase:
                    # Compute cosine similarity between static activation and full plastic activation
                    y_static_norm = F.normalize(y_static, p=2, dim=-1)
                    y_norm_c = F.normalize(y, p=2, dim=-1)
                    cos_sim = torch.mean(torch.sum(y_static_norm * y_norm_c, dim=-1)).item()
                    novelty = 1.0 - abs(cos_sim)
                    
                    # If novelty is below threshold, skip the update to prevent saturation & save CPU
                    if novelty < self.novelty_threshold:
                        return y  # NOVELTY BYPASS: Skip update
                
                # Perform the Hebbian weight update
                if isinstance(update_gate, torch.Tensor):
                    gate = update_gate.view(B, 1, 1).to(device)
                else:
                    gate = torch.tensor(update_gate, device=device).view(1, 1, 1)
                
                y_update = y_target if y_target is not None else y
                y_update_norm = self.out_norm(y_update)
                
                y_col = y_update_norm.unsqueeze(-1)  # [B, d_out, 1]
                x_row = x_norm.unsqueeze(1)         # [B, 1, d_in]
                hebb_term = torch.bmm(y_col, x_row)  # [B, d_out, d_in]
                
                if self.learning_rule == "oja_multidimensional":
                    y_outer = torch.bmm(y_col, y_col.transpose(-1, -2))
                    decay_term = torch.bmm(y_outer, self.W_plastic)
                    dW = self.eta * (hebb_term - decay_term)
                    
                elif self.learning_rule == "oja_scalar":
                    y_norm_sq = torch.sum(y_update_norm ** 2, dim=-1, keepdim=True).unsqueeze(-1)
                    decay_term = y_norm_sq * self.W_plastic
                    dW = self.eta * (hebb_term - decay_term)
                    
                elif self.learning_rule == "gated_decay":
                    dW = -self.decay * self.W_plastic + self.eta * hebb_term
                    
                else:
                    raise ValueError(f"Unknown learning rule: {self.learning_rule}")
                
                self.W_plastic = self.W_plastic + gate * dW
                
                # Apply Frobenius norm containment
                if self.use_frobenius_normalization:
                    frob_norm = torch.norm(self.W_plastic, p="fro", dim=(-1, -2), keepdim=True)
                    scale = torch.clamp(self.max_norm / (frob_norm + 1e-8), max=1.0)
                    self.W_plastic = self.W_plastic * scale
                    
        return y


class ToyHebbianModel(nn.Module):
    """
    A simple sequence model wrapping an Embedding layer, a HebbianMLP, and a Projection Head.
    """
    def __init__(
        self,
        vocab_size,
        d_model,
        learning_rule="oja_multidimensional",
        eta=0.01,
        decay=0.001,
        max_norm=1.0,
        novelty_threshold=0.3,
        sparsity_update_rate=0.05
    ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.hebbian_mlp = HebbianMLP(
            d_in=d_model,
            d_out=d_model,
            learning_rule=learning_rule,
            eta=eta,
            decay=decay,
            max_norm=max_norm,
            novelty_threshold=novelty_threshold,
            sparsity_update_rate=sparsity_update_rate
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embeddings.weight

    def reset_memory(self, batch_size, device):
        self.hebbian_mlp.reset_memory(batch_size, device)

    def forward(self, token_seq, update_gate_seq=None, target_token_seq=None):
        B, T = token_seq.shape
        device = token_seq.device
        
        self.reset_memory(B, device)
        logits_list = []
        
        for t in range(T):
            x_token = token_seq[:, t]
            embeddings_t = self.embeddings(x_token)
            
            gate_t = None
            if update_gate_seq is not None:
                gate_t = update_gate_seq[:, t]
                
            y_target_t = None
            if target_token_seq is not None:
                target_token_t = target_token_seq[:, t]
                y_target_t = self.embeddings(target_token_t)
                
            y_t = self.hebbian_mlp(embeddings_t, gate_t, y_target=y_target_t)
            logits_t = self.head(y_t)
            logits_list.append(logits_t.unsqueeze(1))
            
        return torch.cat(logits_list, dim=1)
