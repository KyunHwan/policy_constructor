import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

class CosmosReason2Encoder(torch.nn.Module):
    """
    Produces multimodal token features you can feed to a policy via cross-attn.

    Returns:
      mem_tokens: [B, S, D]  (combined text+vision sequence)
      mem_mask:   [B, S]     (True for padding if you create padding; often all False)
      img_mask:   [B, S]     (True where tokens are image tokens)
      meta: dict with useful bookkeeping (image_grid_thw, etc.)
    """
    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        last_k_layers: int = 4,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        device_map: str = "auto",
    ):
        super().__init__()
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.last_k = last_k_layers
        self.image_token_id = self.model.config.image_token_id
        self.video_token_id = getattr(self.model.config, "video_token_id", None)

    @torch.inference_mode()
    def forward(
        self,
        image,               # PIL.Image, numpy array, or a path/URL depending on your setup
        question: str,
        system_prompt: str = "You are a robot manipulation reasoning assistant.",
    ):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]},
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Some templates add token_type_ids; many models ignore it
        inputs.pop("token_type_ids", None)

        # Move tensors to model device
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(self.model.device)

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        # hidden_states is a tuple: (embeds_out, layer1_out, ..., layerN_out)
        hs = outputs.hidden_states
        # Combine last K layers (mean is a strong default; concat is also common)
        mem_tokens = torch.stack(hs[-self.last_k:], dim=0).mean(dim=0)  # [B, S, D]

        input_ids = inputs["input_ids"]  # [B, S]
        img_mask = (input_ids == self.image_token_id)

        # If you pad batches, build mem_mask accordingly; otherwise all False.
        # Here we assume no padding:
        mem_mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)

        meta = {}
        if "image_grid_thw" in inputs:
            meta["image_grid_thw"] = inputs["image_grid_thw"]
        if "attention_mask" in inputs:
            meta["attention_mask"] = inputs["attention_mask"]

        return mem_tokens, mem_mask, img_mask, meta