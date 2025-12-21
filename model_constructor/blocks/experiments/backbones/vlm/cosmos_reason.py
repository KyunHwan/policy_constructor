"""
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

model_id = "nvidia/Cosmos-Reason1-7B"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

messages = [{
  "role": "user",
  "content": [
    {"type": "video", "video": "video.mp4", "fps": 4},
    {"type": "text", "text": "Summarize the scene for action planning."}
  ]
}]

prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[prompt],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    out = model(**inputs, output_hidden_states=True, return_dict=True)

# out.hidden_states is a tuple: (embeddings, layer1, ..., layerN)
last_h = out.hidden_states[-1]          # [B, seq, hidden]
prompt_h = last_h[:, :inputs["input_ids"].shape[1], :]
conditioning_vec = prompt_h.mean(dim=1) # [B, hidden] fixed vector
"""
