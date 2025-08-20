from videoxl.videoxl.model.builder import load_pretrained_model
from videoxl.videoxl.mm_utils import tokenizer_image_token, process_images,transform_input_id
from videoxl.videoxl.constants import IMAGE_TOKEN_INDEX,TOKEN_PERFRAME 
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
# fix seed
torch.manual_seed(0)


model_path = "/data2/josephyumss/Video-XL/VideoXL_weight_8/VideoXL_weight_8"
video_path="/data2/josephyumss/Video-XL/Video-XL/assets/Download.mp4"

max_frames_num =100 # you can change this to several thousands so long you GPU memory can handle it :)
gen_kwargs = {"do_sample": True, "temperature": 1, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 512}
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="cuda:1",attn_implementation="sdpa")

model.config.beacon_ratio=[8]   # you can delete this line to realize random compression of {2,4,8} ratio

# device 
device = torch.device("cuda:1")
model.to(device)

m = model.get_model() if hasattr(model, "get_model") else model
vt = model.get_vision_tower() if hasattr(model, "get_vision_tower") else None

if vt is not None:
    try:
        vt.to(device)
    except :
        pass
    if hasattr(vt, "vision_tower") and vt.vision_tower is not None:
        vt.vision_tower.to(device)

if hasattr(m, "mm_projector") and m.mm_projector is not None:
    m.mm_projector.to(device)

if hasattr(m, "vision_resampler") and m.vision_resampler is not None:
    m.vision_resampler.to(device)


#video input
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nDoes this video contain any inserted advertisement? If yes, which is the content of the ad?<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
vr = VideoReader(video_path, ctx=cpu(0))
total_frame_num = len(vr)
uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
frame_idx = uniform_sampled_frames.tolist()
frames = vr.get_batch(frame_idx).asnumpy()
video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(device, dtype=torch.float16)

beacon_skip_first = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[1].item()
num_tokens=TOKEN_PERFRAME *max_frames_num
beacon_skip_last = beacon_skip_first  + num_tokens

with torch.inference_mode():
    output_ids = model.generate(input_ids, images=[video_tensor],  modalities=["video"],beacon_skip_first=beacon_skip_first,beacon_skip_last=beacon_skip_last, **gen_kwargs)

if IMAGE_TOKEN_INDEX in input_ids:
    transform_input_ids=transform_input_id(input_ids,num_tokens,model.config.vocab_size-1)

output_ids=output_ids[:,transform_input_ids.shape[1]:]
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(outputs)