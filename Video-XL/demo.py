from videoxl.videoxl.model.builder import load_pretrained_model
from videoxl.videoxl.mm_utils import tokenizer_image_token, process_images,transform_input_id
from videoxl.videoxl.constants import IMAGE_TOKEN_INDEX,TOKEN_PERFRAME 
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
import math

# fix seed
#autograd.set_detect_anomaly(True)
torch.manual_seed(0)

model_path = "/data2/josephyumss/Video-XL/VideoXL_weight_8/VideoXL_weight_8"
video_path="/data2/josephyumss/Video-XL/Video-XL/assets/Download.mp4"

max_frames_num =10 # you can change this to several thousands so long you GPU memory can handle it :)
gen_kwargs = {"do_sample": True, "temperature": 1, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 256}
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="cuda:1",attn_implementation="sdpa")

model.config.beacon_ratio=[16]   # you can delete this line to realize random compression of {2,4,8} ratio

# device 
device = torch.device("cuda:1")

device_ids = [0, 1]

# batch 1이어서 의미 없다. DataParallel은 배치만 분산해줌
#model = torch.nn.DataParallel(model, device_ids=device_ids)

#video input
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nDoes this video contain any inserted advertisement? If yes, which is the content of the ad?<|im_end|>\n<|im_start|>assistant\n"
# IMAGE_TOKEN_INDEX는 -200으로 정의되어 있음
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

# VideoReader는 영상을 frame 단위로 읽어오는 class인 듯
# ctx는 decord device 지정 변수. cpu로 설정됨
vr = VideoReader(video_path, ctx=cpu(0)) 
total_frame_num = len(vr)
uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int) # input video에서 max frame만큼 sampling 하여 사용
frame_idx = uniform_sampled_frames.tolist() # sampling된 frame list
frames = vr.get_batch(frame_idx).asnumpy()
print(f"[demo.py] frame batch     : {frames.shape}")
video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(device, dtype=torch.float16)
print("[demo.py] Image processor frame to tensor")
print(f"[demo.py] tensor shape    : {video_tensor.shape}")

beacon_skip_first = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[1].item()
num_tokens=TOKEN_PERFRAME *max_frames_num
beacon_skip_last = beacon_skip_first  + num_tokens

with torch.inference_mode(): # torch.no_grad() 와 유사한 작동. 
    # 출력은 model의 generate 함수로 생성
    output_ids = model.generate(input_ids, images=[video_tensor],  modalities=["video"],beacon_skip_first=beacon_skip_first,beacon_skip_last=beacon_skip_last, **gen_kwargs)

if IMAGE_TOKEN_INDEX in input_ids:
    transform_input_ids=transform_input_id(input_ids,num_tokens,model.config.vocab_size-1)

output_ids=output_ids[:,transform_input_ids.shape[1]:]
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(outputs)