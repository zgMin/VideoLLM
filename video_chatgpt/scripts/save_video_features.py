
import torch
from tqdm import tqdm
import torchvision.transforms as T
from decord import VideoReader, cpu
import numpy as np
import argparse
import os
import pickle

from video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

import contextlib
import torch.nn as nn
from video_chatgpt.model.vit import build_vit

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets



def load_video(video_path, num_segments=8, return_msg=False, resolution=224):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    # transform
    crop_size = resolution
    scale_size = resolution
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return torch_imgs, msg
    else:
        return torch_imgs
    
def get_sinusoid_encoding_table(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    
    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    
    print(f"n_position: {n_position}")
    print(f"pre_n_position: {pre_n_position}")
    
    if n_position != pre_n_position:
        T = ckpt_num_frame # checkpoint frame
        P = 14 # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5) # testing size
        if new_P != 14:
            print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
            print(f'Interpolate the position embedding')
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            sinusoid_table = torch.nn.functional.interpolate(
                sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
    
    if cur_frame != ckpt_num_frame:
        print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
        print(f'Interpolate the position embedding')
        T = ckpt_num_frame # checkpoint frame
        new_T = cur_frame # testing frame
        # interpolate
        P = int((n_position // cur_frame) ** 0.5) # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
        
    return sinusoid_table

def init_vision_encoder_umt(config):
    """build vision encoder
    Returns: (vision_encoder, vision_layernorm). Each is a `nn.Module`.

    """
    vision_encoder = build_vit(config)

    if config.vision_encoder.vit_add_ln:
        vision_layernorm = nn.LayerNorm(config.vision_encoder.encoder_embed_dim, eps=1e-12)
    else:
        vision_layernorm = nn.Identity()

    return vision_encoder, vision_layernorm


def vit_to_cpu(vision_encoder,vision_layernorm):
    vision_layernorm.to("cpu")
    vision_layernorm.float()
    vision_encoder.to("cpu")
    vision_encoder.float()

def maybe_autocast(device, dtype=torch.float16):
    # if on cpu, don't use autocast
    # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
    enable_autocast = device != torch.device("cpu")

    if enable_autocast:
        return torch.cuda.amp.autocast(dtype=dtype)
    else:
        return contextlib.nullcontext()


def encode_img(vision_encoder, vision_layernorm, image,low_resource = False):
    device = image.device
    if low_resource:
        vit_to_cpu(vision_encoder, vision_layernorm)
        image = image.to("cpu")

    with maybe_autocast(device):
        #print(device)
        T = image.shape[1]
        use_image = True if T == 1 else False
        image = image.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]

        image_embeds = vision_encoder(image, use_image)
        B, T, L, C = image_embeds.shape
        image_embeds = image_embeds.reshape(B, -1, C)
        image_embeds = vision_layernorm(image_embeds).to(device)  # [B, T*L, C]
    return image_embeds, use_image




def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--video_dir_path", required=True, help="Path to read the videos from.")
    parser.add_argument("--clip_feat_path", required=True, help="The output dir to save the features in.")
    parser.add_argument("--infer_batch", required=False, type=int, default=32,
                        help="Number of frames/images to perform batch inference.")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    video_dir_path = args.video_dir_path
    clip_feat_path = args.clip_feat_path
    infer_batch = args.infer_batch
    os.makedirs(clip_feat_path, exist_ok=True)

    # init model
    # num_frame = 8
    num_frame = 16
    # resolution = 384
    resolution = 224
    # TODO: set config
    config = {
        'vision_encoder':
            {
            'img_size': 224,
            'patch_size': 16,
            'encoder_embed_dim': 1024, 
            'encoder_depth': 12,
            'encoder_num_heads': 12, 
            'drop_path_rate': 0.1, 
            'num_frames': num_frame,
            'tubelet_size': 1,
            'use_checkpoint': False,
            'checkpoint_num': 0,
            'pretrained': 'l16_25m.pth',
            'ckpt_num_frame': 8,
            'return_index': -1,
            'with_ln': False,
            'vit_add_ln':True,
        }
    }
    from easydict import EasyDict
    config = EasyDict(config)
    
    vision_encoder, vision_layernorm = init_vision_encoder_umt(config)
    vision_encoder.eval()
    new_pos_emb = get_sinusoid_encoding_table(n_position=(resolution//16)**2*num_frame, cur_frame=num_frame)
    vision_encoder.encoder.pos_embed = new_pos_emb
    vision_encoder.to('cuda:0')
    vision_layernorm.to('cuda:0')
    all_videos = os.listdir(video_dir_path)
    
    video_clip_features = {}
    counter = 0
    for video_name in tqdm(all_videos):
        video_path = f"{video_dir_path}/{video_name}"
        video_id = video_name.split('.')[0]
        if os.path.exists(f"{clip_feat_path}/{video_id}.pkl"):  # Check if the file is already processed
            continue
        vid,msg = load_video(video_path, num_segments=num_frame, return_msg=True, resolution=resolution)
        #print(msg)
        # The model expects inputs of shape: T x C x H x W
        TC, H, W = vid.shape
        video = vid.reshape(1, TC//3, 3, H, W).to("cuda:0")
        #print(video.shape)
        with torch.no_grad():
            image_emb, _ = encode_img(vision_encoder, vision_layernorm, video)
        #print(image_emb.shape)
        video_clip_features[video_id] = image_emb.squeeze(0).cpu().numpy().astype("float16")
        counter += 1
        # try:
        #     vid,msg = load_video(video_path, num_segments=num_frame, return_msg=True, resolution=resolution)
        #     print(msg)
        #     # The model expects inputs of shape: T x C x H x W
        #     TC, H, W = vid.shape
        #     video = vid.reshape(1, TC//3, 3, H, W).to("cuda:0")
        #     print(video.shape)
        #     with torch.no_grad():
        #         image_emb, _ = encode_img(vision_encoder, vision_layernorm, video)
        #     print(image_emb.shape)
        #     video_clip_features[video_id] = image_emb.squeeze(0).cpu().numpy().astype("float16")
        #     counter += 1
        # except Exception as e:
        #     print(f"Can't process {video_path}")

        if counter % 512 == 0:  # Save after every 512 videos, update this number as per your requirements
            for key in video_clip_features.keys():
                features = video_clip_features[key]
                with open(f"{clip_feat_path}/{key}.pkl", 'wb') as f:
                    pickle.dump(features, f)
            video_clip_features = {}

    for key in video_clip_features.keys():
        features = video_clip_features[key]
        with open(f"{clip_feat_path}/{key}.pkl", 'wb') as f:
            pickle.dump(features, f)


if __name__ == "__main__":
    main()