import os
import sys
current_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#utilities_path = os.path.join(current_path, 'utilities')
sys.path.append(current_path)
import torch

import argparse
import json
import pickle
from tqdm import tqdm
from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.inference import video_chatgpt_infer


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', help='Directory containing video files.', default=None)
    parser.add_argument('--video_feature_dir', help='Directory containing video feature files.', default=None)
    parser.add_argument('--gt_file', help='Path to the ground truth file.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--projection_path", type=str, required=False, default=None)

    return parser.parse_args()


def run_inference(args):
    """
    Run inference on a set of video files using the provided model.

    Args:
        args: Command-line arguments.
    """
    assert not (args.video_feature_dir is None and args.video_dir is None), "video_feature_dir and video_dir cannot be None at the same time!"

    # Initialize the model
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name,
                                                                                        args.projection_path)
    # Load the ground truth file
    with open(args.gt_file) as file:
        gt_contents = json.load(file)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results
    conv_mode = args.conv_mode

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    for sample in tqdm(gt_contents):
        video_name = sample['video_name']
        sample_set = sample
        question = sample['Q']
        video_frames = None
        video_features = None
        if args.video_feature_dir is None:
                # Load the video file
            for fmt in video_formats:  # Added this line
                temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
                if os.path.exists(temp_path):
                    video_path = temp_path
                    break

            # Check if the video exists
            if video_path is not None:  # Modified this line
                video_frames = load_video(video_path)
        else:    
            temp_path = os.path.join(args.video_feature_dir, f"{video_name}.pkl")
            if os.path.exists(temp_path):
                video_path = temp_path
            if video_path is not None:  # Modified this line
                with open(f"{video_path}", "rb") as f:
                    video_features = torch.tensor(pickle.load(f))
                


        try:
            # Run inference on the video and add the output to the list
            output = video_chatgpt_infer(video_frames, video_features, question, conv_mode, model, vision_tower,
                                             tokenizer, image_processor, video_token_len)
            sample_set['pred'] = output
            output_list.append(sample_set)
        except Exception as e:
            print(f"Error processing video file '{video_name}': {e}")

    # Save the output list to a JSON file
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
        json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
