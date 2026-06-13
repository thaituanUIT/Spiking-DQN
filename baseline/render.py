import argparse
import torch
import os
import sys
import cv2
import numpy as np

# Ensure imports work by adding the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.utils.agent import Agent
from v2.helpers.renderer import render_predictions
from v2.data.factory import build_eval_dataset, get_default_weight_path

def main():
    parser = argparse.ArgumentParser(description="Baseline Agent Rendering with v2 Interface")
    
    # Core Parameters
    core_group = parser.add_argument_group('Core Parameters')
    core_group.add_argument('--target', type=str, default='mixing', help="Target class or 'mixing' for all")
    core_group.add_argument('--dataset', type=str, choices=['voc', 'tiny-imagenet'], default='voc', help="Dataset to render from when --image-path is not provided")
    core_group.add_argument('--image-path', type=str, default=None, help="Path to specific image file")
    core_group.add_argument('--num-images', type=int, default=5, help="Number of images if no path provided")
    core_group.add_argument('--extractor', type=str, choices=['vgg16', 'resnet18', 'vit', 'efficientnet', 'mobilenet'], default='vgg16', help="Feature extractor backbone")
    
    # Agent Parameters
    agent_group = parser.add_argument_group('Agent Parameters')
    agent_group.add_argument('--max-steps', type=int, default=20, help="Max steps per image")
    agent_group.add_argument('--alpha', type=float, default=0.1, help="Mask transformation rate")
    agent_group.add_argument('--nu', type=float, default=3.0, help="Trigger reward weight")
    agent_group.add_argument('--threshold', type=float, default=0.5, help="IoU threshold for trigger reward")
    agent_group.add_argument('--replay-device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto', help="Where replay/cache tensors live")
    
    # System Parameters
    sys_group = parser.add_argument_group('System Parameters')
    sys_group.add_argument('--weights', type=str, default=None, help="Path to specific weights file")
    sys_group.add_argument('--save', action='store_true', help="Save rendered images to disk")
    sys_group.add_argument('--save-dir', type=str, default=None, help="Directory to save rendered images")
    
    args = parser.parse_args()
    
    if args.save and not args.save_dir:
        args.save_dir = f"baseline/renders/baseline_{args.target}"
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.image_path:
        if not os.path.exists(args.image_path):
            print(f"Error: Image path {args.image_path} not found.")
            return
        img_bgr = cv2.imread(args.image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        samples = [{
            'image': img_rgb,
            'box': None,
            'filename': os.path.basename(args.image_path)
        }]
    else:
        dataset = build_eval_dataset(
            dataset_name=args.dataset,
            target_class=args.target,
            num_samples=args.num_images,
        )
        samples = [dataset[i] for i in range(len(dataset))]
        
    agent = Agent(
        classe=args.target,
        alpha=args.alpha,
        nu=args.nu,
        threshold=args.threshold,
        max_steps=args.max_steps,
        device=device,
        extractor_name=args.extractor,
        replay_device=args.replay_device,
    )
    
    # Load weights
    weight_prefix = f"baseline_{args.extractor}"
    weight_path = args.weights if args.weights else get_default_weight_path(weight_prefix, args.dataset, args.target, "baseline/weights")
    if os.path.exists(weight_path):
        agent.model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"Loaded weights from {weight_path}")
    else:
        status = "Error" if args.weights else "Warning"
        print(f"{status}: Weights not found at {weight_path}. Cannot render without trained weights.")
        return
        
    render_predictions(agent, samples, save_dir=args.save_dir)

if __name__ == '__main__':
    main()
