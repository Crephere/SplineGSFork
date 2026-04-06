#!/usr/bin/env python3
"""
Test/inference script for train_aligned.py trained model.
Renders video from trained aligned model to visually verify mask alignment improvement.

Usage:
    python test_aligned_inference.py --checkpoint ./output/balloon1_aligned/point_cloud/iteration_5000 \
        --expname test_aligned_output --dataset_type colmap
"""
import os
import sys
import torch
import cv2
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image

sys.path.append(os.path.join(sys.path[0], ".."))

from arguments import ModelHiddenParams, ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render_infer
from scene import GaussianModel, Scene, dataset_readers
from utils.main_utils import get_pixels


def render_evaluation(scene: Scene, cameras, dyn_gaussians, stat_gaussians, 
                      background, output_path, video_output=None):
    """
    Render all cameras and optionally create video.
    
    Args:
        scene: Scene instance with camera/dataset info
        cameras: List of camera viewpoints to render
        dyn_gaussians: Dynamic Gaussian model
        stat_gaussians: Static Gaussian model
        background: Background color
        output_path: Path to save rendered images
        video_output: Path to save video (None to skip video creation)
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Get pixels and viewdirs for pose network
    pixels = get_pixels(
        scene.train_camera.dataset[0].metadata.image_size_x,
        scene.train_camera.dataset[0].metadata.image_size_y,
        use_center=True,
    )
    
    if pixels.shape[-1] != 2:
        raise ValueError("The last dimension of pixels must be 2.")
    
    batch_shape = pixels.shape[:-1]
    pixels = np.reshape(pixels, (-1, 2))
    
    y = (
        pixels[..., 1] - scene.train_camera.dataset[0].metadata.principal_point_y
    ) / dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy()
    x = (
        pixels[..., 0] - scene.train_camera.dataset[0].metadata.principal_point_x
    ) / dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy()
    
    viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    local_viewdirs = viewdirs / np.linalg.norm(viewdirs, axis=-1, keepdims=True)
    
    # Predict camera poses for all frames
    print(f"[INFO] Predicting poses for {len(cameras)} frames...")
    with torch.no_grad():
        for cam in cameras:
            time_in = torch.tensor(cam.time).float().cuda()
            pred_R, pred_T = dyn_gaussians._posenet(time_in.view(1, 1))
            R_ = torch.transpose(pred_R, 2, 1).detach().cpu().numpy()
            t_ = pred_T.detach().cpu().numpy()
            cam.update_cam(
                R_[0],
                t_[0],
                local_viewdirs,
                batch_shape,
                dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy(),
            )
    
    # Render all frames
    print(f"[INFO] Rendering {len(cameras)} frames...")
    rendered_frames = []
    
    with torch.no_grad():
        for idx, viewpoint in enumerate(cameras):
            if (idx + 1) % 10 == 0:
                print(f"[INFO] Rendered {idx + 1}/{len(cameras)} frames")
            
            render_pkg = render_infer(
                viewpoint, stat_gaussians, dyn_gaussians, background
            )
            
            image = render_pkg["render"]
            image = torch.clamp(image, 0.0, 1.0)
            
            # Save individual frame as PNG
            img_np = (np.clip(image.permute(1, 2, 0).detach().cpu().numpy(), 0, 1) * 255).astype("uint8")
            img_pil = Image.fromarray(img_np)
            img_pil.save(os.path.join(output_path, f"frame_{idx:04d}.png"))
            
            rendered_frames.append(img_np)
    
    # Create video if output path specified
    if video_output and rendered_frames:
        print(f"[INFO] Creating video: {video_output}")
        h, w = rendered_frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_output, fourcc, 24.0, (w, h))
        
        for frame in rendered_frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"[INFO] Video saved: {video_output}")
    
    print(f"[INFO] Rendering complete. Frames saved to: {output_path}")
    return rendered_frames


if __name__ == "__main__":
    parser = ArgumentParser(description="Inference script for aligned training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)

    parser.add_argument(
        "--checkpoint", type=str, required=True, 
        help="Path to checkpoint directory (e.g., output/balloon1_aligned/point_cloud/iteration_5000)"
    )
    parser.add_argument("--expname", type=str, default="test_aligned_output")
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--render_test", action="store_true", help="Render test set")
    parser.add_argument("--render_train", action="store_true", help="Render train set")
    parser.add_argument("--video_fps", type=int, default=24, help="Video frame rate")

    args = parser.parse_args(sys.argv[1:])
    
    if args.configs:
        import mmengine as mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    print(f"[INFO] Loading dataset from: {args.source_path}")
    print(f"[INFO] Loading checkpoint from: {args.checkpoint}")
    
    # Extract model configuration
    dataset = lp.extract(args)
    hyper = hp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    # Initialize models
    print("[INFO] Initializing Gaussian models...")
    stat_gaussians = GaussianModel(dataset)
    dyn_gaussians = GaussianModel(dataset)

    # Load scene
    print("[INFO] Loading scene...")
    scene = Scene(dataset, dyn_gaussians, stat_gaussians, load_coarse=None)

    # Create pose network
    print("[INFO] Creating pose network...")
    dyn_gaussians.create_pose_network(hyper, scene.getTrainCameras())

    # Load model weights
    print("[INFO] Loading model weights...")
    checkpoint_path = args.checkpoint
    
    # Load PLY files
    ply_path = os.path.join(checkpoint_path, "point_cloud.ply")
    ply_static_path = os.path.join(checkpoint_path, "point_cloud_static.ply")
    
    if not os.path.exists(ply_path):
        print(f"[ERROR] Dynamic PLY not found: {ply_path}")
        sys.exit(1)
    if not os.path.exists(ply_static_path):
        print(f"[ERROR] Static PLY not found: {ply_static_path}")
        sys.exit(1)
    
    print(f"[INFO] Loading PLY files...")
    dyn_gaussians.load_ply(ply_path)
    stat_gaussians.load_ply(ply_static_path)
    
    # Load deformation parameters
    print("[INFO] Loading deformation parameters...")
    dyn_gaussians.load_deformation(checkpoint_path)
    
    # Load pose network
    print("[INFO] Loading pose network...")
    dyn_gaussians.load_model(checkpoint_path)
    dyn_gaussians._posenet.eval()

    # Setup background
    bg_color = [1] * 9 + [0] if dataset.white_background else [0] * 9 + [0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Get cameras
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    
    # Determine which set to render
    render_sets = []
    if args.render_test:
        render_sets.append(("test", test_cams))
    if args.render_train:
        render_sets.append(("train", train_cams))
    
    # If neither specified, render both by default
    if not render_sets:
        render_sets = [("test", test_cams), ("train", train_cams)]

    # Render and save
    output_base = os.path.join("./output", args.expname)
    
    for set_name, cameras in render_sets:
        if cameras and len(cameras) > 0:
            print(f"\n[INFO] ========== Rendering {set_name} set ({len(cameras)} frames) ==========")
            
            output_dir = os.path.join(output_base, f"renders_{set_name}")
            video_path = os.path.join(output_base, f"output_{set_name}.mp4")
            
            render_evaluation(
                scene, cameras, dyn_gaussians, stat_gaussians, 
                background, output_dir, video_output=video_path
            )
    
    print(f"\n[INFO] ========== Inference complete ==========")
    print(f"[INFO] Output saved to: {output_base}")
