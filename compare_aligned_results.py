#!/usr/bin/env python3
"""
Visual comparison tool for original vs mask-aligned trained models.
Side-by-side video comparison to evaluate mask alignment improvement.

Usage:
    python compare_aligned_results.py \
        --checkpoint_original ./output/balloon1_original/point_cloud/iteration_5000 \
        --checkpoint_aligned ./output/balloon1_aligned/point_cloud/iteration_5000 \
        --source_path ../DIBR示例/Pose-Warping/Data/balloon1 \
        --output_dir ./comparison_results
"""
import os
import sys
import torch
import cv2
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(sys.path[0], ".."))

from arguments import ModelHiddenParams, ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render_infer
from scene import GaussianModel, Scene, dataset_readers
from utils.main_utils import get_pixels


def load_model(checkpoint_path, dataset, hyper):
    """Load Gaussian model from checkpoint."""
    stat_gaussians = GaussianModel(dataset)
    dyn_gaussians = GaussianModel(dataset)
    
    dyn_gaussians.create_pose_network(hyper, None)
    
    ply_path = os.path.join(checkpoint_path, "point_cloud.ply")
    ply_static_path = os.path.join(checkpoint_path, "point_cloud_static.ply")
    
    if not os.path.exists(ply_path) or not os.path.exists(ply_static_path):
        raise FileNotFoundError(f"PLY files not found in {checkpoint_path}")
    
    dyn_gaussians.load_ply(ply_path)
    stat_gaussians.load_ply(ply_static_path)
    dyn_gaussians.load_deformation(checkpoint_path)
    dyn_gaussians.load_model(checkpoint_path)
    dyn_gaussians._posenet.eval()
    
    return stat_gaussians, dyn_gaussians


def render_frame(viewpoint, stat_gaussians, dyn_gaussians, background):
    """Render single frame."""
    with torch.no_grad():
        render_pkg = render_infer(viewpoint, stat_gaussians, dyn_gaussians, background)
        image = render_pkg["render"]
        image = torch.clamp(image, 0.0, 1.0)
        return (np.clip(image.permute(1, 2, 0).detach().cpu().numpy(), 0, 1) * 255).astype("uint8")


def create_comparison_video(checkpoint_original, checkpoint_aligned, scene, dataset, hyper, background, output_path):
    """
    Create side-by-side comparison video.
    
    Layout:
    ┌─────────────────────┬─────────────────────┐
    │   Original Model    │  Aligned Model      │
    │ (No mask constraint)│(With DIBR constraint)|
    └─────────────────────┴─────────────────────┘
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Load both models
    print("[INFO] Loading original model...")
    stat_gs_orig, dyn_gs_orig = load_model(checkpoint_original, dataset, hyper)
    
    print("[INFO] Loading aligned model...")
    stat_gs_align, dyn_gs_align = load_model(checkpoint_aligned, dataset, hyper)
    
    # Get cameras
    cameras = scene.getTestCameras()
    if not cameras:
        print("[WARN] No test cameras, using train cameras")
        cameras = scene.getTrainCameras()
    
    # Get pixels and viewdirs for pose network
    pixels = get_pixels(
        scene.train_camera.dataset[0].metadata.image_size_x,
        scene.train_camera.dataset[0].metadata.image_size_y,
        use_center=True,
    )
    batch_shape = pixels.shape[:-1]
    pixels = np.reshape(pixels, (-1, 2))
    
    y = (pixels[..., 1] - scene.train_camera.dataset[0].metadata.principal_point_y) / \
        dyn_gs_orig._posenet.focal_bias.exp().detach().cpu().numpy()
    x = (pixels[..., 0] - scene.train_camera.dataset[0].metadata.principal_point_x) / \
        dyn_gs_orig._posenet.focal_bias.exp().detach().cpu().numpy()
    
    viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    local_viewdirs = viewdirs / np.linalg.norm(viewdirs, axis=-1, keepdims=True)
    
    # Predict poses for both models
    print("[INFO] Predicting camera poses...")
    with torch.no_grad():
        for cam in cameras:
            # Predict with original model
            time_in = torch.tensor(cam.time).float().cuda()
            pred_R_orig, pred_T_orig = dyn_gs_orig._posenet(time_in.view(1, 1))
            
            # Predict with aligned model (note: different pose network)
            pred_R_align, pred_T_align = dyn_gs_align._posenet(time_in.view(1, 1))
            
            # Store both predictions in camera object
            cam.R_orig = torch.transpose(pred_R_orig, 2, 1).detach().cpu().numpy()[0]
            cam.T_orig = pred_T_orig.detach().cpu().numpy()[0]
            cam.R_align = torch.transpose(pred_R_align, 2, 1).detach().cpu().numpy()[0]
            cam.T_align = pred_T_align.detach().cpu().numpy()[0]
    
    # Render frames and create video
    print("[INFO] Rendering comparison video...")
    h, w = scene.train_camera.dataset[0].metadata.image_size_y, \
            scene.train_camera.dataset[0].metadata.image_size_x
    
    # Output video dimensions: [H, 2*W]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        os.path.join(output_path, "comparison.mp4"),
        fourcc, 24.0, (2 * w, h)
    )
    
    # Add titles as overlays
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 2
    color = (255, 255, 255)
    
    with torch.no_grad():
        for idx, cam in enumerate(tqdm(cameras, desc="Rendering frames")):
            # Create a camera copy for original model
            cam_orig_copy = cam.__class__(
                colmap_id=cam.colmap_id, R=cam.R_orig, T=cam.T_orig,
                FoVx=cam.FoVx, FoVy=cam.FoVy,
                image=cam.original_image if hasattr(cam, 'original_image') else None,
                gt_alpha_mask=cam.gt_alpha_mask if hasattr(cam, 'gt_alpha_mask') else None,
                image_name=cam.image_name if hasattr(cam, 'image_name') else "",
                uid=cam.uid if hasattr(cam, 'uid') else 0,
                trans=cam.trans if hasattr(cam, 'trans') else np.array([0, 0, 0]),
                scale=cam.scale if hasattr(cam, 'scale') else 1.0,
                time=cam.time if hasattr(cam, 'time') else 0.0,
            )
            cam_orig_copy.update_cam(cam.R_orig, cam.T_orig, local_viewdirs, batch_shape,
                                     dyn_gs_orig._posenet.focal_bias.exp().detach().cpu().numpy())
            
            # Create a camera copy for aligned model
            cam_align_copy = cam.__class__(
                colmap_id=cam.colmap_id, R=cam.R_align, T=cam.T_align,
                FoVx=cam.FoVx, FoVy=cam.FoVy,
                image=cam.original_image if hasattr(cam, 'original_image') else None,
                gt_alpha_mask=cam.gt_alpha_mask if hasattr(cam, 'gt_alpha_mask') else None,
                image_name=cam.image_name if hasattr(cam, 'image_name') else "",
                uid=cam.uid if hasattr(cam, 'uid') else 0,
                trans=cam.trans if hasattr(cam, 'trans') else np.array([0, 0, 0]),
                scale=cam.scale if hasattr(cam, 'scale') else 1.0,
                time=cam.time if hasattr(cam, 'time') else 0.0,
            )
            cam_align_copy.update_cam(cam.R_align, cam.T_align, local_viewdirs, batch_shape,
                                      dyn_gs_align._posenet.focal_bias.exp().detach().cpu().numpy())
            
            # Render with original model
            frame_orig = render_frame(cam_orig_copy, stat_gs_orig, dyn_gs_orig, background)
            
            # Render with aligned model
            frame_align = render_frame(cam_align_copy, stat_gs_align, dyn_gs_align, background)
            
            # Create side-by-side frame
            frame_comparison = np.hstack([frame_orig, frame_align])
            
            # Add text labels
            cv2.putText(frame_comparison, "Original", (30, 50), font, font_scale, color, thickness)
            cv2.putText(frame_comparison, "Mask-Aligned", (w + 30, 50), font, font_scale, color, thickness)
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_comparison, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
            # Save first frame as reference
            if idx == 0:
                cv2.imwrite(os.path.join(output_path, "comparison_frame_0.png"), frame_bgr)
    
    out.release()
    print(f"[INFO] Comparison video saved: {os.path.join(output_path, 'comparison.mp4')}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Comparison tool for aligned training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)

    parser.add_argument("--checkpoint_original", type=str, required=True, help="Original model checkpoint")
    parser.add_argument("--checkpoint_aligned", type=str, required=True, help="Aligned model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./comparison_results")
    parser.add_argument("--configs", type=str, default="")

    args = parser.parse_args(sys.argv[1:])
    
    if args.configs:
        import mmengine as mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    # Setup
    dataset = lp.extract(args)
    hyper = hp.extract(args)
    
    stat_gaussians = GaussianModel(dataset)
    dyn_gaussians = GaussianModel(dataset)
    
    scene = Scene(dataset, dyn_gaussians, stat_gaussians, load_coarse=None)
    dyn_gaussians.create_pose_network(hyper, scene.getTrainCameras())
    
    bg_color = [1] * 9 + [0] if dataset.white_background else [0] * 9 + [0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Create comparison
    print("[INFO] Creating comparison video...")
    print(f"[INFO] Original:  {args.checkpoint_original}")
    print(f"[INFO] Aligned:   {args.checkpoint_aligned}")
    print(f"[INFO] Output:    {args.output_dir}")
    
    create_comparison_video(
        args.checkpoint_original,
        args.checkpoint_aligned,
        scene, dataset, hyper,
        background,
        args.output_dir
    )
    
    print("[INFO] ✅ Comparison complete!")
