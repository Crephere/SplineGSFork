#!/usr/bin/env python3
"""
使用训练好的SplineGS模型生成视频
"""

import os
import sys
import torch
import cv2
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

from arguments import ModelHiddenParams, ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.graphics_utils import getWorld2View2


if __name__ == "__main__":
    parser = ArgumentParser(description="Render video from trained model")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (output/balloon1_aligned)")
    parser.add_argument("--output_video", type=str, default="output_video.mp4", help="Output video path")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to render (None=all test frames)")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--configs", type=str, default="", help="Config file path")
    
    args = parser.parse_args()
    
    # 加载配置
    if args.configs:
        import mmengine as mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    print(f"Loading model from: {args.checkpoint}")
    print(f"Data path: {args.source_path}")
    
    # 初始化参数和模型
    dataset = lp.extract(args)
    hyper = hp.extract(args)
    stat_gaussians = GaussianModel(dataset)
    dyn_gaussians = GaussianModel(dataset)
    pipe = pp.extract(args)
    
    # 创建场景
    scene = Scene(dataset, dyn_gaussians, stat_gaussians, load_coarse=None)
    
    # 初始化pose network
    dyn_gaussians.create_pose_network(hyper, scene.getTrainCameras())
    dyn_gaussians._posenet.eval()
    stat_gaussians._posenet = dyn_gaussians._posenet
    
    # 背景颜色
    bg_color = [1] * 9 + [0] if dataset.white_background else [0] * 9 + [0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 加载模型
    dyn_gaussians.load_ply(os.path.join(args.checkpoint, "point_cloud.ply"))
    stat_gaussians.load_ply(os.path.join(args.checkpoint, "point_cloud_static.ply"))
    dyn_gaussians.load_model(args.checkpoint)
    
    # 获取测试相机
    test_cams = scene.getTestCameras()
    if args.num_frames:
        test_cams = test_cams[:args.num_frames]
    
    print(f"Rendering {len(test_cams)} frames...")
    
    # 渲染帧
    frames = []
    with torch.no_grad():
        for idx, viewpoint_cam in enumerate(tqdm(test_cams, desc="Rendering")):
            render_pkg = render(viewpoint_cam, stat_gaussians, dyn_gaussians, pipe, background)
            image = render_pkg["render"]
            
            # 转为numpy
            image = image.cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            image = np.clip(image * 255, 0, 255).astype(np.uint8)  # float -> uint8
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB -> BGR for OpenCV
            
            frames.append(image)
    
    # 保存视频
    if len(frames) > 0:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output_video, fourcc, args.fps, (w, h))
        
        print(f"\nSaving video to: {args.output_video}")
        for frame in tqdm(frames, desc="Writing video"):
            writer.write(frame)
        writer.release()
        
        print(f"✅ Video saved! Resolution: {w}x{h}, Frames: {len(frames)}, FPS: {args.fps}")
    else:
        print("❌ No frames rendered!")
