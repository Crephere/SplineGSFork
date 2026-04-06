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

from arguments import ModelHiddenParams, ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from PIL import Image
from tqdm import tqdm


def render_video(
    model_path,
    output_path,
    data_path,
    config_path,
    num_frames=None,
    fps=30,
):
    """
    使用训练好的模型渲染视频
    
    Args:
        model_path: 模型checkpoint路径
        output_path: 输出视频路径
        data_path: 数据集路径
        config_path: 配置文件路径
        num_frames: 生成帧数（None=全部）
        fps: 视频帧率
    """
    
    print(f"Loading model from: {model_path}")
    
    # 解析参数
    parser = ArgumentParser(description="Render video")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    # 读取配置
    import mmengine as mmcv
    from utils.params_utils import merge_hparams
    config = mmcv.Config.fromfile(config_path)
    
    args = parser.parse_args([
        "--source_path", data_path,
        "--model_path", model_path,
    ])
    args = merge_hparams(args, config)
    
    # 创建场景和模型
    dataset = lp.extract(args)
    scene = Scene(dataset, GaussianModel(dataset), GaussianModel(dataset), load_fine=True)
    
    stat_gaussians = scene.stat_gaussians
    dyn_gaussians = scene.dyn_gaussians
    
    # 获取测试相机
    test_cams = scene.getTestCameras()
    
    if num_frames:
        test_cams = test_cams[:num_frames]
    
    print(f"Rendering {len(test_cams)} frames...")
    
    # 背景颜色
    bg_color = [1, 1, 1, -10] if dataset.white_background else [0, 0, 0, -10]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 渲染帧
    frames = []
    with torch.no_grad():
        for idx, viewpoint_cam in enumerate(tqdm(test_cams, desc="Rendering")):
            render_pkg = render(viewpoint_cam, stat_gaussians, dyn_gaussians, background)
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
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        print(f"\nSaving video to: {output_path}")
        for frame in tqdm(frames, desc="Writing video"):
            writer.write(frame)
        writer.release()
        
        print(f"✅ Video saved! Resolution: {w}x{h}, Frames: {len(frames)}, FPS: {fps}")
    else:
        print("❌ No frames rendered!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Render video from trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output_video", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--data_path", type=str, required=True, help="Data directory")
    parser.add_argument("--config_path", type=str, required=True, help="Config file path")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to render")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    
    args = parser.parse_args()
    
    render_video(
        model_path=args.model_path,
        output_path=args.output_video,
        data_path=args.data_path,
        config_path=args.config_path,
        num_frames=args.num_frames,
        fps=args.fps,
    )
