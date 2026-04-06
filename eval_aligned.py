"""
简单的eval脚本 - 用train_aligned.py生成的模型进行推理并生成图像
"""
import os
import sys
import torch
import numpy as np
from argparse import ArgumentParser
from PIL import Image

from arguments import ModelParams, OptimizationParams, PipelineParams, ModelHiddenParams
from gaussian_renderer import render_infer
from scene import GaussianModel, Scene, dataset_readers, deformation
from utils.graphics_utils import getWorld2View2
from utils.image_utils import psnr

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("[WARNING] LPIPS not available, LPIPS scores will be skipped")


def normalize_image(img):
    """用于LPIPS的图像归一化"""
    return (2.0 * img - 1.0)[None, ...]


def eval_aligned_model(checkpoint_path, data_dir, output_dir, split='test', save_images=True):
    """
    评估train_aligned.py生成的模型
    
    Args:
        checkpoint_path: 模型checkpoint所在目录 (如 output/balloon1_aligned/point_cloud/iteration_5000/)
        data_dir: 数据集目录
        output_dir: 输出目录
        split: 'test' 或 'train'
        save_images: 是否保存图像
    """
    
    # ========== 参数设置 ==========
    parser = ArgumentParser(description="Eval script for aligned model")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    parser.add_argument("--iteration", type=int, default=-1)
    
    # 手动构建args对象
    args = parser.parse_args([])
    args.source_path = data_dir
    args.output_path = output_dir
    
    # ========== 加载数据集 ==========
    dataset = lp.extract(args)
    hyper = hp.extract(args)
    
    # 初始化静态和动态高斯
    stat_gaussians = GaussianModel(dataset)
    dyn_gaussians = GaussianModel(dataset)
    
    print(f"Loading scene from {data_dir}")
    scene = Scene(dataset, dyn_gaussians, stat_gaussians, load_coarse=None)
    
    # ========== 加载模型权重 ==========
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # 加载点云
    dyn_pt_path = os.path.join(checkpoint_path, "point_cloud.ply")
    stat_pt_path = os.path.join(checkpoint_path, "point_cloud_static.ply")
    
    if not os.path.exists(dyn_pt_path):
        print(f"[ERROR] Point cloud not found: {dyn_pt_path}")
        return
    
    dyn_gaussians.load_ply(dyn_pt_path)
    stat_gaussians.load_ply(stat_pt_path)
    
    print("✓ Loaded point clouds")
    
    # 加载模型配置（包括deformation network等）
    try:
        dyn_gaussians.load_model(checkpoint_path)
        print("✓ Loaded deformation model")
    except Exception as e:
        print(f"[WARNING] Failed to load deformation model: {e}")
    
    # 设置为eval模式
    dyn_gaussians._posenet.eval()
    stat_gaussians._posenet.eval() if hasattr(stat_gaussians, '_posenet') else None
    
    # ========== 准备背景颜色 ==========
    if dataset.white_background:
        bg_color = [1, 1, 1, -10]
    else:
        bg_color = [0, 0, 0, -10]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe = pp.extract(args)
    
    # ========== 获取cameras ==========
    if split == 'test':
        cameras = scene.getTestCameras()
        split_name = 'test'
    else:
        cameras = scene.getTrainCameras()
        split_name = 'train'
    
    if not cameras or len(cameras) == 0:
        print(f"[ERROR] No {split} cameras found")
        return
    
    print(f"Evaluating on {len(cameras)} {split_name} cameras")
    
    # ========== 初始化metrics ==========
    psnr_list = []
    lpips_list = []
    render_times = []
    
    if LPIPS_AVAILABLE:
        lpips_loss = lpips.LPIPS(net="alex").cuda()
    
    # ========== 推理循环 ==========
    os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    print(f"\n{'Frame':<6} {'PSNR':<8} {'LPIPS':<8} {'Time(ms)':<10}")
    print("-" * 40)
    
    with torch.no_grad():
        for frame_idx, viewpoint in enumerate(cameras):
            # Warmup (skip first frame)
            if frame_idx == 0:
                for _ in range(3):
                    _ = render_infer(
                        viewpoint, stat_gaussians, dyn_gaussians, background
                    )
            
            # 渲染
            torch.cuda.synchronize()
            start_event.record()
            
            render_pkg = render_infer(
                viewpoint, stat_gaussians, dyn_gaussians, background
            )
            
            end_event.record()
            torch.cuda.synchronize()
            
            render_time = start_event.elapsed_time(end_event)
            render_times.append(render_time)
            
            # 获取渲染结果
            image = render_pkg["render"]
            image = torch.clamp(image, 0.0, 1.0)
            
            # 计算指标
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            
            psnr_val = psnr(image, gt_image, mask=None).mean().double().item()
            psnr_list.append(psnr_val)
            
            lpips_val = None
            if LPIPS_AVAILABLE:
                lpips_val = lpips_loss.forward(
                    normalize_image(image), 
                    normalize_image(gt_image)
                ).item()
                lpips_list.append(lpips_val)
            
            # 打印进度
            lpips_str = f"{lpips_val:.4f}" if lpips_val else "N/A"
            print(f"{frame_idx:<6} {psnr_val:<8.4f} {lpips_str:<8} {render_time:<10.2f}")
            
            # 保存图像
            if save_images:
                img_np = (np.clip(
                    image.permute(1, 2, 0).detach().cpu().numpy(), 
                    0, 1
                ) * 255).astype("uint8")
                
                img_pil = Image.fromarray(img_np)
                img_path = os.path.join(output_dir, split_name, f"img_{frame_idx:04d}.png")
                img_pil.save(img_path)
    
    # ========== 打印总结 ==========
    print("-" * 40)
    avg_psnr = np.mean(psnr_list)
    avg_time = np.mean(render_times)
    fps = 1000.0 / avg_time if avg_time > 0 else 0
    
    print(f"\n📊 Summary ({split_name} split):")
    print(f"  Average PSNR: {avg_psnr:.4f}")
    if LPIPS_AVAILABLE and lpips_list:
        avg_lpips = np.mean(lpips_list)
        print(f"  Average LPIPS: {avg_lpips:.4f}")
    print(f"  Average render time: {avg_time:.2f} ms")
    print(f"  FPS: {fps:.2f}")
    print(f"  Images saved to: {os.path.join(output_dir, split_name)}/")
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Eval aligned SplineGS model")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to checkpoint directory (e.g., output/balloon1_aligned/point_cloud/iteration_5000/)")
    parser.add_argument("--data-dir", type=str, required=True, 
                        help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="eval_output", 
                        help="Output directory for images and results")
    parser.add_argument("--split", type=str, choices=['test', 'train'], default='test',
                        help="Which split to evaluate on")
    parser.add_argument("--no-save-images", action='store_true',
                        help="Don't save rendered images")
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 Evaluating train_aligned.py model")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Split: {args.split}")
    print("="*60)
    
    eval_aligned_model(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        save_images=not args.no_save_images
    )
    
    print("\n✅ Evaluation complete!")
