import os
import sys
import torch
sys.path.append(os.path.join(sys.path[0], ".."))
import cv2
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm

from arguments import ModelHiddenParams, ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render_infer
from scene import GaussianModel, Scene
from utils.main_utils import get_pixels
from utils.image_utils import psnr


if __name__ == "__main__":
    parser = ArgumentParser(description="Eval aligned model - render video from train_aligned.py checkpoint")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory (e.g., output/balloon1_aligned/point_cloud/iteration_5000)")
    parser.add_argument("--expname", type=str, default="eval_aligned")
    parser.add_argument("--configs", type=str, default="")

    args = parser.parse_args(sys.argv[1:])
    if args.configs:
        import mmengine as mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    # 1. 加载数据集和模型配置
    dataset = lp.extract(args)
    hyper = hp.extract(args)
    stat_gaussians = GaussianModel(dataset)
    dyn_gaussians = GaussianModel(dataset)
    opt = op.extract(args)
    
    # 2. 加载场景
    scene = Scene(dataset, dyn_gaussians, stat_gaussians, load_coarse=None)
    dyn_gaussians.create_pose_network(hyper, scene.getTrainCameras())

    # 3. 加载checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    dyn_gaussians.load_ply(os.path.join(args.checkpoint, "point_cloud.ply"))
    stat_gaussians.load_ply(os.path.join(args.checkpoint, "point_cloud_static.ply"))
    dyn_gaussians.load_model(args.checkpoint)
    dyn_gaussians._posenet.eval()

    # 4. 获取像素和viewdir信息
    pixels = get_pixels(
        scene.train_camera.dataset[0].metadata.image_size_x,
        scene.train_camera.dataset[0].metadata.image_size_y,
        use_center=True,
    )
    batch_shape = pixels.shape[:-1]
    pixels = np.reshape(pixels, (-1, 2))
    
    y = (pixels[..., 1] - scene.train_camera.dataset[0].metadata.principal_point_y) / \
        dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy()
    x = (pixels[..., 0] - scene.train_camera.dataset[0].metadata.principal_point_x) / \
        dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy()
    
    viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    local_viewdirs = viewdirs / np.linalg.norm(viewdirs, axis=-1, keepdims=True)

    # 5. 后台颜色
    bg_color = [1] * 9 + [0] if dataset.white_background else [0] * 9 + [0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 6. 获取测试摄像机
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    
    print(f"\nRender evaluation from checkpoint: {args.checkpoint}")
    print(f"Expname: {args.expname}")
    
    # 7. 渲染并生成视频
    output_base = f"./output/{args.expname}"
    
    for config_name, cameras in [("test", test_cams), ("train", train_cams)]:
        if not cameras or len(cameras) == 0:
            print(f"⚠️  No {config_name} cameras found, skipping")
            continue
            
        print(f"\n{'='*60}")
        print(f"Rendering {config_name} set ({len(cameras)} frames)")
        print(f"{'='*60}")
        
        output_dir = os.path.join(output_base, config_name)
        os.makedirs(output_dir, exist_ok=True)
        
        rendered_frames = []
        psnr_list = []
        
        with torch.no_grad():
            for idx, viewpoint in enumerate(tqdm(cameras, desc=f"Rendering {config_name}")):
                # 更新相机姿态
                time_in = torch.tensor(viewpoint.time).float().cuda()
                pred_R, pred_T = dyn_gaussians._posenet(time_in.view(1, 1))
                R_ = torch.transpose(pred_R, 2, 1).detach().cpu().numpy()
                t_ = pred_T.detach().cpu().numpy()
                
                viewpoint.update_cam(
                    R_[0], t_[0], local_viewdirs, batch_shape,
                    dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy(),
                )
                
                # 渲染
                render_pkg = render_infer(viewpoint, stat_gaussians, dyn_gaussians, background)
                image = render_pkg["render"]
                image = torch.clamp(image, 0.0, 1.0)
                
                # 转换为numpy
                img_np = (np.clip(image.permute(1, 2, 0).detach().cpu().numpy(), 0, 1) * 255).astype("uint8")
                rendered_frames.append(img_np)
                
                # 计算PSNR（如果有GT图像）
                if hasattr(viewpoint, 'original_image') and viewpoint.original_image is not None:
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    psnr_val = psnr(image, gt_image).mean().item()
                    psnr_list.append(psnr_val)
                
                # 保存单帧
                img_pil = Image.fromarray(img_np)
                img_pil.save(os.path.join(output_dir, f"frame_{idx:04d}.png"))
        
        # 生成视频
        if rendered_frames:
            h, w = rendered_frames[0].shape[:2]
            video_path = os.path.join(output_base, f"video_{config_name}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 24.0, (w, h))
            
            for frame in rendered_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            
            print(f"✅ Video saved: {video_path}")
            print(f"   Frames: {len(rendered_frames)}, Resolution: {w}x{h}")
            
            if psnr_list:
                avg_psnr = np.mean(psnr_list)
                print(f"   Avg PSNR: {avg_psnr:.4f}")
            
            print(f"   Images saved to: {output_dir}")
    
    print(f"\n{'='*60}")
    print(f"✅ Evaluation complete!")
    print(f"Output: {output_base}")
    print(f"{'='*60}\n")
