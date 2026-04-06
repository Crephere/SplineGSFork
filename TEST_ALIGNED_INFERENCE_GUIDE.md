# train_aligned.py 输出结构和推理说明

## 1. train_aligned.py 的输出

### 保存目录结构
```
output/
└── balloon1_aligned/          # model_path
    └── point_cloud/
        ├── coarse_iteration_XXX/      # 热身阶段检查点
        ├── iteration_YYYY/            # 常规检查点 (每 1000 iter)
        │   ├── point_cloud.ply        # 动态高斯
        │   ├── point_cloud_static.ply # 静态高斯
        │   └── deformation_*.pt       # 变形网络权重
        ├── fine_best/                 # 最佳PSNR的检查点
        │   ├── point_cloud.ply
        │   ├── point_cloud_static.ply
        │   └── deformation_*.pt
        └── events.out.tfevents         # TensorBoard日志 (可选)
```

### 关键文件说明

| 文件 | 作用 |
|-----|------|
| `point_cloud.ply` | 动态高斯的位置、颜色、协方差等参数 |
| `point_cloud_static.ply` | 静态高斯的参数 |
| `deformation_*.pt` | 变形网络（Spline网络）的权重 |
| `args.txt` | 训练使用的所有参数 |

### 尾部检查点策略
- **coarse_iteration_**: 热身阶段（位置追踪）
- **iteration_5000**: 最终训练迭代的检查点
- **fine_best**: 根据验证集PSNR最优的检查点

在本次训练中：
- 最佳PSNR：24.27 dB @ iteration 4200
- 最终PSNR：24.20 dB @ iteration 5000
- 推荐使用：`point_cloud/iteration_5000/` 或 `point_cloud/fine_best/`

---

## 2. eval_nvidia.py 现有评估流程

eval_nvidia.py的核心流程：

```python
# 1. 加载模型
dyn_gaussians.load_ply(checkpoint + "point_cloud.ply")
stat_gaussians.load_ply(checkpoint + "point_cloud_static.ply")
dyn_gaussians.load_model(checkpoint)  # 加载姿态网络权重
dyn_gaussians._posenet.eval()

# 2. 预测相机姿态
for cam in cameras:
    time_in = torch.tensor(cam.time).float().cuda()
    pred_R, pred_T = dyn_gaussians._posenet(time_in.view(1, 1))
    cam.update_cam(R, T, local_viewdirs, ...)

# 3. 渲染
for viewpoint in cameras:
    render_pkg = render_infer(viewpoint, stat_gaussians, dyn_gaussians, background)
    image = render_pkg["render"]  # [3, H, W] 张量
    
# 4. 计算指标
psnr = PSNR(rendered, gt)
lpips = LPIPS(rendered, gt)
```

---

## 3. 新脚本：test_aligned_inference.py 使用方法

### 核心改进点
与eval_nvidia.py相比，test_aligned_inference.py增加：
- ✅ **视频生成**：自动创建MP4视频（fps=24）
- ✅ **灵活相机选择**：可分别或同时渲染训练集/测试集
- ✅ **个别帧保存**：同时保存PNG帧用于细致检查
- ✅ **无需GT数据**：不计算PSNR/LPIPS（仅渲染验证）

### 使用方式

#### 基础用法（渲染测试集 + 生成视频）
```bash
cd SplineGS/
python test_aligned_inference.py \
    --checkpoint ./output/balloon1_aligned/point_cloud/iteration_5000 \
    --source_path ../DIBR示例/Pose-Warping/Data/balloon1 \
    --expname test_aligned_output \
    --render_test
```

#### 渲染训练集
```bash
python test_aligned_inference.py \
    --checkpoint ./output/balloon1_aligned/point_cloud/iteration_5000 \
    --source_path ../DIBR示例/Pose-Warping/Data/balloon1 \
    --render_train
```

#### 同时渲染训练集和测试集（完整验证）
```bash
python test_aligned_inference.py \
    --checkpoint ./output/balloon1_aligned/point_cloud/iteration_5000 \
    --source_path ../DIBR示例/Pose-Warping/Data/balloon1 \
    --render_test --render_train
```

### 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | ✓ | - | 检查点路径（必须指向包含point_cloud.ply的目录） |
| `--source_path` | ✓ | - | 数据集路径 |
| `--expname` | × | test_aligned_output | 输出目录名 |
| `--render_test` | × | False | 渲染测试集 |
| `--render_train` | × | False | 渲染训练集 |
| `--dataset_type` | × | colmap | 数据集类型 |
| `--configs` | × | - | 配置文件路径（可选） |

### 输出结构

```
output/
└── test_aligned_output/
    ├── renders_test/
    │   ├── frame_0000.png
    │   ├── frame_0001.png
    │   ├── ...
    │   └── frame_NNNN.png
    ├── renders_train/
    │   ├── frame_0000.png
    │   ├── ...
    ├── output_test.mp4      # 测试集视频
    └── output_train.mp4     # 训练集视频
```

---

## 4. 推理流程详解

### 步骤1: 加载检查点
```python
dyn_gaussians.load_ply(checkpoint_path + "point_cloud.ply")
stat_gaussians.load_ply(checkpoint_path + "point_cloud_static.ply")
dyn_gaussians.load_deformation(checkpoint_path)  # 加载spline网络
dyn_gaussians.load_model(checkpoint_path)  # 加载姿态网络
```

### 步骤2: 预测帧的相机姿态
对于视频序列中的每一帧，根据**时间戳**预测其动态对象的相机相对位置：
```python
for frame in sequence:
    time_in = torch.tensor(frame.timestamp)  # 归一化时间 [0, 1]
    pred_R, pred_T = pose_network(time_in)   # 预测旋转和平移
    frame.update_cam(pred_R, pred_T)         # 更新相机内外参
```

关键点：**这是mask对齐的核心**
- DIBR定义动作：通过刚性mask变形
- SplineGS学习动作：通过时间→相机姿态映射
- 训练约束：使SplineGS渲染的d_alpha ≈ DIBR的目标mask

### 步骤3: 渲染动态场景
```python
for viewpoint in cameras:
    render_output = render_infer(viewpoint, static_gs, dynamic_gs, bg)
    image = render_output["render"]  # RGB图像
    # 可选：image还包含其他通道如深度、法线等
```

### 步骤4: 保存为视频
```
H265编码 24fps MP4 → output/test_aligned_output/output_test.mp4
```

---

## 5. 验证mask对齐的方法

### 方法1：视觉检查
运行test_aligned_inference.py生成的视频，观察：
- ✓ 动态对象轮廓是否清晰
- ✓ 是否有"幽灵"伪影（mask裂口）
- ✓ 运动是否平滑自然

### 方法2：定量对比
在DIBR-GS中添加mask对齐度量：
```python
def compute_mask_alignment(d_alpha, dibr_target_mask):
    """计算SplineGS d_alpha与DIBR目标mask的IoU"""
    intersection = (d_alpha > 0.5) & (dibr_target_mask > 0.5)
    union = (d_alpha > 0.5) | (dibr_target_mask > 0.5)
    iou = intersection.sum() / union.sum()
    return iou
```

### 方法3：边界对齐测试
检查mask边界是否在同驾对齐约束下改善。可修改eval_nvidia.py
计算边界像素误差（距离变换）。

---

## 6. 快速开始命令

快速测试（渲染10帧）：
```bash
# 编辑test_aligned_inference.py，在render_evaluation()中修改：
for idx, viewpoint in enumerate(cameras[:10]):  # 仅渲染前10帧
    ...
```

完整渲染（所有帧生成视频）：
```bash
python test_aligned_inference.py \
    --checkpoint ./output/balloon1_aligned/point_cloud/fine_best \
    --source_path /path/to/balloon1_data \
    --expname final_aligned_video \
    --render_test
```

---

## 7. 常见问题

**Q: 如何选择iteration检查点vs fine_best？**
- `iteration_5000`：最后一次训练迭代
- `fine_best`：验证集PSNR最优时的权重（iter 4200）
- 推荐：先试fine_best（质量最好），再用iteration_5000（最终状态）

**Q: 视频生成失败？**
- 检查ffmpeg/cv2是否正确安装：`python -c "import cv2; print(cv2.VideoWriter_fourcc)"`
- 或修改代码使用scipy.io.wavfile保存视频

**Q: 生成的视频帧数不对？**
- 检查dataset loader是否正确加载所有帧
- 查看print输出的"Rendering X/Y frames"

**Q: 如何对比原始train.py vs train_aligned.py的结果？**
1. 分别用两个脚本训练同个数据集
2. 运行test_aligned_inference.py生成两个视频
3. 并排观看，评估mask对齐是否改善
