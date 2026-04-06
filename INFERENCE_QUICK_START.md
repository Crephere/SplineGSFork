# 🎬 SplineGS Mask-Aligned Model 推理工具套件

## 概览

train_aligned.py已成功训练。现在使用这些工具生成视频验证mask对齐的效果。

```
📊 训练结果（balloon1数据集）
├─ 最佳PSNR：24.27 dB @ iteration 4200
├─ 最终PSNR：24.20 dB @ iteration 5000
├─ 训练时间：~25分钟（5000 iterations）
└─ 检查点：output/balloon1_aligned/point_cloud/iteration_5000/
```

---

## 🛠️ 工具清单

| 工具 | 用途 | 推荐场景 |
|------|------|--------|
| **test_aligned_inference.py** | 推理+视频生成 | 快速验证单个模型效果 |
| **run_inference.sh** | 命令行快速启动 | 不想记命令行参数 |
| **compare_aligned_results.py** | 并行对比渲染 | 对比原始模型vs对齐模型 |
| **TEST_ALIGNED_INFERENCE_GUIDE.md** | 详细文档 | 深入理解参数 |

---

## ⚡ 快速开始（3步）

### 1️⃣ 渲染单个模型（推荐首先尝试）

```bash
cd SplineGS/

# 方式A：使用Shell脚本（最简单）
bash run_inference.sh \
    ./output/balloon1_aligned/point_cloud/iteration_5000 \
    ../DIBR示例/Pose-Warping/Data/balloon1 \
    test_aligned_output

# 方式B：直接Python（完全控制）
python test_aligned_inference.py \
    --checkpoint ./output/balloon1_aligned/point_cloud/iteration_5000 \
    --source_path ../DIBR示例/Pose-Warping/Data/balloon1 \
    --expname test_aligned_output \
    --render_test --render_train
```

**输出：**
```
output/test_aligned_output/
├── output_test.mp4              ← 测试集视频
├── output_train.mp4             ← 训练集视频
├── renders_test/frame_*.png     ← 单帧图像（测试）
└── renders_train/frame_*.png    ← 单帧图像（训练）
```

### 2️⃣ 对比两个模型（如果有原始模型）

如果还有train.py生成的原始模型checkpoint，可生成并排对比视频：

```bash
python compare_aligned_results.py \
    --checkpoint_original ./output/balloon1_original/point_cloud/iteration_5000 \
    --checkpoint_aligned ./output/balloon1_aligned/point_cloud/iteration_5000 \
    --source_path ../DIBR示例/Pose-Warping/Data/balloon1 \
    --output_dir ./comparison_results
```

**输出：**
```
comparison_results/
├── comparison.mp4           ← 左侧原始模型 | 右侧对齐模型
└── comparison_frame_0.png   ← 首帧参考图
```

### 3️⃣ 查看结果

```bash
# macOS/Linux
open output/test_aligned_output/output_test.mp4

# 或用任意视频播放器打开
```

---

## 📋 参数详解

### test_aligned_inference.py 主要参数

```python
# 必需参数
--checkpoint        # 路径到包含point_cloud.ply的检查点目录
--source_path       # 数据集根目录

# 可选参数
--expname           # 输出目录名（默认：test_aligned_output）
--render_test       # 渲染测试集（默认：不渲染）
--render_train      # 渲染训练集（默认：不渲染）
--dataset_type      # 数据格式（默认：colmap）
--configs           # 配置文件路径

# 示例：都不指定时，默认同时渲染test和train
python test_aligned_inference.py --checkpoint ... --source_path ...
```

### Checkpoint 路径选择

| 路径 | 何时使用 | 说明 |
|------|---------|------|
| `iteration_5000/` | 快速测试 | 最终训练迭代（通常最稳定） |
| `fine_best/` | 官方评估 | 验证集PSNR最优点（iter 4200） |
| `iteration_3000/` | 中间调试 | 观察训练进度 |

**推荐：** 两个都试一遍，对比质量

---

## 🔍 验证mask对齐的要点

### ✅ 应该看到的特征

- **运动连贯性**：相机的运动平滑，无跳跃
- **边界清晰**：动态对象轮廓边界清晰锐利
- **无鬼影**：动态区域内不应有源视图的残迹重影
- **一致性**：不同视角下同一帧应呈现相同动态对象形状

### ❌ 问题迹象

- **运动不稳定**：动态对象频繁跳跃
- **边界模糊**：轮廓混乱或破碎
- **鬼影伪影**：动态区域内有残留纹理
- **视角不一致**：不同视角下形状差异大

### 📊 定量比较（如果有原始模型）

使用`compare_aligned_results.py`生成并排视频，关键指标：
- **PSNR**：渲染图像质量（通过cal_psnr.py）
- **Mask IoU**：d_alpha与DIBR目标mask重叠度
- **边界距离误差**：轮廓精度

---

## 🐛 常见问题排查

### Q: 运行test_aligned_inference.py报错"PLY not found"
**A:** 检查checkpoint路径是否正确指向包含point_cloud.ply的目录
```bash
ls your_checkpoint_path/point_cloud.ply
# 应该存在，否则检查点路径错误
```

### Q: 视频生成失败
**A:** 检查是否安装了编码库
```bash
python -c "import cv2; print(cv2.__version__)"
# 需要 opencv-python >= 4.5
```

### Q: 内存不足（CUDA Out of Memory）
**A:** 修改rendering batch size
```python
# 在test_aligned_inference.py中修改
for idx, viewpoint in enumerate(cameras[::2]):  # 每2帧渲染一帧
    ...
```

### Q: 生成的视频帧率不对
**A:** 修改get_pixels中的fps参数
```bash
python test_aligned_inference.py ... --video_fps 30  # 改为30fps
```

---

## 📁 输出文件结构详解

```
output/
└── test_aligned_output/              # expname指定
    ├── renders_test/                 # --render_test生成
    │   ├── frame_0000.png            # 高质量PNG
    │   ├── frame_0001.png
    │   └── ... (所有测试帧)
    ├── renders_train/                # --render_train生成
    │   ├── frame_0000.png
    │   └── ...
    ├── output_test.mp4               # 24fps MP4视频
    └── output_train.mp4
```

每个PNG保存完整分辨率（未压缩），MP4利用H.264编码。

---

## 🚀 高级用法

### 只渲染部分帧（快速测试）

编辑test_aligned_inference.py：
```python
# 修改render_evaluation()函数
for idx, viewpoint in enumerate(cameras[:20]):  # 仅前20帧
    ...
```

### 自定义FPS或编码

```python
# 在render_evaluation()中修改
out = cv2.VideoWriter(video_output, fourcc, 30.0, (w, h))  # 改为30fps
```

### 调试模式（打印更多信息）

修改test_aligned_inference.py：
```python
# 在render_evaluation()中添加
print(f"Frame {idx}: time={viewpoint.time:.3f}, R_shape={viewpoint.R.shape}")
```

---

## 🎯 推荐工作流程

### 第1天：快速验证
```bash
# 仅测试集，快速看效果
bash run_inference.sh
# ↓ 检查output/test_aligned_output/output_test.mp4
```

### 第2天：完整渲染
```bash
# 测试集+训练集
python test_aligned_inference.py \
    --checkpoint ./output/balloon1_aligned/point_cloud/iteration_5000 \
    --source_path ../DIBR示例/Pose-Warping/Data/balloon1 \
    --expname final_validation \
    --render_test --render_train
```

### 第3天：对比分析（如果有原始模型）
```bash
python compare_aligned_results.py \
    --checkpoint_original ./output/balloon1_original/point_cloud/iteration_5000 \
    --checkpoint_aligned ./output/balloon1_aligned/point_cloud/iteration_5000 \
    --source_path ../DIBR示例/Pose-Warping/Data/balloon1 \
    --output_dir ./final_comparison
# 并排对比效果
```

---

## 📝 注意事项

1. **GPU内存**：默认配置需要~8GB显存，GTX3090 Recommended
2. **首帧慢**：集成第一帧会触发CUDA warming，属正常现象
3. **数据集兼容**：目前仅测试COLMAP格式，其他格式修改--dataset_type
4. **视频格式**：输出为H.264 MP4，兼容所有主流播放器

---

## 📚 更多资料

- [完整参数文档](TEST_ALIGNED_INFERENCE_GUIDE.md)
- [train_aligned.py 源码注释](train_aligned.py)
- [原始 eval_nvidia.py](eval_nvidia.py)

---

## ✨ 总结

| 步骤 | 命令 | 耗时 |
|------|------|------|
| ✅ 模型训练（已完成） | train_aligned.py | ~25min |
| 🎯 单模型推理 | test_aligned_inference.py | ~5-10min（取决于帧数） |
| 🔄 双模型对比 | compare_aligned_results.py | ~10-20min（2x渲染） |
| 📹 视频编码 | 自动 | ~1-2min |

**下一步：** 运行test_aligned_inference.py生成视频，visually evaluate mask alignment效果！
