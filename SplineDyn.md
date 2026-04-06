# SplineGS 动态区域独立生成修改说明 (SplineDyn)

## 1. 修改背景与目标

在原版的 SplineGS 项目中，场景被显式地解耦为“静态点云 (Static Gaussians)”和“动态点云 (Dynamic Gaussians)”。在执行端到端推理或评估时（即调用 `gaussian_renderer/__init__.py` 中的 `render_infer` 函数），原有的逻辑会将静态和动态高斯属性进行 `torch.cat`（拼接），然后统一统一送入光栅化器（Rasterization）中渲染出全景图像。

为了满足当前需求——**“在推理阶段只生成场景中的动态区域，忽略并剔除背景/静态区域，并且绝不能影响此前的任何训练逻辑”**，我们需要掐断静、动点云的拼接操作。

## 2. 具体修改点

**文件路径**: `/SplineGS/gaussian_renderer/__init__.py`  
**核心函数**: `render_infer`

我们去掉了静态高斯参量与动态高斯参量拼接的过程，将传入光栅化器 (`rasterization`) 的参数严格限制在只使用从 `deform_means3D` 计算得来的动态网格之中。

### 原始代码逻辑 (修改前)
```python
    # Combine stat and dyn gaussians
    means3D_final = torch.cat((smeans3D_final, means3D_final), 0)
    scales_final = torch.cat((sscales_final, scales_final), 0)
    rotations_final = torch.cat((srotations_final, rotations_final), 0)
    opacity_final = torch.cat((sopacity_final, opacity_final), 0)
    colors_precomp_final = torch.cat((stat_colors_precomp, colors_precomp), 0)
```

### 替换后逻辑 (修改后)
```python
    # Use only dyn gaussians to generate only the dynamic region
    means3D_final = means3D_final
    scales_final = scales_final
    rotations_final = rotations_final
    opacity_final = opacity_final
    colors_precomp_final = colors_precomp
```

## 3. 修改结果与影响分析

1. **推理解耦**：光栅化器 (`rasterization`) 将不再接收任何静态高斯基元（`stat_means3D` 等参量被直接舍弃），它只会渲染处在运动/变形状态的高斯点。
2. **背景表现**：所有原本应属于“静态”或“空白”的区域，由于没有任何高斯基元覆盖，自然会露出由在 `eval_nvidia.py` 中定义的背景底色（通过 `bg_color` 控制，这里默认情况为全黑色 `[0, 0, 0]`）。
3. **显著的性能跃升 (FPS)**：静态点云通常占据百万甚至千万级基元，而动态物体点云通常只有十几到几十万量级。去掉静态拼接后，极大地减少了光栅化的排序与投射计算，将极大程度提升渲染动态视频的帧率，并降低显存的占用。
4. **无损原训练管道**：所有的训练文件（如 `train.py` 以及它内部调用的具有梯度反馈的 `render` 方法）未被改变，仅仅从测试推理出口（`render_infer`）截流了静态数据的渲染。保障了训练时可以照常学习，但推断时只“抠出”纯净的动态场景。