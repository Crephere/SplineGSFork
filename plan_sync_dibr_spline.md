# 解决方案：DIBR-SplineGS运动对齐约束

## 问题根源

**当前融合失败的原因：**

1. **PoseWarp+DIBR**: 在target视角通过刚体变换(相机参数+深度)得到动态区边界
   - `DIBRMasked._warp_dynamic_mask_to_target()` → 源mask通过DIBR投影到目标视角
   - 结果：目标视角的动态区边界由 `(depth, w2c_src, w2c_tgt)` 完全决定
   
2. **SplineGS**: 独立训练，学自己的高斯运动轨迹
   - 仅通过RGB+GT motion_mask(源视角)监督
   - **未约束**与DIBR目标视角的mask边界对齐
   - 结果：学出的动态物体轨迹与DIBR定义的轨迹完全不同→融合时位置错位

---

## 核心解决思路

**关键洞察**：SplineGS的 `d_alpha`(渲染的动态mask)需要与**DIBR目标视角的mask边界**一致，而不仅仅与源视角GT mask一致。

```
源视角GT mask     DIBR刚体变换      目标视角DIBR mask
    ↓              ↓                    ↓
motion_mask  → _warp_dynamic_mask  → dibr_target_mask
(固定)           (确定性变换)         (确定)
                                        ↑
                                        需要与d_alpha对齐
                                        ← SplineGS约束
```

**修改策略**：
1. 在train.py中引入DIBR并计算目标视角的mask边界（`dibr_target_mask`）
2. 修改mask loss：从 `L_mask(d_alpha, motion_mask)` 改为 `L_mask(d_alpha, dibr_target_mask)`
3. 这样SplineGS被迫学一个与DIBR轨迹一致的运动

---

## 实现步骤

### Phase 1: 准备工作
1. 在train.py顶部导入DIBR相关模块
   - 源：`DIBR-GS/dibr/__init__.py`
   - 需要：`DIBRMasked`, `PoseWarpCore`

2. 验证train_cams中有以下属性（应该已有）：
   - `frame` 或 `original_image`（源图像）
   - `depth`（源深度）
   - `metadata` → 内参K
   - `w2c` 或通过camera参数重建

### Phase 2: 在training loop中添加DIBR mask计算

在[train.py主循环中](train.py#L330-L380)，渲染后的循环里：

```python
# 为每个target viewpoint计算DIBR目标视角的动态mask边界
# 需要的输入：源图像, 源深度, 源mask, K, w2c_src=Identity, w2c_target(当前预测)

# 核心操作（伪代码）：
for n_batch, viewpoint_cam in enumerate(viewpoint_cams):
    # 从源视角获取
    src_frame = viewpoint_cam.original_image 
    src_depth = viewpoint_cam.depth
    src_mask = viewpoint_cam.mask
    
    # 计算目标视角DIBR mask
    w2c_tgt = w2c_target[n_batch]  # 当前预测的目标相机参数
    dibr_target_mask = DIBRMasked._warp_dynamic_mask_to_target(
        src_mask, src_depth, K_tensor[n_batch], 
        w2c_src=Identity(3,4),  # 源视角通常为Identity
        w2c_tgt=w2c_tgt,
        backend='posewarp'
    )
    # 结果：dibr_target_mask (H, W) bool
```

### Phase 3: 修改mask loss

在[train.py L468](train.py#L468)处，修改mask loss约束：

**当前代码**：
```python
mask_loss = opt.w_mask * mask_dice_loss(d_alpha_tensor, motion_mask_tensor)
```

**修改为**：
```python
# 使用DIBR目标视角mask而不是源视角GT mask
dibr_target_mask_tensor = torch.cat(dibr_target_masks, 0)  # 从Phase 2收集
mask_loss = opt.w_mask * mask_dice_loss(d_alpha_tensor, dibr_target_mask_tensor.float())
```

### Phase 4: 数据流检查

确保：
1. `dibr_target_mask_tensor` shape: `(B, 1, H, W)` uint8或float → 需要转为float
2. `d_alpha_tensor` shape: `(B, 1, H, W)` float [0, 1] ✓
3. 都在CUDA上 ✓

---

## 具体修改位置

| 位置 | 操作 | 备注 |
|------|------|------|
| [顶部import](train.py#L1-L30) | 添加: `from dibr import DIBRMasked, PoseWarpCore` | 若需要posewarp操作 |
| [Fine stage初始化](train.py#L170) | 如需约束identity矩阵 | `w2c_src = np.eye(3,4)` |
| [主循环-渲染后](train.py#L380-L420) | 添加DIBR mask计算 | 在`for n_batch, viewpoint_cam`循环内 |
| [收集dibr_target_masks](train.py#L417) | 新增list: `dibr_target_masks=[]` | 随motion_masks同时处理 |
| [Tensor聚合](train.py#L438) | 添加: `dibr_target_mask_tensor = torch.cat(...)` | 与motion_mask_tensor同行 |
| [Mask loss](train.py#L468) | 替换: `motion_mask_tensor` → `dibr_target_mask_tensor` | **关键修改** |

---

## 验证步骤

### 1. 代码层面
- ✅ 编译/import无错误
- ✅ dibr_target_mask_tensor shape与d_alpha_tensor对齐
- ✅ CUDA显存不溢出（新增约2-3MB）

### 2. 数值层面  
- 打印log验证mask一致性：
  ```python
  print(f"dibr_mask覆盖率: {dibr_target_mask.float().mean():.3f}")
  print(f"d_alpha覆盖率: {d_alpha_tensor.mean():.3f}")
  ```
  应接近（不要求完全相同，loss会逐步对齐）

### 3. 融合效果
- 生成part1: PoseWarp帧（使用posewarp后端）
- 生成part2: 修改后SplineGS的动态高斯（应该轨迹对齐）
- 融合結果：位置错位应该显著减少

---

## 预期效果

**融合前**（当前）: 动态物体和DIBR留白区位置相差很大 ❌

**融合后**（修改后）: 
- SplineGS学出的高斯运动轨迹与DIBR target mask边界对齐 ✅
- 融合时动态物体自然填充在DIBR的空白区域 ✅
- 边界处理更平滑（mask约束让边界更清晰）✅

---

## 风险与缓解

| 风险 | 原因 | 缓解方案 |
|------|------|---------|
| mask loss过强压制其他优化 | w_mask权重过大 | 从小权重开始(0.01)，逐步增加 |
| DIBR mask边界有holes | 几何occlusion | 用filled DIBR valid_mask作为soft约束 |
| 颜色loss与mask loss冲突 | RGB优化vs mask优化目标不同 | 在warm/fine分阶段加权 |

---

## 其他考虑（可选优化）

### Option A: 加入confidence加权
```python
# DIBR可信度(有效区) 当做soft约束而不是hard约束
loss_mask = mask_dice_loss(d_alpha, dibr_target_mask) * dibr_valid_mask
```

### Option B: 在warm stage不加此约束
当前train.py在warm stage已经用motion_mask了，可跳过warm的修改，仅在fine stage应用DIBR约束。

### Option C: 对称性约束（高级）
若同时有前后帧，可加：
```python
dibr_prev_mask, dibr_next_mask = ...
loss_temporal_consistency = L1(d_alpha_t - warp(d_alpha_prev))
```

---

## 总结

**核心改动**：1行关键修改(mask loss) + ~30行infrastructure code

**时间成本**：~30分钟实现 + 重新训练验证

**预期收益**：解决位置错位，融合质量显著提升
