_base_ = "./default.py"

ModelParams = dict(
    source_path="data/nvidia_rodynrf/Balloon1",
    depth_type="disp",
)

OptimizationParams = dict(
    densify_grad_threshold_dynamic = 0.0002,
    densify_grad_threshold = 0.0008,
    use_instance_mask=True,
) 