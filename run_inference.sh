#!/bin/bash
# Quick inference runner for train_aligned.py outputs
# Usage: ./run_inference.sh [checkpoint_path] [data_path] [expname]

set -e

# Default values
CHECKPOINT="${1:-.output/balloon1_aligned/point_cloud/iteration_5000}"
DATA_PATH="${2:../DIBR示例/Pose-Warping/Data/balloon1}"
EXPNAME="${3:-test_aligned_output}"

# Verify checkpoint exists
if [ ! -d "$CHECKPOINT" ]; then
    echo "❌ Checkpoint directory not found: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$CHECKPOINT/point_cloud.ply" ]; then
    echo "❌ point_cloud.ply not found in: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$CHECKPOINT/point_cloud_static.ply" ]; then
    echo "❌ point_cloud_static.ply not found in: $CHECKPOINT"
    exit 1
fi

# Verify data exists
if [ ! -d "$DATA_PATH" ]; then
    echo "❌ Dataset path not found: $DATA_PATH"
    exit 1
fi

echo "✅ Configuration:"
echo "   Checkpoint: $CHECKPOINT"
echo "   Dataset: $DATA_PATH"
echo "   Output: output/$EXPNAME"
echo ""

# Run inference
echo "🎬 Starting inference..."
python test_aligned_inference.py \
    --checkpoint "$CHECKPOINT" \
    --source_path "$DATA_PATH" \
    --expname "$EXPNAME" \
    --render_test \
    --render_train

echo ""
echo "✅ Inference complete!"
echo "📁 Results saved to: output/$EXPNAME/"
echo "📹 Videos:"
ls -lh output/$EXPNAME/*.mp4 2>/dev/null || echo "   (No videos found)"
echo "🖼️  Frames:"
echo "   - Test:  output/$EXPNAME/renders_test/ ($(ls output/$EXPNAME/renders_test/*.png 2>/dev/null | wc -l) files)"
echo "   - Train: output/$EXPNAME/renders_train/ ($(ls output/$EXPNAME/renders_train/*.png 2>/dev/null | wc -l) files)"
