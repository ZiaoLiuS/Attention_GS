#!/bin/bash

# 定义场景列表
# scenes=("bonsai")
# scenes=("bonsai" "counter" "garden" "kitchen" "room" "stump")
scenes=("bicycle" "bonsai" "counter" "garden" "kitchen" "room" "stump")
# scenes=("counter" "garden" "kitchen" "room" "stump")

# 定义方法名称路径
method_path="none_ours"
# method_path="non_init_only_cross"
# method_path="Non_init_Attention"

# 定义其他参数
image_dir="images_4"
port=1225
ours=0

# 遍历场景并执行训练
for scene in "${scenes[@]}"; do
    python render.py \
        -m "./output/${scene}/${method_path}" 
done
