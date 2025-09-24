#!/bin/bash

RUN_COUNT=10
export CUDA_VISIBLE_DEVICES=0

echo "脚本开始执行，总共将运行 $RUN_COUNT 次。"
echo "---------------------------------------------------"

for i in $(seq 1 $RUN_COUNT)
do
    # 打印提示语，提示当前是第几次执行
    echo "--- 开始执行第 $i 次 ---"

    # 执行你的 Python 脚本，只输出到终端
    python main.py | tail -n 10

    # 打印分隔线，方便区分每次的输出
    echo "--- 第 $i 次执行结束 ---"
    echo ""
done

echo "---------------------------------------------------"
echo "所有任务已完成！"