from collections import defaultdict, deque
import matplotlib.pyplot as plt
import numpy as np
import torch

def build_edge_type_alignment(alignment, nClusters):
    """
    保留边的方向，构建从源边类型 (i,j) 到目标边类型 (i',j') 的映射。
    返回: dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]
    """
    edge_mapping = {}

    for i in range(nClusters):
        aligned_i = alignment.get(i, [])  # [(j, score)]
        for j in range(nClusters):
            aligned_j = alignment.get(j, [])  # 保留顺序，不做排序
            key = (i, j)  # 有向边类型

            mapped = []
            for (i_, si) in aligned_i:
                for (j_, sj) in aligned_j:
                    target_key = (i_, j_)
                    weight = si * sj
                    mapped.append((target_key, weight))

            # 合并相同 target_key 的权重
            merged = {}
            for k, w in mapped:
                merged[k] = merged.get(k, 0) + w
            mapped_list = [(k, w) for k, w in merged.items()]
            edge_mapping[key] = mapped_list

    return edge_mapping



def build_positive_edge_dict(data, cluster_labels):
    edge_dict = defaultdict(list)
    edge_index = data.edge_index
    for u, v in edge_index.t().tolist():
        c1, c2 = cluster_labels[u], cluster_labels[v]
        key = (c1, c2)
        edge_dict[key].append((u, v))
    return edge_dict


# 返回 True 则进入增强期，返回 False 则不进入增强期
def judge_loss_window(loss_window: deque, last_diff: float):
    diff_window = []
    for i in range(1, len(loss_window)):
        diff_window.append(abs(loss_window[i]-loss_window[i-1]))
    now_diff = np.mean(diff_window)
    if abs(now_diff - last_diff)/last_diff <= 0.1:
        return True, now_diff
    return False, now_diff

def draw_loss_plot(loss_record: list):
    x = []
    for i in range(1, len(loss_record)+1):
        x.append(i)
    plt.plot(x, loss_record, marker='o', linestyle='-', color='b', label='Client')

    plt.title('Loss-Round', fontsize=16)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.show()

