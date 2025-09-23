import torch
from torch_geometric.data import Data
import numpy as np  # 导入 numpy 以便处理特征向量


def inspect_pyg_data_file(file_path: str):
    """
    读取一个 PyTorch Geometric Data 文件 (.pt)，并输出其包含的关键信息。

    Args:
        file_path (str): PyG Data 文件的完整路径。
    """
    print(f"--- 正在检查文件: {file_path} ---")

    try:
        data = torch.load(file_path)
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。请检查路径是否正确。")
        return
    except Exception as e:
        print(f"加载文件 '{file_path}' 时发生错误: {e}")
        return

    if not isinstance(data, Data):
        print(f"错误: 加载的文件 '{file_path}' 不是一个有效的 PyTorch Geometric Data 对象。")
        return

    print("\n--- PyG Data 对象信息概览 ---")
    print(f"文件路径: {file_path}")
    print(f"是否是 PyG Data 对象: {isinstance(data, Data)}")

    # 1. 节点数量和边数量
    print(f"节点数量 (data.num_nodes): {data.num_nodes}")
    print(
        f"边数量 (data.num_edges / 2 if bidirectional): {data.num_edges / 2 if data.num_edges % 2 == 0 else data.num_edges} (若为无向图，通常是双向存储，实际边数需除以2)")
    print(f"原始边对数量 (data.num_edges): {data.num_edges} (内部存储的边对数量，包含双向边)")

    # 2. 节点特征 (x)
    if hasattr(data, 'x') and data.x is not None:
        print(f"\n--- 节点特征 (x) ---")
        print(f"是否存在节点特征 (data.x): 是")
        print(f"节点特征张量形状 (data.x.shape): {data.x.shape}")
        if data.x.shape[0] > 0:  # 确保有节点
            # 获取第一个节点的特征向量。注意：这是 PyG 内部索引为0的节点。
            # 如果你的节点是全局索引，这个索引0的节点可能是任意原始ID的节点。
            first_node_feature = data.x[0].numpy()  # 转换为 NumPy 数组以便打印
            print(f"第一个节点特征向量维度: {first_node_feature.shape[0]}")
            print(f"第一个节点特征向量（前5个值）: {first_node_feature[:min(5, first_node_feature.shape[0])]}")
            print(f"特征向量的 dtype: {data.x.dtype}")
        else:
            print("  节点特征张量为空 (可能没有节点或特征)。")
    else:
        print(f"\n--- 节点特征 (x) ---")
        print(f"是否存在节点特征 (data.x): 否")

    # 3. 节点标签 (y)
    if hasattr(data, 'y') and data.y is not None:
        print(f"\n--- 节点标签 (y) ---")
        print(f"是否存在节点标签 (data.y): 是")
        print(f"节点标签张量形状 (data.y.shape): {data.y.shape}")
        if data.y.numel() > 0:
            print(f"前5个节点标签: {data.y[:min(5, data.y.numel())].tolist()}")
        else:
            print("  节点标签张量为空。")
    else:
        print(f"\n--- 节点标签 (y) ---")
        print(f"是否存在节点标签 (data.y): 否")

    # 4. 边索引 (edge_index)
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        print(f"\n--- 边索引 (edge_index) ---")
        print(f"是否存在边索引 (data.edge_index): 是")
        print(f"边索引张量形状 (data.edge_index.shape): {data.edge_index.shape}")
        if data.edge_index.numel() > 0:
            print(f"前3条边对 (全局内部索引): \n{data.edge_index[:, :min(3, data.edge_index.shape[1])].tolist()}")
        else:
            print("  边索引张量为空。")
    else:
        print(f"\n--- 边索引 (edge_index) ---")
        print(f"是否存在边索引 (data.edge_index): 否")

    # 5. 边特征 (edge_attr)
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        print(f"\n--- 边特征 (edge_attr) ---")
        print(f"是否存在边特征 (data.edge_attr): 是")
        print(f"边特征张量形状 (data.edge_attr.shape): {data.edge_attr.shape}")
        if data.edge_attr.numel() > 0:
            print(f"前3条边特征: \n{data.edge_attr[:min(3, data.edge_attr.shape[0])].tolist()}")
        else:
            print("  边特征张量为空。")
    else:
        print(f"\n--- 边特征 (edge_attr) ---")
        print(f"是否存在边特征 (data.edge_attr): 否")

    # 6. 其他可能的属性 (批量处理、掩码等)
    print(f"\n--- 其他 PyG Data 属性 ---")
    if hasattr(data, 'pos') and data.pos is not None:
        print(f"是否存在节点位置 (data.pos): 是 (形状: {data.pos.shape})")
    else:
        print(f"是否存在节点位置 (data.pos): 否")

    if hasattr(data, 'face') and data.face is not None:
        print(f"是否存在面 (data.face): 是 (形状: {data.face.shape})")
    else:
        print(f"是否存在面 (data.face): 否")

    if hasattr(data, 'train_mask') and data.train_mask is not None:
        print(f"是否存在训练掩码 (data.train_mask): 是")
    else:
        print(f"是否存在训练掩码 (data.train_mask): 否")

    if hasattr(data, 'val_mask') and data.val_mask is not None:
        print(f"是否存在验证掩码 (data.val_mask): 是")
    else:
        print(f"是否存在验证掩码 (data.val_mask): 否")

    if hasattr(data, 'test_mask') and data.test_mask is not None:
        print(f"是否存在测试掩码 (data.test_mask): 是")
    else:
        print(f"是否存在测试掩码 (data.test_mask): 否")

    print("\n--- 检查完毕 ---")


if __name__ == "__main__":
    data_path = "../Parsed_dataset/wd/wd_1.pt"
    inspect_pyg_data_file(data_path)
