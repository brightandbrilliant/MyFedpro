import os

def read_unknown_file(file_path, max_lines=20):
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    print(f"正在读取文件: {file_path}")

    try:
        # 以文本方式尝试读取
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            print(f"\n🔍 文件前 {max_lines} 行内容如下（按文本方式读取）:\n")
            for i in range(max_lines):
                line = f.readline()
                if not line:
                    break
                print(f"{i+1:02d}: {repr(line.strip())}")
        print("\n✅ 读取成功，文件是文本格式。你可以尝试按行 split 或用 pandas 读取。")
    except UnicodeDecodeError:
        print("\n⚠️ 文件不是文本格式，可能是二进制文件。你可以尝试使用以下方式解析：")
        print("- 尝试 `pickle.load()`")
        print("- 尝试 `torch.load()`")
        print("- 尝试 `np.load()`")
        print("- 或用 `open(file_path, 'rb')` 查看原始字节")


if __name__ == "__main__":
    """
    file_path = 'networks'  # 没有后缀也可以
    import pickle

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    print("读取成功，类型是：", type(data))
    if isinstance(data, tuple):
        for i, item in enumerate(data):
            if hasattr(item, 'nodes') and hasattr(item, 'edges'):
                print(f"第 {i} 个元素是一个可能的 networkx 图：{type(item)}")
                G = item
                break
        else:
            print("⚠️ 元组中没有发现 networkx 图对象。")
            G = None
    else:
        G = data if hasattr(data, 'nodes') and hasattr(data, 'edges') else None

    if G:
        import networkx as nx

        print(f"成功提取图，节点数: {G.number_of_nodes()}，边数: {G.number_of_edges()}")
        # 可视化、操作图
        nx.draw(G, with_labels=True)
    else:
        print("❌ 无法从数据中提取 networkx 图。")
    """

    file_path = '../dataset/wd/stop_words_cn.pkl'  # 没有后缀也可以
    read_unknown_file(file_path, max_lines=10)

