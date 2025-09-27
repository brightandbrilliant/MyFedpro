import os
import torch
from collections import deque, defaultdict
from Client import Client
import random
from Model.GraphSage import GraphSAGE
from Model.ResMLP import ResMLP
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected
from Cluster import (
    kmeans_cluster_single,
    compute_anchor_feature_differences,
    build_cluster_cooccurrence_matrix,
    extract_clear_alignments
)
from Parse_Anchors import read_anchors, parse_anchors
from Utils import (build_positive_edge_dict,
                   build_edge_type_alignment, judge_loss_window,
                   draw_loss_plot)


def split_client_data(data, val_ratio=0.1, test_ratio=0.1, device='cpu'):
    data = data.to(device)
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        neg_sampling_ratio=1.0
    )
    train_data, val_data, test_data = transform(data)

    val_mask = val_data.edge_label.bool()
    test_mask = test_data.edge_label.bool()
    train_data.val_pos_edge_index = val_data.edge_label_index[:, val_mask]
    train_data.val_neg_edge_index = val_data.edge_label_index[:, ~val_mask]
    train_data.test_pos_edge_index = test_data.edge_label_index[:, test_mask]
    train_data.test_neg_edge_index = test_data.edge_label_index[:, ~test_mask]

    return train_data


def load_all_clients(pyg_data_paths, encoder_params, decoder_params, training_params, device, nClusters=10, enhance_interval=5):
    clients, all_cluster_labels, raw_data_list, edge_dicts = [], [], [], []

    for client_id, path in enumerate(pyg_data_paths):
        raw_data = torch.load(path)
        raw_data_list.append(raw_data)
        data = split_client_data(raw_data)

        cluster_labels, _ = kmeans_cluster_single(data, n_clusters=nClusters)
        all_cluster_labels.append(cluster_labels)

        edge_dict = build_positive_edge_dict(data, cluster_labels)
        edge_dicts.append(edge_dict)

        encoder = GraphSAGE(**encoder_params)
        decoder = ResMLP(input_dim=encoder_params['output_dim'] * 2, **decoder_params)

        client = Client(
            client_id=client_id,
            data=data,
            encoder=encoder,
            decoder=decoder,
            device=device,
            lr=training_params['lr'],
            weight_decay=training_params['weight_decay']
        )
        clients.append(client)

    return clients, all_cluster_labels, raw_data_list, edge_dicts


def average_state_dicts(state_dicts):
    avg_state = {}
    for key in state_dicts[0].keys():
        avg_state[key] = torch.stack([sd[key] for sd in state_dicts], dim=0).mean(dim=0)
    return avg_state


def extract_augmented_pos_edges(target_fn_types, edge_dict, edge_alignment, top_k=100):
    selected_edges = []
    for (c1, c2) in target_fn_types:
        aligned_targets = edge_alignment.get((c1, c2), [])
        for (c1_p, c2_p), weight in aligned_targets:
            candidate_edges = edge_dict.get((c1_p, c2_p), [])
            selected_edges.extend(candidate_edges[:top_k])
    return selected_edges


def construct_augmented_neg_edges(aggregated_fp, alignment, cluster_labels_j, pos_edges_j, top_k=100):
    """
    基于类型对齐和负属性保证，构造增强负边列表，并确保无重复。

    Args:
        aggregated_fp (dict): 客户端 i 的困难假正边类型 (c1, c2) -> 计数。
        alignment (dict): 客户端 i 到客户端 j 的边类型对齐关系。
        cluster_labels_j (torch.Tensor): 客户端 j 的节点聚类标签。
        pos_edges_j (set): 客户端 j 的所有正边集合 (用于排除，保证负属性)。
        top_k (int): 每种对齐类型目标采样的最大边数。 <--- 已修正参数名

    Returns:
        list: 增强负边 [(u, v), ...] 列表。
    """
    neg_edge_list = []
    # 用于存储和查重已采样的负边集合
    sampled_neg_edges = set()
    MAX_ATTEMPTS = 500

    for (c1_i, c2_i) in aggregated_fp:
        aligned_targets = alignment.get((c1_i, c2_i), [])

        for (c1_j, c2_j), weight in aligned_targets:
            nodes_c1 = (cluster_labels_j == c1_j).nonzero(as_tuple=True)[0].tolist()
            nodes_c2 = (cluster_labels_j == c2_j).nonzero(as_tuple=True)[0].tolist()

            if not nodes_c1 or not nodes_c2:
                continue

            sampled_count = 0
            attempts = 0

            # 注意：这里使用修正后的参数名 top_k
            target_samples = int(top_k * weight) if len(aligned_targets) > 1 else top_k

            while sampled_count < target_samples and attempts < MAX_ATTEMPTS:
                u = random.choice(nodes_c1)
                v = random.choice(nodes_c2)

                edge_tuple = (int(u), int(v))  # 有向边元组

                # 检查条件：
                # 1. u != v
                # 2. edge_tuple 不在客户端 j 的正边集合中（排除正边，只检查当前方向）
                # 3. edge_tuple 不在本次已采样的集合中（排除重复）
                if (u != v and
                        edge_tuple not in pos_edges_j and  # <--- 修正：不再检查 (v, u)
                        edge_tuple not in sampled_neg_edges):
                    neg_edge_list.append(edge_tuple)
                    sampled_neg_edges.add(edge_tuple)
                    sampled_count += 1

                attempts += 1

    return neg_edge_list



def evaluate_all_clients(clients, cluster_labels, use_test=False):
    metrics = []
    for i, client in enumerate(clients):
        acc, recall, precision, f1 = client.evaluate(use_test=use_test)
        metrics.append((acc, recall, precision, f1))
        print(f"Client {client.client_id}: Acc={acc:.4f}, Recall={recall:.4f}, Prec={precision:.4f}, F1={f1:.4f}")
    avg = torch.tensor(metrics).mean(dim=0).tolist()
    print(f"\n===> Average: Acc={avg[0]:.4f}, Recall={avg[1]:.4f}, Prec={avg[2]:.4f}, F1={avg[3]:.4f}")
    return avg


def aggregate_from_window(sliding_window, top_percent=0.3):
    aggregate = defaultdict(int)
    for it in sliding_window:
        for pair, count in it.items():
            aggregate[pair] += count
    sorted_items = sorted(aggregate.items(), key=lambda x: x[1], reverse=True)
    cutoff = max(1, int(len(sorted_items) * top_percent))
    return dict(sorted_items[:cutoff])


if __name__ == "__main__":
    data_dir = "../Parsed_dataset/dblp"
    anchor_path = "../dataset/dblp/anchors.txt"
    pyg_data_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")])

    encoder_params = {
        'input_dim': torch.load(pyg_data_files[0]).x.shape[1],
        'hidden_dim': 128,
        'output_dim': 64,
        'num_layers': 3,
        'dropout': 0.5
    }
    decoder_params = {'hidden_dim': 128, 'num_layers': 8, 'dropout': 0.3}
    training_params = {'lr': 0.001, 'weight_decay': 1e-4, 'local_epochs': 5}

    num_rounds = 800
    top_fp_fn_percent = 0.3
    enhance_interval = 20
    top_k_per_type = 100
    nClusters = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clients, cluster_labels, raw_data_list, edge_dicts = load_all_clients(
        pyg_data_files, encoder_params, decoder_params, training_params, device, nClusters, enhance_interval
    )

    client_pos_edges = [
        set(map(tuple, clients[k].data.edge_index.t().tolist())) for k in range(len(clients))
    ]

    anchor_raw = read_anchors(anchor_path)
    anchor_pairs = parse_anchors(anchor_raw, point=9714)
    results = compute_anchor_feature_differences(raw_data_list[0], raw_data_list[1], anchor_pairs)
    co_matrix = build_cluster_cooccurrence_matrix(cluster_labels[0], cluster_labels[1], results, nClusters, top_percent=0.75)
    alignment1 = extract_clear_alignments(co_matrix, min_ratio=0.25, min_count=30, mode=1)
    alignment2 = extract_clear_alignments(co_matrix, min_ratio=0.25, min_count=30, mode=2)
    edge_alignment1 = build_edge_type_alignment(alignment1, nClusters)
    edge_alignment2 = build_edge_type_alignment(alignment2, nClusters)

    best_f1 = -1
    best_encoder_state = None
    best_decoder_states = None
    best_weight_state = None

    # 初始化滑动窗口
    sliding_fn_window = [deque(maxlen=5) for _ in range(len(clients))]
    sliding_fp_window = [deque(maxlen=5) for _ in range(len(clients))]
    sliding_loss_window = [deque(maxlen=20) for _ in range(len(clients))]
    loss_record = [[],[]]
    augment_flag = [False, False]
    rnds = [-1, -1]
    last_diff = [10000,10000] # 设一个很大的值
    fn_fp_ignore_flags = [False, False]

    print("\n================ Federated Training Start ================")
    for rnd in range(1, num_rounds + 1):
        print(f"\n--- Round {rnd} ---")

        z_others = [client.encoder(client.data.x, client.data.edge_index).detach() for client in clients]

        for i, client in enumerate(clients):
            if fn_fp_ignore_flags[i] is False:
                fn, fp = client.analyze_prediction_errors(cluster_labels[i], use_test=False, top_percent=top_fp_fn_percent)
                sliding_fn_window[i].append(fn)
                sliding_fp_window[i].append(fp)
            else:
                fn_fp_ignore_flags[i] = False

            if rnd >= 150 and rnd % 5 == 0 and augment_flag[i] is False:
                augment_flag[i], last_diff[i] = judge_loss_window(sliding_loss_window[i], last_diff[i])
                if augment_flag[i] is True:
                    rnds[i] = rnd

            if augment_flag[i] is True and rnd % enhance_interval == 0:
                aggregated_fn = aggregate_from_window(sliding_fn_window[i], top_percent=top_fp_fn_percent)
                aggregated_fp = aggregate_from_window(sliding_fp_window[i], top_percent=top_fp_fn_percent)


                j = 1 - i
                pos_edge_list = extract_augmented_pos_edges(
                    aggregated_fn,
                    edge_dicts[j],
                    edge_alignment1 if i == 0 else edge_alignment2,
                    top_k=top_k_per_type
                )

                neg_edge_list = construct_augmented_neg_edges(
                    aggregated_fp,
                    edge_alignment1 if i == 0 else edge_alignment2,           # 客户端 i 到 j 的对齐矩阵
                    cluster_labels[j],   # 客户端 j 的聚类标签
                    client_pos_edges[j], # 客户端 j 的正边集合 (用于排除)
                    top_k=top_k_per_type
                )

                client.inject_augmented_positive_edges(pos_edge_list, z_others[j])
                client.inject_augmented_negative_edges(neg_edge_list, z_others[j])

        for i, client in enumerate(clients):
            loss_avg = 0
            for _ in range(training_params['local_epochs']):
                loss = client.train()
                loss_avg += loss

            if augment_flag[i] is True and rnd % enhance_interval == 0:
                print("Negative Augmentation Implementing.")
                client.train_on_augmented_negatives()
                fn_fp_ignore_flags[i] = True
            if augment_flag[i] is True and rnd % enhance_interval == enhance_interval/2:
                print("Positive Augmentation Implementing.")
                client.train_on_augmented_positives()
                fn_fp_ignore_flags[i] = True

            loss_avg /= training_params['local_epochs']
            sliding_loss_window[i].append(loss_avg)
            loss_record[i].append(loss_avg)
            print(f'Client{i} loss: {loss_avg}')

        encoder_states = [client.get_encoder_state() for client in clients]
        decoder_states = [client.get_decoder_state() for client in clients]
        weight_states = [client.get_loss_weight_state() for client in clients]
        global_encoder_state = average_state_dicts(encoder_states)
        global_weight_state = average_state_dicts(weight_states)

        for client in clients:
            client.set_encoder_state(global_encoder_state)
            client.set_loss_weight_state(global_weight_state)

        avg_acc, avg_recall, avg_prec, avg_f1 = evaluate_all_clients(clients, cluster_labels, use_test=False)

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_encoder_state = global_encoder_state
            best_decoder_states = decoder_states
            best_weight_state = global_weight_state
            print("===> New best model saved")

    print("\n================ Federated Training Finished ================")
    for i, client in enumerate(clients):
        client.set_encoder_state(best_encoder_state)
        client.set_loss_weight_state(best_weight_state)
        client.set_decoder_state(best_decoder_states[i])

    print("\n================ Final Evaluation ================")
    evaluate_all_clients(clients, cluster_labels, use_test=True)
    # draw_loss_plot(loss_record[0])
    # draw_loss_plot(loss_record[1])

