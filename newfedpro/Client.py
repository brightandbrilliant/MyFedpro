import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import DataParallel as PyGDataParallel  # <--- 核心修改 1：引入 PyG DataParallel
from collections import defaultdict, OrderedDict
from typing import List, Tuple, Any
import random


class Client:
    def __init__(self, client_id, data, encoder, decoder, device='cpu', lr=0.005,
                 weight_decay=1e-4, max_grad_norm=30000.0,
                 pos_augment_weight=0.1, neg_augment_weight=0.1):
        self.client_id = client_id
        self.data = data
        self.device = device  # 初始设备
        self.encoder = encoder
        self.decoder = decoder

        self.pos_augment_weight = pos_augment_weight
        self.neg_augment_weight = neg_augment_weight

        # 存储优化器参数，在 to() 中初始化
        self.optimizer_params = {
            'lr': lr,
            'weight_decay': weight_decay
        }
        self.optimizer = None

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.augmented_pos_embeddings = None
        self.augmented_neg_embeddings = None
        self.max_grad_norm = max_grad_norm
        self.soft_classify = 1.0

        self.is_dp = False  # 标记是否使用 DataParallel

    # 核心修改 2：to() 方法，使用 PyG DataParallel
    def to(self, devices: List[torch.device] = None) -> 'Client':
        if devices is None or not devices:
            self.device = torch.device('cpu')
            devices = [self.device]

        self.device = devices[0]  # 主设备

        if len(devices) > 1 and all(d.type == 'cuda' for d in devices):
            print(f"Client {self.client_id} 启用 DataParallel，使用设备: {devices}")

            # 使用 PyGDataParallel 包装 GNN Encoder
            self.encoder = PyGDataParallel(self.encoder.to(self.device), device_ids=devices, output_device=self.device)
            # 使用 nn.DataParallel 包装 MLP Decoder (处理简单张量)
            self.decoder = nn.DataParallel(self.decoder.to(self.device), device_ids=devices, output_device=self.device)
            self.is_dp = True
        else:
            self.encoder.to(self.device)
            self.decoder.to(self.device)
            self.is_dp = False

        self.data.to(self.device)

        # 重新初始化优化器 (确保参数在正确设备/包装上)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()),
            **self.optimizer_params
        )
        self.clear_augmented_edges()

        return self

    # 核心修改 3：train() 方法，传入 data 对象给 encoder
    def train(self):
        """常规训练：只使用原始正边和负采样的负边"""
        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()

        pos_edge_index = self.data.edge_index
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=self.data.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )

        # <--- 关键修正 --->
        # 使用 PyG DataParallel 时，传入完整的 data 对象
        z = self.encoder(self.data)

        pos_pred = self.decoder(z[pos_edge_index[0]], z[pos_edge_index[1]])
        neg_pred = self.decoder(z[neg_edge_index[0]], z[neg_edge_index[1]])
        # ... (后续训练逻辑保持不变) ...

        labels = torch.cat([
            torch.full((pos_pred.size(0),), self.soft_classify, device=self.device),
            torch.full((neg_pred.size(0),), 1 - self.soft_classify, device=self.device)
        ])

        pred = torch.cat([pos_pred, neg_pred], dim=0)

        loss = self.criterion(pred.squeeze(), labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            self.max_grad_norm
        )
        self.optimizer.step()

        return loss.item()

    def train_on_augmented_positives(self):
        """增强训练：仅在增强正边上训练 (L = L_aug)"""
        if self.augmented_pos_embeddings is None:
            return 0.0

        self.decoder.train()
        self.optimizer.zero_grad()

        # 增强正边部分 (L_aug)
        z_u_aug, z_v_aug = zip(*self.augmented_pos_embeddings)
        z_u_aug = torch.stack(z_u_aug).to(self.device)
        z_v_aug = torch.stack(z_v_aug).to(self.device)

        # Decoder 预测 (nn.DataParallel 会自动处理)
        pos_pred_aug = self.decoder(z_u_aug, z_v_aug)
        labels_aug = torch.full((pos_pred_aug.size(0),), self.soft_classify, device=self.device)

        loss_aug = self.criterion(pos_pred_aug.squeeze(), labels_aug)
        print(f"Positive loss:{loss_aug}")

        loss = self.pos_augment_weight * loss_aug

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            self.max_grad_norm
        )
        self.optimizer.step()

        return loss.item()

    def train_on_augmented_negatives(self):
        """增强训练：仅在注入的跨图负边上训练 (L = L_aug)"""
        if self.augmented_neg_embeddings is None:
            return 0.0

        self.decoder.train()
        self.optimizer.zero_grad()

        # 增强负边部分 (L_aug)
        z_u_aug, z_v_aug = zip(*self.augmented_neg_embeddings)
        z_u_aug = torch.stack(z_u_aug).to(self.device)
        z_v_aug = torch.stack(z_v_aug).to(self.device)

        neg_pred_aug = self.decoder(z_u_aug, z_v_aug)
        labels_aug = torch.full((neg_pred_aug.size(0),), 1 - self.soft_classify, device=self.device)

        loss_aug = self.criterion(neg_pred_aug.squeeze(), labels_aug)
        print(f"Negative loss:{loss_aug}")

        loss = self.neg_augment_weight * loss_aug

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            self.max_grad_norm
        )
        self.optimizer.step()

        return loss.item()

    def evaluate(self, use_test=False):
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            # <--- 关键修正：传入 data 对象 --->
            z = self.encoder(self.data)

            if use_test:
                pos_edge_index = self.data.test_pos_edge_index
                neg_edge_index = self.data.test_neg_edge_index
            else:
                pos_edge_index = self.data.val_pos_edge_index
                neg_edge_index = self.data.val_neg_edge_index

            pos_pred = self.decoder(z[pos_edge_index[0]], z[pos_edge_index[1]])
            neg_pred = self.decoder(z[neg_edge_index[0]], z[neg_edge_index[1]])

            # ... (后续评估逻辑保持不变) ...

            pred = torch.cat([pos_pred, neg_pred], dim=0).squeeze()
            labels = torch.cat([
                torch.ones(pos_pred.size(0), device=self.device),
                torch.zeros(neg_pred.size(0), device=self.device)
            ]).squeeze()

            pred_label = (torch.sigmoid(pred) > 0.5).float()
            correct = (pred_label == labels).sum().item()
            acc = correct / labels.size(0)

            TP = ((pred_label == 1) & (labels == 1)).sum().item()
            FP = ((pred_label == 1) & (labels == 0)).sum().item()
            FN = ((pred_label == 0) & (labels == 1)).sum().item()

            recall = TP / (TP + FN + 1e-8)
            precision = TP / (TP + FP + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return acc, recall, precision, f1

    def analyze_prediction_errors(self, cluster_labels, use_test=False, top_percent=0.3):
        self.encoder.eval()
        self.decoder.eval()

        false_negatives = defaultdict(int)
        false_positives = defaultdict(int)

        with torch.no_grad():
            # <--- 关键修正：传入 data 对象 --->
            z = self.encoder(self.data)

            if use_test:
                pos_edge_index = self.data.test_pos_edge_index
                neg_edge_index = self.data.test_neg_edge_index
            else:
                pos_edge_index = self.data.val_pos_edge_index
                neg_edge_index = self.data.val_neg_edge_index

            pos_pred = self.decoder(z[pos_edge_index[0]], z[pos_edge_index[1]])
            neg_pred = self.decoder(z[neg_edge_index[0]], z[neg_edge_index[1]])

            # ... (后续分析逻辑保持不变) ...

            pos_pred_label = (torch.sigmoid(pos_pred).squeeze() > 0.5).float()
            neg_pred_label = (torch.sigmoid(neg_pred).squeeze() > 0.5).float()

            fn_mask = (pos_pred_label == 0)
            fp_mask = (neg_pred_label == 1)

            fn_edges = pos_edge_index[:, fn_mask]
            fp_edges = neg_edge_index[:, fp_mask]

            for u, v in fn_edges.t().tolist():
                c1, c2 = cluster_labels[u], cluster_labels[v]
                false_negatives[(c1, c2)] += 1

            for u, v in fp_edges.t().tolist():
                c1, c2 = cluster_labels[u], cluster_labels[v]
                false_positives[(c1, c2)] += 1

        def filter_top_percent(dictionary, top_percent):
            items = list(dictionary.items())
            items.sort(key=lambda x: x[1], reverse=True)
            cutoff = max(1, int(len(items) * top_percent))
            return dict(items[:cutoff])

        return (
            filter_top_percent(false_negatives, top_percent),
            filter_top_percent(false_positives, top_percent)
        )

    def inject_augmented_positive_edges(self, edge_list, other_embeddings):
        # ... (注入逻辑不变) ...
        if edge_list:
            # 确保注入的嵌入最终在正确的设备上
            self.augmented_pos_embeddings = [
                (other_embeddings[u].detach().to(self.device),
                 other_embeddings[v].detach().to(self.device))
                for u, v in edge_list
            ]
        else:
            self.augmented_pos_embeddings = None

    def inject_augmented_negative_edges(self, edge_list, other_embeddings):
        # ... (注入逻辑不变) ...
        if edge_list:
            # 确保注入的嵌入最终在正确的设备上
            self.augmented_neg_embeddings = [
                (other_embeddings[u].detach().to(self.device),
                 other_embeddings[v].detach().to(self.device))
                for u, v in edge_list
            ]
        else:
            self.augmented_neg_embeddings = None

    def clear_augmented_edges(self):
        self.augmented_pos_embeddings = None
        self.augmented_neg_embeddings = None

    # 核心修改 4：处理 DP 的状态字典导出
    def get_encoder_state(self):
        state_dict = self.encoder.state_dict()
        if self.is_dp:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # 移除 'module.'
                new_state_dict[name] = v.cpu()  # 移到 CPU 以便安全聚合
            return new_state_dict
        else:
            return state_dict

    def get_decoder_state(self):
        state_dict = self.decoder.state_dict()
        if self.is_dp:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v.cpu()
            return new_state_dict
        else:
            return state_dict

    # 核心修改 5：处理 DP 的状态字典导入
    def set_encoder_state(self, state_dict):
        if self.is_dp:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict['module.' + k] = v.to(self.device)
                # 检查 load_state_dict 的参数是否正确
            self.encoder.load_state_dict(new_state_dict, strict=True)
        else:
            self.encoder.load_state_dict(state_dict)

    def set_decoder_state(self, state_dict):
        if self.is_dp:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict['module.' + k] = v.to(self.device)
            self.decoder.load_state_dict(new_state_dict, strict=True)
        else:
            self.decoder.load_state_dict(state_dict)