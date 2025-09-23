import torch
from torch_geometric.utils import negative_sampling
from collections import defaultdict
import random


class Client:
    def __init__(self, client_id, data, encoder, decoder, device='cpu', lr=0.005,
                 weight_decay=1e-4, max_grad_norm=3.0):
        self.client_id = client_id
        self.data = data.to(device)
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

        # 将 loss_weight 改为可训练参数，并用随机值初始化
        self.pos_loss_weight = torch.nn.Parameter(torch.rand(1, device=device))
        self.neg_loss_weight = torch.nn.Parameter(torch.rand(1, device=device))

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            [self.pos_loss_weight, self.neg_loss_weight],  # 将新参数添加到优化器中
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.hard_neg_edges = None
        self.augmented_pos_embeddings = None
        self.augmented_neg_embeddings = None
        self.max_grad_norm = max_grad_norm

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

        z = self.encoder(self.data.x, self.data.edge_index)
        pos_pred = self.decoder(z[pos_edge_index[0]], z[pos_edge_index[1]])
        neg_pred = self.decoder(z[neg_edge_index[0]], z[neg_edge_index[1]])

        labels = torch.cat([
            torch.full((pos_pred.size(0),), 0.9, device=self.device),
            torch.full((neg_pred.size(0),), 0.1, device=self.device)
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
        """增强训练：在增强正边上训练"""
        if self.augmented_pos_embeddings is None:
            return 0.0

        self.encoder.eval()
        self.decoder.train()
        self.optimizer.zero_grad()

        # 原始边部分
        pos_edge_index = self.data.edge_index
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=self.data.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )

        z = self.encoder(self.data.x, self.data.edge_index)
        pos_pred = self.decoder(z[pos_edge_index[0]], z[pos_edge_index[1]])
        neg_pred = self.decoder(z[neg_edge_index[0]], z[neg_edge_index[1]])

        labels = torch.cat([
            torch.full((pos_pred.size(0),), 0.9, device=self.device),
            torch.full((neg_pred.size(0),), 0.1, device=self.device)
        ])
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        loss_ori = self.criterion(pred.squeeze(), labels)

        # 增强正边部分
        z_u_aug, z_v_aug = zip(*self.augmented_pos_embeddings)
        z_u_aug = torch.stack(z_u_aug).to(self.device)
        z_v_aug = torch.stack(z_v_aug).to(self.device)
        pos_pred_aug = self.decoder(z_u_aug, z_v_aug)
        labels_aug = torch.full((pos_pred_aug.size(0),), 0.9, device=self.device)
        loss_aug = self.criterion(pos_pred_aug.squeeze(), labels_aug)

        # 动态调整权重
        current_weight = torch.sigmoid(self.pos_loss_weight)
        loss = (1 - current_weight) * loss_ori + current_weight * loss_aug

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()) + [self.pos_loss_weight],
            self.max_grad_norm
        )
        self.optimizer.step()

        return loss.item()

    def train_on_augmented_negatives(self):
        """增强训练：在注入的跨图负边上训练（仅训练解码器）"""
        if self.augmented_neg_embeddings is None:
            return 0.0

        self.encoder.eval()  # 注意这里不训练编码器
        self.decoder.train()
        self.optimizer.zero_grad()

        # 原始边部分
        pos_edge_index = self.data.edge_index
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=self.data.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )

        z = self.encoder(self.data.x, self.data.edge_index)
        pos_pred = self.decoder(z[pos_edge_index[0]], z[pos_edge_index[1]])
        neg_pred = self.decoder(z[neg_edge_index[0]], z[neg_edge_index[1]])

        labels = torch.cat([
            torch.full((pos_pred.size(0),), 0.9, device=self.device),
            torch.full((neg_pred.size(0),), 0.1, device=self.device)
        ])
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        loss_ori = self.criterion(pred.squeeze(), labels)

        # 增强负边部分
        z_u_aug, z_v_aug = zip(*self.augmented_neg_embeddings)
        z_u_aug = torch.stack(z_u_aug).to(self.device)
        z_v_aug = torch.stack(z_v_aug).to(self.device)

        neg_pred_aug = self.decoder(z_u_aug, z_v_aug)
        labels_aug = torch.full((neg_pred_aug.size(0),), 0.1, device=self.device)
        loss_aug = self.criterion(neg_pred_aug.squeeze(), labels_aug)

        # 动态调整权重
        current_weight = torch.sigmoid(self.neg_loss_weight)
        loss = (1 - current_weight) * loss_ori + current_weight * loss_aug

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()) + [self.neg_loss_weight],
            self.max_grad_norm
        )
        self.optimizer.step()

        return loss.item()

    def evaluate(self, use_test=False):
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            z = self.encoder(self.data.x, self.data.edge_index)

            if use_test:
                pos_edge_index = self.data.test_pos_edge_index
                neg_edge_index = self.data.test_neg_edge_index
            else:
                pos_edge_index = self.data.val_pos_edge_index
                neg_edge_index = self.data.val_neg_edge_index

            pos_pred = self.decoder(z[pos_edge_index[0]], z[pos_edge_index[1]])
            neg_pred = self.decoder(z[neg_edge_index[0]], z[neg_edge_index[1]])

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
        """分析误判边，用于辅助增强注入"""
        self.encoder.eval()
        self.decoder.eval()

        false_negatives = defaultdict(int)
        false_positives = defaultdict(int)

        with torch.no_grad():
            z = self.encoder(self.data.x, self.data.edge_index)

            if use_test:
                pos_edge_index = self.data.test_pos_edge_index
                neg_edge_index = self.data.test_neg_edge_index
            else:
                pos_edge_index = self.data.val_pos_edge_index
                neg_edge_index = self.data.val_neg_edge_index

            pos_pred = self.decoder(z[pos_edge_index[0]], z[pos_edge_index[1]])
            neg_pred = self.decoder(z[neg_edge_index[0]], z[neg_edge_index[1]])

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
        """注入跨图的增强正边（边嵌入形式）"""
        if edge_list:
            self.augmented_pos_embeddings = [
                (other_embeddings[u].detach(), other_embeddings[v].detach())
                for u, v in edge_list
            ]
        else:
            self.augmented_pos_embeddings = None

    def inject_augmented_negative_edges(self, edge_list, other_embeddings):
        """注入跨图的增强负边（边嵌入形式）"""
        if edge_list:
            self.augmented_neg_embeddings = [
                (other_embeddings[u].detach(), other_embeddings[v].detach())
                for u, v in edge_list
            ]
        else:
            self.augmented_neg_embeddings = None

    def clear_augmented_edges(self):
        self.augmented_pos_embeddings = None
        self.augmented_neg_embeddings = None

    def get_encoder_state(self):
        return self.encoder.state_dict()

    def get_decoder_state(self):
        return self.decoder.state_dict()

    def set_encoder_state(self, state_dict):
        self.encoder.load_state_dict(state_dict)

    def set_decoder_state(self, state_dict):
        self.decoder.load_state_dict(state_dict)