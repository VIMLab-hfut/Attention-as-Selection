# layers/SemanticLabelGenerator.py

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class SemanticLabelGenerator:
    def __init__(self, num_clusters=8):
        self.num_clusters = num_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # 预定义的语义标签名称
        self.cluster_names = [
            "平稳模式", "上升趋势", "下降趋势", "周期波动",
            "尖峰异常", "噪声模式", "复杂混合", "趋势反转"
        ]
    
    def extract_temporal_features(self, time_series_data):
        """从时间序列中提取有意义的特征"""
        # time_series_data: [batch_size, seq_len, features]
        time_series_data = time_series_data.float()

        batch_size, seq_len, features = time_series_data.shape
        
        features_list = []
        for i in range(batch_size):
            series = time_series_data[i]  # [seq_len, features]
            
            # 基本统计特征
            mean_val = series.mean(dim=0)
            std_val = series.std(dim=0)
            trend = torch.diff(series, dim=0).mean(dim=0)  # 趋势
            
            # 简单的时间序列特征
            abs_max = torch.abs(series).max(dim=0).values
            energy = (series ** 2).mean(dim=0)
            
            # 组合所有特征
            combined_features = torch.cat([mean_val, std_val, trend, abs_max, energy])
            features_list.append(combined_features)
        
        return torch.stack(features_list)  # [batch_size, feature_dim]
    
    def fit(self, time_series_data):
        """拟合聚类器"""
        print(f"Fitting semantic clusters with {time_series_data.shape[0]} samples...")
        
        # 提取特征
        features = self.extract_temporal_features(time_series_data)
        features_np = features.cpu().numpy().astype(np.float32)
        
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features_np)
        
        # K-means聚类
        self.kmeans = KMeans(
            n_clusters=self.num_clusters, 
            random_state=42,
            n_init=10
        )
        self.kmeans.fit(features_scaled)
        
        self.is_fitted = True
        print("Semantic clusters fitted successfully!")
        
        # 打印聚类大小
        cluster_counts = np.bincount(self.kmeans.labels_)
        for i, count in enumerate(cluster_counts):
            print(f"Cluster {i} ({self.cluster_names[i]}): {count} samples")
    
    def predict(self, time_series_batch):
        """预测批次数据的语义标签"""
        if not self.is_fitted:
            # 返回随机标签作为fallback
            return torch.randint(0, self.num_clusters, 
                               (time_series_batch.shape[0],), 
                               device=time_series_batch.device)
         # 确保输入数据为float32
        time_series_batch = time_series_batch.float()
        # 提取特征
        features = self.extract_temporal_features(time_series_batch)
        features_np = features.detach().cpu().numpy().astype(np.float32)  # 明确指定为float32
        
        # 标准化并预测
        features_scaled = self.scaler.transform(features_np)
        labels = self.kmeans.predict(features_scaled)
        
        return torch.tensor(labels, dtype=torch.long, device=time_series_batch.device)
    
    def get_cluster_name(self, cluster_id):
        """获取聚类名称"""
        if 0 <= cluster_id < len(self.cluster_names):
            return self.cluster_names[cluster_id]
        return f"Cluster_{cluster_id}"