"""
特征工程模块
处理原始数据，生成模型所需的特征
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


class FeatureEngineer:
    """特征工程处理器"""
    
    def __init__(self, data):
        """
        初始化特征工程器
        
        Parameters:
        -----------
        data : DataFrame
            原始数据
        """
        self.raw_data = data.copy()
        self.data = data.copy()
        self.scaler = StandardScaler()
        self.feature_cols = []
        
    def create_interaction_features(self):
        """
        创建交互特征
        考虑变量间的相互作用
        """
        print("\n[特征工程] 创建交互特征...")
        
        # 关键特征
        features_to_interact = {
            'Temp_C': '温度',
            'C_in_gNm3': '入口浓度',
            'Q_Nm3h': '烟气流量',
            'U_avg_kV': '平均电压',
            'T_avg_s': '平均振打周期'
        }
        
        # 创建交互项：入口浓度 × 电压
        if 'C_in_gNm3' in self.data.columns and 'U_avg_kV' in self.data.columns:
            self.data['C_in_U_interaction'] = self.data['C_in_gNm3'] * self.data['U_avg_kV']
            print("  ✓ 入口浓度 × 电压")
        
        # 创建交互项：温度 × 入口浓度
        if 'Temp_C' in self.data.columns and 'C_in_gNm3' in self.data.columns:
            self.data['Temp_C_in_interaction'] = self.data['Temp_C'] * self.data['C_in_gNm3']
            print("  ✓ 温度 × 入口浓度")
        
        # 创建交互项：振打周期 × 入口浓度
        if 'T_avg_s' in self.data.columns and 'C_in_gNm3' in self.data.columns:
            self.data['T_C_in_interaction'] = self.data['T_avg_s'] * self.data['C_in_gNm3']
            print("  ✓ 振打周期 × 入口浓度")
        
        # 创建交互项：电压 × 振打周期
        if 'U_avg_kV' in self.data.columns and 'T_avg_s' in self.data.columns:
            self.data['U_T_interaction'] = self.data['U_avg_kV'] * self.data['T_avg_s']
            print("  ✓ 电压 × 振打周期")
        
        return self
    
    def create_polynomial_features(self, degree=2):
        """
        创建多项式特征
        
        Parameters:
        -----------
        degree : int
            多项式次数
        """
        print(f"\n[特征工程] 创建{degree}次多项式特征...")
        
        base_features = ['Temp_C', 'C_in_gNm3', 'Q_Nm3h', 'U_avg_kV', 'T_avg_s']
        available_features = [f for f in base_features if f in self.data.columns]
        
        # 手工创建平方项
        for feat in available_features:
            self.data[f'{feat}_squared'] = self.data[feat] ** 2
            print(f"  ✓ {feat}²")
        
        return self
    
    def create_ratio_features(self):
        """创建比例特征"""
        print("\n[特征工程] 创建比例特征...")
        
        # 温度与流量比
        if 'Temp_C' in self.data.columns and 'Q_Nm3h' in self.data.columns:
            self.data['Temp_Q_ratio'] = self.data['Temp_C'] / (self.data['Q_Nm3h'] + 1e-6)
            print("  ✓ 温度/流量比")
        
        # 入口浓度与电压比（除尘难度与努力程度）
        if 'C_in_gNm3' in self.data.columns and 'U_avg_kV' in self.data.columns:
            self.data['C_in_U_ratio'] = self.data['C_in_gNm3'] / (self.data['U_avg_kV'] + 1e-6)
            print("  ✓ 入口浓度/电压比")
        
        # 振打频率（1/周期）
        if 'T_avg_s' in self.data.columns:
            self.data['T_freq'] = 1.0 / (self.data['T_avg_s'] + 1e-6)
            print("  ✓ 振打频率")
        
        return self
    
    def create_lagged_features(self, lag=5):
        """
        创建滞后特征（时间序列特征）
        
        Parameters:
        -----------
        lag : int
            滞后步数
        """
        print(f"\n[特征工程] 创建{lag}步滞后特征...")
        
        lag_features = ['C_in_gNm3', 'Temp_C', 'C_out_mgNm3', 'U_avg_kV', 'T_avg_s']
        available_features = [f for f in lag_features if f in self.data.columns]
        
        for feat in available_features:
            for i in range(1, lag + 1):
                self.data[f'{feat}_lag_{i}'] = self.data[feat].shift(i)
            print(f"  ✓ {feat} (lag 1-{lag})")
        
        # 删除NaN行
        self.data = self.data.dropna()
        print(f"  ✓ 删除NaN行，剩余数据：{len(self.data)}")
        
        return self
    
    def create_rolling_features(self, window=10):
        """
        创建滚动窗口特征
        
        Parameters:
        -----------
        window : int
            滚动窗口大小
        """
        print(f"\n[特征工程] 创建{window}步滚动特征...")
        
        roll_features = ['C_in_gNm3', 'Temp_C', 'P_total_kW']
        available_features = [f for f in roll_features if f in self.data.columns]
        
        for feat in available_features:
            self.data[f'{feat}_roll_mean'] = self.data[feat].rolling(window=window).mean()
            self.data[f'{feat}_roll_std'] = self.data[feat].rolling(window=window).std()
            print(f"  ✓ {feat} (rolling mean & std)")
        
        # 删除NaN行
        self.data = self.data.dropna()
        print(f"  ✓ 删除NaN行，剩余数据：{len(self.data)}")
        
        return self
    
    def normalize_features(self, features=None):
        """
        标准化特征
        
        Parameters:
        -----------
        features : list
            要标准化的特征列表，如果为None则标准化所有数值特征
        """
        print("\n[特征工程] 标准化特征...")
        
        if features is None:
            features = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 排除目标变量和时间戳
        features = [f for f in features if f not in ['C_out_mgNm3', 'P_total_kW', 'timestamp']]
        
        self.data[features] = self.scaler.fit_transform(self.data[features])
        print(f"  ✓ 标准化{len(features)}个特征")
        
        return self
    
    def get_feature_list(self):
        """获取所有特征列表"""
        excluded = ['timestamp', 'C_out_mgNm3', 'P_total_kW']
        features = [col for col in self.data.columns 
                   if col not in excluded and self.data[col].dtype in [np.float64, np.int64]]
        return features
    
    def get_data(self):
        """返回特征工程后的数据"""
        return self.data
    
    def execute_full_pipeline(self, use_lagged=False, use_rolling=False):
        """
        执行完整的特征工程流程
        
        Parameters:
        -----------
        use_lagged : bool
            是否使用滞后特征
        use_rolling : bool
            是否使用滚动特征
        """
        print("\n" + "="*60)
        print("执行完整特征工程流程")
        print("="*60)
        
        self.create_interaction_features()
        self.create_polynomial_features(degree=2)
        self.create_ratio_features()
        
        if use_lagged:
            self.create_lagged_features(lag=5)
        
        if use_rolling:
            self.create_rolling_features(window=10)
        
        print(f"\n✓ 特征工程完成")
        print(f"  原始特征数：{len(self.raw_data.columns)}")
        print(f"  最终特征数：{len(self.data.columns)}")
        print(f"  数据行数：{len(self.data)}")
        
        return self


if __name__ == "__main__":
    # 使用示例
    from data_loader import load_data
    
    data = load_data('data/Cement_ESP_Data.csv')
    
    fe = FeatureEngineer(data)
    fe.execute_full_pipeline(use_lagged=False, use_rolling=False)
    
    processed_data = fe.get_data()
    print("\n" + processed_data.head())
