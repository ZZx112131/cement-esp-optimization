"""
数据加载和预处理模块
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class DataLoader:
    """电除尘器数据加载器"""
    
    def __init__(self, filepath):
        """
        初始化数据加载器
        
        Parameters:
        -----------
        filepath : str
            CSV数据文件路径
        """
        self.filepath = filepath
        self.raw_data = None
        self.data = None
        
    def load(self):
        """加载原始数据"""
        try:
            self.raw_data = pd.read_csv(self.filepath)
            print(f"✓ 数据加载成功，共 {len(self.raw_data)} 行")
            print(f"✓ 字段列表：{list(self.raw_data.columns)}")
            return self.raw_data
        except FileNotFoundError:
            print(f"✗ 文件不存在：{self.filepath}")
            return None
        except Exception as e:
            print(f"✗ 加载失败：{e}")
            return None
    
    def preprocess(self):
        """
        数据预处理
        - 处理缺失值
        - 异常值检测与处理
        - 时间戳转换
        - 特征提取
        """
        if self.raw_data is None:
            print("✗ 请先调用 load() 加载数据")
            return None
        
        self.data = self.raw_data.copy()
        
        # 1. 缺失值处理
        print("\n[1] 缺失值处理...")
        missing_counts = self.data.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"  发现缺失值：\n{missing_counts[missing_counts > 0]}")
            self.data = self.data.dropna()
            print(f"  删除缺失行后，数据量：{len(self.data)}")
        else:
            print("  ✓ 无缺失值")
        
        # 2. 异常值检测（使用3σ原则）
        print("\n[2] 异常值检测...")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        outlier_indices = set()
        for col in numeric_cols:
            mean = self.data[col].mean()
            std = self.data[col].std()
            outliers = np.abs(self.data[col] - mean) > 3 * std
            outlier_indices.update(self.data[outliers].index)
            if outliers.sum() > 0:
                print(f"  {col}: {outliers.sum()} 个异常值")
        
        if len(outlier_indices) > 0:
            print(f"  删除异常值{len(outlier_indices)}行")
            self.data = self.data.drop(outlier_indices)
        else:
            print("  ✓ 无异常值")
        
        # 3. 时间戳处理
        print("\n[3] 时间处理...")
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data = self.data.sort_values('timestamp').reset_index(drop=True)
            print(f"  时间范围：{self.data['timestamp'].min()} ~ {self.data['timestamp'].max()}")
        
        # 4. 特征工程基础：计算平均值
        print("\n[4] 特征提取...")
        
        # 电压平均值
        if all(f'U{i}_kV' in self.data.columns for i in range(1, 5)):
            self.data['U_avg_kV'] = self.data[['U1_kV', 'U2_kV', 'U3_kV', 'U4_kV']].mean(axis=1)
            self.data['U_std_kV'] = self.data[['U1_kV', 'U2_kV', 'U3_kV', 'U4_kV']].std(axis=1)
            print("  ✓ 电压特征：U_avg, U_std")
        
        # 振打周期平均值
        if all(f'T{i}_s' in self.data.columns for i in range(1, 5)):
            self.data['T_avg_s'] = self.data[['T1_s', 'T2_s', 'T3_s', 'T4_s']].mean(axis=1)
            self.data['T_std_s'] = self.data[['T1_s', 'T2_s', 'T3_s', 'T4_s']].std(axis=1)
            print("  ✓ 振打周期特征：T_avg, T_std")
        
        # 数据质量统计
        print(f"\n✓ 预处理完成，最终数据量：{len(self.data)}")
        print(f"  特征数：{len(self.data.columns)}")
        
        return self.data
    
    def get_statistics(self):
        """获取数据统计信息"""
        if self.data is None:
            print("✗ 请先调用 preprocess() 处理数据")
            return None
        
        print("\n" + "="*80)
        print("数据统计信息")
        print("="*80)
        print(self.data.describe().to_string())
        
        return self.data.describe()
    
    def get_data(self):
        """返回预处理后的数据"""
        return self.data
    
    def split_by_time(self, n_splits=7):
        """
        按时间均匀分割数据
        
        Parameters:
        -----------
        n_splits : int
            分割数量
            
        Returns:
        --------
        list : 分割后的数据片段
        """
        if self.data is None:
            return None
        
        if 'timestamp' not in self.data.columns:
            print("✗ 数据中没有 timestamp 列")
            return None
        
        time_splits = []
        n = len(self.data)
        split_size = n // n_splits
        
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size if i < n_splits - 1 else n
            split_data = self.data.iloc[start_idx:end_idx].copy()
            time_splits.append(split_data)
        
        print(f"✓ 数据按时间分为 {len(time_splits)} 段")
        for i, chunk in enumerate(time_splits):
            print(f"  段{i+1}：{len(chunk)} 行")
        
        return time_splits


def load_data(filepath):
    """
    便利函数：加载并预处理数据
    
    Parameters:
    -----------
    filepath : str
        CSV数据文件路径
        
    Returns:
    --------
    DataFrame : 预处理后的数据
    """
    loader = DataLoader(filepath)
    loader.load()
    loader.preprocess()
    return loader.get_data()


if __name__ == "__main__":
    # 使用示例
    loader = DataLoader('data/Cement_ESP_Data.csv')
    loader.load()
    data = loader.preprocess()
    loader.get_statistics()
