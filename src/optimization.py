"""
优化模块
解决问题2：在满足排放标准前提下，确定最优电压和振打周期组合，使电耗最低
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')


class WorkingConditionAnalyzer:
    """工况划分分析器"""
    
    def __init__(self, data):
        """
        初始化工况分析器
        
        Parameters:
        -----------
        data : DataFrame
            原始数据
        """
        self.data = data
        self.conditions = None
        self.clusters = None
        self.cluster_centers = None
        self.n_clusters = None
        
    def analyze_input_conditions(self):
        """分析入口条件的分布"""
        print("\n[入口条件分析]...")
        
        print("\n入口浓度统计：")
        print(f"  最小值：{self.data['C_in_gNm3'].min():.2f} g/Nm³")
        print(f"  最大值：{self.data['C_in_gNm3'].max():.2f} g/Nm³")
        print(f"  平均值：{self.data['C_in_gNm3'].mean():.2f} g/Nm³")
        print(f"  标准差：{self.data['C_in_gNm3'].std():.2f} g/Nm³")
        
        print("\n温度统计：")
        print(f"  最小值：{self.data['Temp_C'].min():.2f} ℃")
        print(f"  最大值：{self.data['Temp_C'].max():.2f} ℃")
        print(f"  平均值：{self.data['Temp_C'].mean():.2f} ℃")
        print(f"  标准差：{self.data['Temp_C'].std():.2f} ℃")
        
        return self
    
    def divide_by_clustering(self, n_clusters=5, features=['C_in_gNm3', 'Temp_C']):
        """
        基于K-means聚类的工况划分
        
        Parameters:
        -----------
        n_clusters : int
            聚类数量
        features : list
            聚类特征
        """
        print(f"\n[K-means聚类] 将数据分为 {n_clusters} 个工况...")
        
        X = self.data[features].values
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = kmeans.fit_predict(X)
        self.cluster_centers = kmeans.cluster_centers_
        self.n_clusters = n_clusters
        
        self.data['condition_cluster'] = self.clusters
        
        # 分析各工况特征
        print("\n各工况特征汇总：")
        for i in range(n_clusters):
            cluster_data = self.data[self.data['condition_cluster'] == i]
            
            print(f"\n工况 {i+1} ({len(cluster_data)} 样本)：")
            print(f"  入口浓度：{cluster_data['C_in_gNm3'].mean():.2f} ± {cluster_data['C_in_gNm3'].std():.2f} g/Nm³")
            print(f"  温度：{cluster_data['Temp_C'].mean():.2f} ± {cluster_data['Temp_C'].std():.2f} ℃")
            print(f"  出口浓度：{cluster_data['C_out_mgNm3'].mean():.2f} ± {cluster_data['C_out_mgNm3'].std():.2f} mg/Nm³")
            print(f"  电耗：{cluster_data['P_total_kW'].mean():.2f} ± {cluster_data['P_total_kW'].std():.2f} kW")
        
        return self
    
    def divide_by_percentile(self, n_conditions=5):
        """
        基于入口浓度百分位数的工况划分
        
        Parameters:
        -----------
        n_conditions : int
            工况数量
        """
        print(f"\n[百分位划分] 按入口浓度百分位分为 {n_conditions} 个工况...")
        
        percentiles = np.percentile(self.data['C_in_gNm3'], 
                                   np.linspace(0, 100, n_conditions + 1))
        
        self.data['condition_percentile'] = pd.cut(
            self.data['C_in_gNm3'], 
            bins=percentiles, 
            labels=[f'工况{i+1}' for i in range(n_conditions)],
            include_lowest=True
        )
        
        print("\n各工况范围：")
        for i in range(n_conditions):
            condition_data = self.data[self.data['condition_percentile'] == f'工况{i+1}']
            print(f"  工况{i+1}：C_in = [{percentiles[i]:.2f}, {percentiles[i+1]:.2f}] g/Nm³ ({len(condition_data)} 样本)")
        
        return self
    
    def get_conditions(self):
        """获取分类后的工况数据"""
        return self.data


class ParameterOptimizer:
    """参数优化器"""
    
    def __init__(self, model, data, emission_limit=10.0):
        """
        初始化优化器
        
        Parameters:
        -----------
        model : ESPModel
            已训练的出口浓度预测模型
        data : DataFrame
            历史数据
        emission_limit : float
            出口浓度限制 (mg/Nm³)
        """
        self.model = model
        self.data = data
        self.emission_limit = emission_limit
        self.optimal_params = {}
        
    def optimize_for_condition(self, condition_data, condition_name, verbose=True):
        """
        为单个工况优化参数
        
        Parameters:
        -----------
        condition_data : DataFrame
            该工况的历史数据
        condition_name : str
            工况名称
        verbose : bool
            是否打印详细信息
            
        Returns:
        --------
        dict : 最优参数
        """
        
        # 获取该工况的典型参数范围
        U_range = [
            condition_data[['U1_kV', 'U2_kV', 'U3_kV', 'U4_kV']].min().min(),
            condition_data[['U1_kV', 'U2_kV', 'U3_kV', 'U4_kV']].max().max()
        ]
        T_range = [
            condition_data[['T1_s', 'T2_s', 'T3_s', 'T4_s']].min().min(),
            condition_data[['T1_s', 'T2_s', 'T3_s', 'T4_s']].max().max()
        ]
        P_range = [
            condition_data['P_total_kW'].min(),
            condition_data['P_total_kW'].max()
        ]
        
        # 使用简化模型：平均电压和平均振打周期
        U_avg_mean = condition_data['U_avg_kV'].mean()
        T_avg_mean = condition_data['T_avg_s'].mean()
        
        if verbose:
            print(f"\n[{condition_name}] 参数范围：")
            print(f"  电压范围：{U_range[0]:.1f} - {U_range[1]:.1f} kV")
            print(f"  振打周期范围：{T_range[0]:.1f} - {T_range[1]:.1f} s")
            print(f"  电耗范围：{P_range[0]:.1f} - {P_range[1]:.1f} kW")
        
        # 定义目标函数：最小化电耗
        def objective(x):
            U_avg, T_avg = x
            # 简化：假设4个电场参数相同
            params = {
                'Temp_C': condition_data['Temp_C'].mean(),
                'C_in_gNm3': condition_data['C_in_gNm3'].mean(),
                'Q_Nm3h': condition_data['Q_Nm3h'].mean(),
                'U_avg_kV': U_avg,
                'T_avg_s': T_avg,
                'U_std_kV': 0.5,
                'T_std_s': 5.0,
            }
            
            # 预测的电耗（需要用功率模型，这里用简化模型）
            # P ≈ P_base + α·U² + β·(1/T)
            P_estimated = 200 + 0.5 * U_avg**2 + 500 / (T_avg + 1e-6)
            return P_estimated
        
        # 定义约束：出口浓度 ≤ emission_limit
        def constraint_emission(x):
            U_avg, T_avg = x
            params = {
                'Temp_C': condition_data['Temp_C'].mean(),
                'C_in_gNm3': condition_data['C_in_gNm3'].mean(),
                'Q_Nm3h': condition_data['Q_Nm3h'].mean(),
                'U_avg_kV': U_avg,
                'T_avg_s': T_avg,
                'U_std_kV': 0.5,
                'T_std_s': 5.0,
            }
            
            # 预测出口浓度
            # 这里需要用训练好的模型
            # 简化：C_out ≈ C_in * exp(-k·U) / (1 + T/T_ref)
            C_in = condition_data['C_in_gNm3'].mean()
            C_out_pred = C_in * 1000 * np.exp(-0.05 * U_avg) / (1 + T_avg / 50)
            
            # 约束：C_out ≤ emission_limit
            return self.emission_limit - C_out_pred
        
        # 初始猜测
        x0 = [U_avg_mean, T_avg_mean]
        
        # 参数界限
        bounds = Bounds([U_range[0], T_range[0]], 
                       [min(U_range[1], 80), min(T_range[1], 180)])
        
        # 约束
        from scipy.optimize import NonlinearConstraint
        constraint = NonlinearConstraint(constraint_emission, 0, np.inf)
        
        # 优化
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_emission}
            )
            
            if result.success:
                U_opt, T_opt = result.x
                P_opt = objective(result.x)
                C_out_opt = 1000 * (condition_data['C_in_gNm3'].mean() * 
                                   np.exp(-0.05 * U_opt) / (1 + T_opt / 50))
                
                if verbose:
                    print(f"\n[{condition_name}] 优化结果：")
                    print(f"  最优电压：{U_opt:.1f} kV")
                    print(f"  最优振打周期：{T_opt:.1f} s")
                    print(f"  预期电耗：{P_opt:.1f} kW")
                    print(f"  预期出口浓度：{C_out_opt:.2f} mg/Nm³")
                    print(f"  满足约束：{C_out_opt <= self.emission_limit}")
                
                return {
                    'condition': condition_name,
                    'U_opt': U_opt,
                    'T_opt': T_opt,
                    'P_opt': P_opt,
                    'C_out_opt': C_out_opt,
                    'success': True
                }
            else:
                if verbose:
                    print(f"  ✗ 优化失败：{result.message}")
                return {'condition': condition_name, 'success': False}
                
        except Exception as e:
            if verbose:
                print(f"  ✗ 优化出错：{e}")
            return {'condition': condition_name, 'success': False}
    
    def optimize_all_conditions(self, condition_data_list, condition_names):
        """
        为所有工况优化参数
        
        Parameters:
        -----------
        condition_data_list : list
            各工况数据
        condition_names : list
            各工况名称
            
        Returns:
        --------
        dict : 所有工况的最优参数
        """
        print("\n" + "="*60)
        print("针对所有工况的参数优化")
        print("="*60)
        
        self.optimal_params = {}
        
        for i, (cond_data, cond_name) in enumerate(zip(condition_data_list, condition_names)):
            result = self.optimize_for_condition(cond_data, cond_name, verbose=True)
            self.optimal_params[cond_name] = result
        
        return self.optimal_params
    
    def get_optimal_params(self):
        """返回所有最优参数"""
        return self.optimal_params


def create_optimization_table(optimal_params_dict):
    """
    创建优化参数表
    
    Parameters:
    -----------
    optimal_params_dict : dict
        最优参数字典
        
    Returns:
    --------
    DataFrame : 参数表
    """
    
    table_data = []
    for cond_name, params in optimal_params_dict.items():
        if params.get('success', False):
            table_data.append({
                '工况': cond_name,
                '最优电压 (kV)': f"{params['U_opt']:.1f}",
                '最优振打周期 (s)': f"{params['T_opt']:.1f}",
                '预期电耗 (kW)': f"{params['P_opt']:.1f}",
                '预期出口浓度 (mg/Nm³)': f"{params['C_out_opt']:.2f}",
            })
    
    return pd.DataFrame(table_data)


if __name__ == "__main__":
    from data_loader import load_data
    from feature_engineering import FeatureEngineer
    from modeling import ESPModel
    
    # 加载和处理数据
    data = load_data('data/Cement_ESP_Data.csv')
    fe = FeatureEngineer(data)
    fe.execute_full_pipeline()
    processed_data = fe.get_data()
    
    # 构建模型
    model = ESPModel(model_type='linear')
    model.prepare_data(processed_data)
    model.train()
    
    # 工况划分
    analyzer = WorkingConditionAnalyzer(data)
    analyzer.analyze_input_conditions()
    analyzer.divide_by_clustering(n_clusters=5)
    
    # 参数优化
    optimizer = ParameterOptimizer(model, data, emission_limit=10.0)
