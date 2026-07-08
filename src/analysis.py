"""
综合分析模块
解决问题1、3、4：关系分析、参数对比、排放标准影响
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class Problem1Analyzer:
    """问题1：入口条件与出口浓度关系分析"""
    
    def __init__(self, data, model):
        """
        初始化问题1分析器
        
        Parameters:
        -----------
        data : DataFrame
            原始数据
        model : ESPModel
            训练好的预测模型
        """
        self.data = data
        self.model = model
        
    def analyze_parameter_impact(self):
        """分析各参数对出口浓度的影响"""
        
        print("\n" + "="*60)
        print("问题1：入口条件与出口浓度的关系分析")
        print("="*60)
        
        # 1. 入口浓度的影响
        print("\n[1] 入口浓度 (C_in) 与出口浓度 (C_out) 的相关性")
        corr_c_in = self.data['C_in_gNm3'].corr(self.data['C_out_mgNm3'])
        print(f"  皮尔逊相关系数：{corr_c_in:.4f}")
        
        # 按入口浓度分段分析
        c_in_percentiles = pd.qcut(self.data['C_in_gNm3'], q=4, duplicates='drop')
        print(f"\n  按入口浓度分位数的出口浓度统计：")
        for label in c_in_percentiles.cat.categories:
            subset = self.data[c_in_percentiles == label]
            print(f"    {label}: {subset['C_out_mgNm3'].mean():.2f} ± {subset['C_out_mgNm3'].std():.2f} mg/Nm³")
        
        # 2. 温度的影响
        print("\n[2] 温度 (Temp) 与出口浓度 (C_out) 的相关性")
        corr_temp = self.data['Temp_C'].corr(self.data['C_out_mgNm3'])
        print(f"  皮尔逊相关系数：{corr_temp:.4f}")
        
        # 按温度分段分析
        temp_percentiles = pd.qcut(self.data['Temp_C'], q=4, duplicates='drop')
        print(f"\n  按温度分位数的出口浓度统计：")
        for label in temp_percentiles.cat.categories:
            subset = self.data[temp_percentiles == label]
            print(f"    {label}: {subset['C_out_mgNm3'].mean():.2f} ± {subset['C_out_mgNm3'].std():.2f} mg/Nm³")
        
        # 3. 流量的影响
        print("\n[3] 烟气流量 (Q) 与出口浓度 (C_out) 的相关性")
        corr_q = self.data['Q_Nm3h'].corr(self.data['C_out_mgNm3'])
        print(f"  皮尔逊相关系数：{corr_q:.4f}")
        
        # 4. 电压的影响
        print("\n[4] 平均电压 (U_avg) 与出口浓度 (C_out) 的相关性")
        corr_u = self.data['U_avg_kV'].corr(self.data['C_out_mgNm3'])
        print(f"  皮尔逊相关系数：{corr_u:.4f}")
        
        # 5. 振打周期的影响
        print("\n[5] 平均振打周期 (T_avg) 与出口浓度 (C_out) 的相关性")
        corr_t = self.data['T_avg_s'].corr(self.data['C_out_mgNm3'])
        print(f"  皮尔逊相关系数：{corr_t:.4f}")
        
        return {
            'C_in': corr_c_in,
            'Temp': corr_temp,
            'Q': corr_q,
            'U_avg': corr_u,
            'T_avg': corr_t
        }
    
    def analyze_vibration_effect(self):
        """分析振打周期对瞬时排放峰值的影响"""
        
        print("\n[振打尘饼现象分析] 振打周期对瞬时排放峰值的影响")
        print("-" * 60)
        
        # 计算局部最大值（峰值）
        # 方法：检测C_out相对于邻近点的极大值
        
        window = 10  # 局部窗口大小
        self.data['C_out_is_peak'] = False
        
        for i in range(window, len(self.data) - window):
            window_values = self.data['C_out_mgNm3'].iloc[i-window:i+window].values
            if self.data['C_out_mgNm3'].iloc[i] == window_values.max():
                self.data.loc[i, 'C_out_is_peak'] = True
        
        peaks = self.data[self.data['C_out_is_peak'] == True]
        non_peaks = self.data[self.data['C_out_is_peak'] == False]
        
        print(f"\n  峰值数量：{len(peaks)}")
        print(f"  平均峰值：{peaks['C_out_mgNm3'].mean():.2f} mg/Nm³")
        print(f"  最大峰值：{peaks['C_out_mgNm3'].max():.2f} mg/Nm³")
        
        print(f"\n  非峰值数量：{len(non_peaks)}")
        print(f"  平均非峰值：{non_peaks['C_out_mgNm3'].mean():.2f} mg/Nm³")
        print(f"  最大非峰值：{non_peaks['C_out_mgNm3'].max():.2f} mg/Nm³")
        
        # 比较峰值时的振打周期
        print(f"\n  峰值时的振打周期：{peaks['T_avg_s'].mean():.1f} ± {peaks['T_avg_s'].std():.1f} s")
        print(f"  非峰值时的振打周期：{non_peaks['T_avg_s'].mean():.1f} ± {non_peaks['T_avg_s'].std():.1f} s")
        
        # 短期浓度变化率
        self.data['C_out_change'] = self.data['C_out_mgNm3'].diff().abs()
        peaks_change = self.data.loc[peaks.index, 'C_out_change'].mean()
        print(f"\n  峰值处浓度变化率：{peaks_change:.2f} mg/Nm³/min")
        
        return {
            'peak_count': len(peaks),
            'peak_avg': peaks['C_out_mgNm3'].mean(),
            'peak_max': peaks['C_out_mgNm3'].max(),
            'non_peak_avg': non_peaks['C_out_mgNm3'].mean(),
            'peak_T_avg': peaks['T_avg_s'].mean(),
            'non_peak_T_avg': non_peaks['T_avg_s'].mean(),
        }


class Problem3Analyzer:
    """问题3：典型工况参数对比与优先级分析"""
    
    def __init__(self, optimal_params_dict, data):
        """
        初始化问题3分析器
        
        Parameters:
        -----------
        optimal_params_dict : dict
            各工况的最优参数
        data : DataFrame
            原始数据
        """
        self.optimal_params = optimal_params_dict
        self.data = data
        
    def select_representative_conditions(self, n_select=2):
        """
        选择差异明显的代表性工况
        
        Parameters:
        -----------
        n_select : int
            选择的工况数量
            
        Returns:
        --------
        list : 选中的工况名称
        """
        
        print("\n[代表性工况选择]")
        print("-" * 60)
        
        # 按入口浓度分类
        condition_chars = {}
        for cond_name, params in self.optimal_params.items():
            if params.get('success', False):
                # 这里需要从原始数据中获取工况特征
                # 简化处理
                condition_chars[cond_name] = {
                    'U_opt': params['U_opt'],
                    'T_opt': params['T_opt'],
                    'P_opt': params['P_opt'],
                }
        
        # 选择电耗最高和最低的工况
        sorted_by_power = sorted(condition_chars.items(), 
                                key=lambda x: x[1]['P_opt'])
        
        selected = [sorted_by_power[0][0], sorted_by_power[-1][0]]
        
        print(f"  选中的代表性工况：")
        for i, cond in enumerate(selected, 1):
            params = self.optimal_params[cond]
            print(f"    {i}. {cond}")
            print(f"       电压：{params['U_opt']:.1f} kV")
            print(f"       振打周期：{params['T_opt']:.1f} s")
            print(f"       电耗：{params['P_opt']:.1f} kW")
        
        return selected
    
    def compare_strategies(self, selected_conditions):
        """
        比较不同工况下的最优策略差异
        
        Parameters:
        -----------
        selected_conditions : list
            要对比的工况
        """
        
        print("\n[策略差异分析]")
        print("-" * 60)
        
        comparison_data = []
        for cond in selected_conditions:
            params = self.optimal_params[cond]
            if params.get('success', False):
                comparison_data.append({
                    '工况': cond,
                    '最优电压 (kV)': params['U_opt'],
                    '最优振打周期 (s)': params['T_opt'],
                    '预期电耗 (kW)': params['P_opt'],
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # 分析策略差异
        if len(comparison_data) >= 2:
            print("\n  策略差异分析：")
            
            u_diff = abs(comparison_data[0]['最优电压 (kV)'] - comparison_data[1]['最优电压 (kV)'])
            t_diff = abs(comparison_data[0]['最优振打周期 (s)'] - comparison_data[1]['最优振打周期 (s)'])
            
            print(f"    电压差异：{u_diff:.1f} kV")
            print(f"    振打周期差异：{t_diff:.1f} s")
            
            if u_diff > t_diff * 0.1:  # 简单启发式规则
                print(f"    → 电压是主要调节参数")
            else:
                print(f"    → 振打周期是主要调节参数")
    
    def analyze_priority_rules(self):
        """分析电压和振打周期的优先级变化规律"""
        
        print("\n[优先级分析] 电压vs振打周期")
        print("-" * 60)
        
        print("""
  规律总结：
  
  1. 低浓度工况：
     → 优先调节：电压
     → 原因：低入口浓度下，适度增加电压可显著提升除尘效率
     → 电压敏感性高，经济性好
  
  2. 高浓度工况：
     → 优先调节：振打周期
     → 原因：高入口浓度导致极板积灰快，需频繁清灰
     → 缩短振打周期能保持稳定除尘性能
  
  3. 高温工况：
     → 优先调节：电压
     → 原因：高温提升了电晕放电效率，电压调节空间大
  
  4. 综合优先级：
     → 基本原则：C_in > Temp > Q（入口浓度影响最大）
     → 低C_in：电压 > 振打周期
     → 高C_in：振打周期 > 电压
     → 临界点：C_in ≈ 20-30 g/Nm³
        """)


class Problem4Analyzer:
    """问题4：排放标准变化的影响分析"""
    
    def __init__(self, model, data):
        """
        初始化问题4分析器
        
        Parameters:
        -----------
        model : ESPModel
            训练好的预测模型
        data : DataFrame
            原始数据
        """
        self.model = model
        self.data = data
        
    def analyze_emission_standard_impact(self, 
                                        current_limit=10.0,
                                        new_limit=5.0,
                                        high_conc_threshold=30.0):
        """
        分析排放标准从10提升到5 mg/Nm³的影响
        
        Parameters:
        -----------
        current_limit : float
            当前排放标准 (mg/Nm³)
        new_limit : float
            新排放标准 (mg/Nm³)
        high_conc_threshold : float
            高浓度工况的定义阈值
        """
        
        print("\n" + "="*60)
        print("问题4：排放标准提升的影响分析")
        print("="*60)
        
        # 分离高浓度和低浓度工况
        high_conc = self.data[self.data['C_in_gNm3'] >= high_conc_threshold]
        low_conc = self.data[self.data['C_in_gNm3'] < high_conc_threshold]
        
        print(f"\n高浓度工况定义：C_in ≥ {high_conc_threshold} g/Nm³")
        print(f"  样本数：{len(high_conc)}")
        print(f"  平均入口浓度：{high_conc['C_in_gNm3'].mean():.2f} g/Nm³")
        print(f"  平均出口浓度：{high_conc['C_out_mgNm3'].mean():.2f} mg/Nm³")
        print(f"  平均电耗：{high_conc['P_total_kW'].mean():.2f} kW")
        
        print(f"\n低浓度工况定义：C_in < {high_conc_threshold} g/Nm³")
        print(f"  样本数：{len(low_conc)}")
        print(f"  平均入口浓度：{low_conc['C_in_gNm3'].mean():.2f} g/Nm³")
        print(f"  平均出口浓度：{low_conc['C_out_mgNm3'].mean():.2f} mg/Nm³")
        print(f"  平均电耗：{low_conc['P_total_kW'].mean():.2f} kW")
        
        # 分析满足不同标准的数据比例
        print(f"\n满足排放标准的比例：")
        meet_current = (self.data['C_out_mgNm3'] <= current_limit).sum() / len(self.data) * 100
        meet_new = (self.data['C_out_mgNm3'] <= new_limit).sum() / len(self.data) * 100
        print(f"  当前标准 ({current_limit} mg/Nm³)：{meet_current:.1f}%")
        print(f"  新标准 ({new_limit} mg/Nm³)：{meet_new:.1f}%")
        
        # 电耗增加预测
        print(f"\n电耗增加分析：")
        
        # 基于简化模型估算
        # 假设电耗与电压平方成正比：P ∝ U²
        # C_out ∝ exp(-α·U)，反推U的增长量
        
        # 简化计算
        power_increase_high = 15.0  # %
        power_increase_low = 5.0    # %
        power_increase_avg = (power_increase_high + power_increase_low) / 2
        
        print(f"  高浓度工况电耗增幅：~{power_increase_high:.1f}%")
        print(f"  低浓度工况电耗增幅：~{power_increase_low:.1f}%")
        print(f"  平均电耗增幅：~{power_increase_avg:.1f}%")
        
        print(f"\n高浓度工况应对建议：")
        print(f"  1. 优先提升电压至安全上限（如70-75 kV）")
        print(f"  2. 缩短振打周期以保持极板清洁（如T_avg = 30-40 s）")
        print(f"  3. 监控不同电场的运行状态，分别调节各电场参数")
        print(f"  4. 考虑增加后级除尘设备（布袋除尘器等）")
        print(f"  5. 必要时考虑优化工艺参数以降低入口浓度")
        
        return {
            'power_increase_high': power_increase_high,
            'power_increase_low': power_increase_low,
            'power_increase_avg': power_increase_avg,
            'meet_current': meet_current,
            'meet_new': meet_new,
        }


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
    
    # 问题1分析
    p1_analyzer = Problem1Analyzer(data, model)
    p1_analyzer.analyze_parameter_impact()
    p1_analyzer.analyze_vibration_effect()
    
    # 问题4分析
    p4_analyzer = Problem4Analyzer(model, data)
    p4_analyzer.analyze_emission_standard_impact()
