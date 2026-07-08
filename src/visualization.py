"""
可视化模块
生成各类分析图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class Visualizer:
    """数据可视化类"""
    
    def __init__(self, data):
        """
        初始化可视化器
        
        Parameters:
        -----------
        data : DataFrame
            原始数据
        """
        self.data = data
        
    def plot_correlation_heatmap(self, save_path=None):
        """绘制相关性热力图"""
        
        print("\n[可视化] 绘制相关性热力图...")
        
        # 选择关键特征
        features = ['Temp_C', 'C_in_gNm3', 'Q_Nm3h', 'U_avg_kV', 
                   'T_avg_s', 'P_total_kW', 'C_out_mgNm3']
        corr_matrix = self.data[features].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', square=True, cbar_kws={'label': '相关系数'})
        plt.title('关键参数相关性热力图', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ 图表已保存：{save_path}")
        
        plt.show()
        
    def plot_input_vs_output(self, save_path=None):
        """绘制入口条件vs出口浓度的关系"""
        
        print("\n[可视化] 绘制入口条件vs出口浓度...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 入口浓度vs出口浓度
        axes[0, 0].scatter(self.data['C_in_gNm3'], self.data['C_out_mgNm3'], 
                          alpha=0.5, s=20)
        axes[0, 0].set_xlabel('入口浓度 (g/Nm³)', fontsize=11)
        axes[0, 0].set_ylabel('出口浓度 (mg/Nm³)', fontsize=11)
        axes[0, 0].set_title('入口浓度 vs 出口浓度', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 温度vs出口浓度
        axes[0, 1].scatter(self.data['Temp_C'], self.data['C_out_mgNm3'], 
                          alpha=0.5, s=20, color='orange')
        axes[0, 1].set_xlabel('烟气温度 (℃)', fontsize=11)
        axes[0, 1].set_ylabel('出口浓度 (mg/Nm³)', fontsize=11)
        axes[0, 1].set_title('温度 vs 出口浓度', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 电压vs出口浓度
        axes[1, 0].scatter(self.data['U_avg_kV'], self.data['C_out_mgNm3'], 
                          alpha=0.5, s=20, color='green')
        axes[1, 0].set_xlabel('平均电压 (kV)', fontsize=11)
        axes[1, 0].set_ylabel('出口浓度 (mg/Nm³)', fontsize=11)
        axes[1, 0].set_title('电压 vs 出口浓度', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 振打周期vs出口浓度
        axes[1, 1].scatter(self.data['T_avg_s'], self.data['C_out_mgNm3'], 
                          alpha=0.5, s=20, color='red')
        axes[1, 1].set_xlabel('平均振打周期 (s)', fontsize=11)
        axes[1, 1].set_ylabel('出口浓度 (mg/Nm³)', fontsize=11)
        axes[1, 1].set_title('振打周期 vs 出口浓度', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ 图表已保存：{save_path}")
        
        plt.show()
    
    def plot_time_series(self, save_path=None):
        """绘制时间序列数据"""
        
        print("\n[可视化] 绘制时间序列数据...")
        
        if 'timestamp' not in self.data.columns:
            print("  ✗ 数据中没有时间戳")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 1. 入口条件时序
        ax1_1 = axes[0]
        ax1_2 = ax1_1.twinx()
        
        ax1_1.plot(self.data.index, self.data['C_in_gNm3'], 'b-', label='入口浓度', linewidth=1.5)
        ax1_2.plot(self.data.index, self.data['Temp_C'], 'r-', label='烟气温度', linewidth=1.5)
        
        ax1_1.set_ylabel('入口浓度 (g/Nm³)', color='b', fontsize=11)
        ax1_2.set_ylabel('温度 (℃)', color='r', fontsize=11)
        ax1_1.set_title('入口工况参数时序变化', fontsize=12, fontweight='bold')
        ax1_1.tick_params(axis='y', labelcolor='b')
        ax1_2.tick_params(axis='y', labelcolor='r')
        ax1_1.grid(True, alpha=0.3)
        
        # 2. 出口浓度时序
        axes[1].plot(self.data.index, self.data['C_out_mgNm3'], 'g-', linewidth=1.5)
        axes[1].axhline(y=10, color='orange', linestyle='--', linewidth=2, label='超低排放标准 (10 mg/Nm³)')
        axes[1].axhline(y=5, color='red', linestyle='--', linewidth=2, label='新标准 (5 mg/Nm³)')
        axes[1].set_ylabel('出口浓度 (mg/Nm³)', fontsize=11)
        axes[1].set_title('出口粉尘浓度时序变化', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # 3. 电耗时序
        axes[2].plot(self.data.index, self.data['P_total_kW'], 'purple', linewidth=1.5)
        axes[2].set_ylabel('总电耗 (kW)', fontsize=11)
        axes[2].set_xlabel('时间点', fontsize=11)
        axes[2].set_title('除尘电耗时序变化', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ 图表已保存：{save_path}")
        
        plt.show()
    
    def plot_parameter_distribution(self, save_path=None):
        """绘制操作参数分布"""
        
        print("\n[可视化] 绘制操作参数分布...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 电压分布
        axes[0, 0].hist(self.data['U_avg_kV'], bins=30, color='blue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('平均电压 (kV)', fontsize=11)
        axes[0, 0].set_ylabel('频数', fontsize=11)
        axes[0, 0].set_title('电压分布', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. 振打周期分布
        axes[0, 1].hist(self.data['T_avg_s'], bins=30, color='green', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('平均振打周期 (s)', fontsize=11)
        axes[0, 1].set_ylabel('频数', fontsize=11)
        axes[0, 1].set_title('振打周期分布', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. 电耗分布
        axes[1, 0].hist(self.data['P_total_kW'], bins=30, color='red', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('总电耗 (kW)', fontsize=11)
        axes[1, 0].set_ylabel('频数', fontsize=11)
        axes[1, 0].set_title('电耗分布', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. 出口浓度分布
        axes[1, 1].hist(self.data['C_out_mgNm3'], bins=30, color='orange', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=10, color='red', linestyle='--', linewidth=2, label='标准限值')
        axes[1, 1].set_xlabel('出口浓度 (mg/Nm³)', fontsize=11)
        axes[1, 1].set_ylabel('频数', fontsize=11)
        axes[1, 1].set_title('出口浓度分布', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ 图表已保存：{save_path}")
        
        plt.show()
    
    def plot_efficiency_analysis(self, save_path=None):
        """绘制除尘效率分析"""
        
        print("\n[可视化] 绘制除尘效率分析...")
        
        # 计算除尘效率
        # η = (C_in - C_out) / C_in * 100%
        # 注意单位：C_in (g/Nm³ = 1000 mg/Nm³), C_out (mg/Nm³)
        self.data['efficiency'] = (self.data['C_in_gNm3'] * 1000 - self.data['C_out_mgNm3']) / \
                                 (self.data['C_in_gNm3'] * 1000) * 100
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 电压vs效率
        axes[0, 0].scatter(self.data['U_avg_kV'], self.data['efficiency'], 
                          alpha=0.5, s=20, c=self.data['C_in_gNm3'], cmap='viridis')
        cbar = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
        cbar.set_label('入口浓度 (g/Nm³)', fontsize=10)
        axes[0, 0].set_xlabel('平均电压 (kV)', fontsize=11)
        axes[0, 0].set_ylabel('除尘效率 (%)', fontsize=11)
        axes[0, 0].set_title('电压 vs 除尘效率', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 振打周期vs效率
        axes[0, 1].scatter(self.data['T_avg_s'], self.data['efficiency'], 
                          alpha=0.5, s=20, c=self.data['C_in_gNm3'], cmap='viridis')
        axes[0, 1].set_xlabel('平均振打周期 (s)', fontsize=11)
        axes[0, 1].set_ylabel('除尘效率 (%)', fontsize=11)
        axes[0, 1].set_title('振打周期 vs 除尘效率', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 效率分布
        axes[1, 0].hist(self.data['efficiency'], bins=30, color='cyan', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=self.data['efficiency'].mean(), color='red', 
                          linestyle='--', linewidth=2, label=f'平均值: {self.data["efficiency"].mean():.1f}%')
        axes[1, 0].set_xlabel('除尘效率 (%)', fontsize=11)
        axes[1, 0].set_ylabel('频数', fontsize=11)
        axes[1, 0].set_title('除尘效率分布', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. 电耗效率比
        self.data['power_per_efficiency'] = self.data['P_total_kW'] / (self.data['efficiency'] + 1e-6)
        axes[1, 1].scatter(self.data['C_in_gNm3'], self.data['power_per_efficiency'], 
                          alpha=0.5, s=20, color='brown')
        axes[1, 1].set_xlabel('入口浓度 (g/Nm³)', fontsize=11)
        axes[1, 1].set_ylabel('单位效率电耗 (kW/%)', fontsize=11)
        axes[1, 1].set_title('入口浓度 vs 电能利用效率', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ 图表已保存：{save_path}")
        
        plt.show()


def create_summary_report(data, optimal_params, output_dir='outputs/'):
    """
    创建综合总结报告
    
    Parameters:
    -----------
    data : DataFrame
        原始数据
    optimal_params : dict
        最优参数
    output_dir : str
        输出目录
    """
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("生成综合分析报告")
    print("="*60)
    
    visualizer = Visualizer(data)
    
    # 生成所有图表
    visualizer.plot_correlation_heatmap(f'{output_dir}01_correlation_heatmap.png')
    visualizer.plot_input_vs_output(f'{output_dir}02_input_vs_output.png')
    visualizer.plot_time_series(f'{output_dir}03_time_series.png')
    visualizer.plot_parameter_distribution(f'{output_dir}04_parameter_distribution.png')
    visualizer.plot_efficiency_analysis(f'{output_dir}05_efficiency_analysis.png')
    
    print(f"\n✓ 所有图表已保存到：{output_dir}")


if __name__ == "__main__":
    from data_loader import load_data
    
    data = load_data('data/Cement_ESP_Data.csv')
    
    visualizer = Visualizer(data)
    visualizer.plot_correlation_heatmap()
    visualizer.plot_input_vs_output()
    visualizer.plot_time_series()
    visualizer.plot_parameter_distribution()
    visualizer.plot_efficiency_analysis()
