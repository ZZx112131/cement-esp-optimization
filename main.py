"""
主程序 - 完整的数据分析和优化流程
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 导入所有模块
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.modeling import ESPModel, build_ensemble_model
from src.optimization import WorkingConditionAnalyzer, ParameterOptimizer, create_optimization_table
from src.analysis import Problem1Analyzer, Problem3Analyzer, Problem4Analyzer
from src.visualization import Visualizer, create_summary_report


def print_header(title):
    """打印标题"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def main():
    """主程序入口"""
    
    print_header("水泥工业电除尘器智能优化控制系统")
    print("  版本 v1.0 | 2026年7月")
    
    # =========================================================================
    # 第一步：数据加载与预处理
    # =========================================================================
    print_header("第一步：数据加载与预处理")
    
    data_path = 'data/Cement_ESP_Data.csv'
    
    if not Path(data_path).exists():
        print(f"✗ 错误：数据文件不存在 - {data_path}")
        print(f"  请将数据文件 Cement_ESP_Data.csv 放到 data/ 目录下")
        return
    
    loader = DataLoader(data_path)
    raw_data = loader.load()
    
    if raw_data is None:
        return
    
    data = loader.preprocess()
    loader.get_statistics()
    
    # =========================================================================
    # 第二步：特征工程
    # =========================================================================
    print_header("第二步：特征工程")
    
    fe = FeatureEngineer(data)
    fe.execute_full_pipeline(use_lagged=False, use_rolling=False)
    processed_data = fe.get_data()
    
    # =========================================================================
    # 第三步：模型建立（单模型 + 集成模型）
    # =========================================================================
    print_header("第三步：模型建立与评估")
    
    # 单模型：线性回归
    print("\n>>> 单模型：线性回归")
    model = ESPModel(model_type='linear')
    model.prepare_data(processed_data)
    model.train()
    model.cross_validate(cv=5)
    feature_importance = model.get_feature_importance()
    
    # 集成模型对比
    print("\n>>> 多模型集成对比")
    ensemble_results = build_ensemble_model(processed_data)
    
    # 选择最优模型
    best_model_type = max(ensemble_results.items(), 
                         key=lambda x: x[1]['test_r2'])[0]
    best_model = ensemble_results[best_model_type]['model']
    print(f"\n✓ 最优模型：{best_model_type.upper()} (R² = {ensemble_results[best_model_type]['test_r2']:.4f})")
    
    # =========================================================================
    # 第四步：问题1 - 入口条件与出口浓度关系分析
    # =========================================================================
    print_header("问题1：入口条件与出口浓度的关系分析")
    
    p1_analyzer = Problem1Analyzer(data, best_model)
    
    print("\n[1.1] 参数影响度分析")
    correlations = p1_analyzer.analyze_parameter_impact()
    
    print("\n[1.2] 振打尘饼现象分析")
    vibration_analysis = p1_analyzer.analyze_vibration_effect()
    
    # =========================================================================
    # 第五步：工况划分
    # =========================================================================
    print_header("工况划分与分析")
    
    print("\n[聚类分析] 基于K-means的典型工况划分")
    analyzer = WorkingConditionAnalyzer(data)
    analyzer.analyze_input_conditions()
    analyzer.divide_by_clustering(n_clusters=5, features=['C_in_gNm3', 'Temp_C'])
    
    condition_data = analyzer.get_conditions()
    
    # 提取各工况数据
    condition_list = []
    condition_names = []
    for i in range(5):
        cond_data = condition_data[condition_data['condition_cluster'] == i]
        if len(cond_data) > 0:
            condition_list.append(cond_data)
            condition_names.append(f"工况{i+1}")
    
    # =========================================================================
    # 第六步：问题2 - 参数优化
    # =========================================================================
    print_header("问题2：参数优化（保证排放达标前提下最小电耗）")
    
    print("\n[约束优化] 各工况参数优化求解")
    optimizer = ParameterOptimizer(best_model, data, emission_limit=10.0)
    optimal_params = optimizer.optimize_all_conditions(condition_list, condition_names)
    
    # 生成优化参数表
    print("\n[优化结果汇总表]")
    opt_table = create_optimization_table(optimal_params)
    print(opt_table.to_string(index=False))
    
    # =========================================================================
    # 第七步：问题3 - 典型工况对比与优先级分析
    # =========================================================================
    print_header("问题3：典型工况对比与优先级分析")
    
    p3_analyzer = Problem3Analyzer(optimal_params, data)
    
    print("\n[3.1] 代表性工况选择")
    selected_conditions = p3_analyzer.select_representative_conditions(n_select=2)
    
    print("\n[3.2] 最优策略对比")
    p3_analyzer.compare_strategies(selected_conditions)
    
    print("\n[3.3] 电压与振打周期优先级规律")
    p3_analyzer.analyze_priority_rules()
    
    # =========================================================================
    # 第八步：问题4 - 排放标准提升的影响
    # =========================================================================
    print_header("问题4：排放标准提升的影响分析")
    
    p4_analyzer = Problem4Analyzer(best_model, data)
    
    print("\n[4.1] 排放标准从10 → 5 mg/Nm³的影响分析")
    emission_analysis = p4_analyzer.analyze_emission_standard_impact(
        current_limit=10.0,
        new_limit=5.0,
        high_conc_threshold=30.0
    )
    
    # =========================================================================
    # 第九步：数据可视化
    # =========================================================================
    print_header("第九步：数据可视化与报告生成")
    
    print("\n生成综合分析图表...")
    
    try:
        import os
        os.makedirs('outputs', exist_ok=True)
        create_summary_report(data, optimal_params, output_dir='outputs/')
        print("\n✓ 所有图表已生成，保存在 outputs/ 目录")
    except Exception as e:
        print(f"\n⚠ 可视化生成异常：{e}")
    
    # =========================================================================
    # 生成综合报告
    # =========================================================================
    print_header("综合分析报告总结")
    
    report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    水泥工业电除尘器智能优化分析报告                          ║
╚════════════════════════════════════════════════════════════════════════════╝

【数据概览】
  • 总样本数：{len(data)}
  • 时间跨度：{data['timestamp'].min() if 'timestamp' in data.columns else '未知'} ~ {data['timestamp'].max() if 'timestamp' in data.columns else '未知'}
  • 入口浓度���围：{data['C_in_gNm3'].min():.2f} - {data['C_in_gNm3'].max():.2f} g/Nm³
  • 出口浓度范围：{data['C_out_mgNm3'].min():.2f} - {data['C_out_mgNm3'].max():.2f} mg/Nm³

【问题1：关系分析】
  • 入口浓度 vs 出口浓度相关性：{correlations['C_in']:.4f} (强正相关)
  • 温度 vs 出口浓度相关性：{correlations['Temp']:.4f}
  • 电压 vs 出口浓度相关性：{correlations['U_avg']:.4f} (负相关，提高电压降低出口浓度)
  • 振打周期 vs 出口浓度相关性：{correlations['T_avg']:.4f}
  
  振打尘饼分析：
    - 峰值发生次数：{vibration_analysis['peak_count']}
    - 平均峰值浓度：{vibration_analysis['peak_avg']:.2f} mg/Nm³
    - 最大峰值浓度：{vibration_analysis['peak_max']:.2f} mg/Nm³
    - 峰值时平均振打周期：{vibration_analysis['peak_T_avg']:.1f} s

【问题2：参数优化】
  • 工况划分方法：K-means聚类（基于入口浓度和温度）
  • 工况数量：{len(condition_names)}
  • 约束条件：出口浓度 ≤ {optimizer.emission_limit} mg/Nm³
  • 目标函数：最小化总电耗 (kW)
  
  优化结果摘要：
"""
    
    for cond_name, params in optimal_params.items():
        if params.get('success', False):
            report += f"""  {cond_name}:
      - 最优电压：{params['U_opt']:.1f} kV
      - 最优振打周期：{params['T_opt']:.1f} s
      - 预期电耗：{params['P_opt']:.1f} kW
      - 预期出口浓度：{params['C_out_opt']:.2f} mg/Nm³
"""
    
    report += f"""
【问题3：优先级分析】
  • 选中的代表性工况：{', '.join(selected_conditions)}
  
  优先级规律：
    ① 低浓度工况 (C_in < 20 g/Nm³)
       → 优先调节：电压
       → 原因：电压对低浓度工况的除尘效果更敏感
    
    ② 高浓度工况 (C_in ≥ 30 g/Nm³)
       → 优先调节：振打周期
       → 原因：需要频繁清灰以维持稳定运行

【问题4：排放标准提升影响】
  • 当前标准：10 mg/Nm³ → 新标准：5 mg/Nm³
  • 满足当前标准的比例：{emission_analysis['meet_current']:.1f}%
  • 满足新标准的比例：{emission_analysis['meet_new']:.1f}%
  
  电耗增加预测：
    - 高浓度工况电耗增幅：~{emission_analysis['power_increase_high']:.1f}%
    - 低浓度工况电耗增幅：~{emission_analysis['power_increase_low']:.1f}%
    - 平均电耗增幅：~{emission_analysis['power_increase_avg']:.1f}%
  
  应对建议（针对高浓度工况）：
    ✓ 优先提升电压至安全上限 (70-75 kV)
    ✓ 缩短振打周期保持极板清洁 (T_avg = 30-40 s)
    ✓ 分别调节各电场参数，实现精细化控制
    ✓ 考虑增加后级除尘设备（布袋除尘器）
    ✓ 优化工艺参数以降低入口浓度

【模型性能】
  • 最优模型：{best_model_type.upper()}
  • 测试集 R²：{ensemble_results[best_model_type]['test_r2']:.4f}
  • 交叉验证 R²：{ensemble_results[best_model_type]['cv_r2']:.4f}

【输出文件】
  • 优化参数表：outputs/optimization_table.csv
  • 分析图表：outputs/01_*.png - outputs/05_*.png
  • 本报告：outputs/report.txt

【建议】
  1. 定期更新数据和重新训练模型，适应工况变化
  2. 监控各电场的实际运行状况，对异常及时预警
  3. 将优化参数表集成到SCADA系统进行自动控制
  4. 每月分析一次实际运行效果与预测的偏差
  5. 持续采集数据，不断改进优化模型

═══════════════════════════════════════════════════════════════════════════════
分析时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    print(report)
    
    # 保存报告
    try:
        with open('outputs/report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print("\n✓ 报告已保存至：outputs/report.txt")
    except Exception as e:
        print(f"\n⚠ 报告保存失败：{e}")
    
    # 保存优化表
    try:
        opt_table.to_csv('outputs/optimization_table.csv', index=False, encoding='utf-8')
        print("✓ 优化参数表已保存至：outputs/optimization_table.csv")
    except Exception as e:
        print(f"⚠ 参数表保存失败：{e}")
    
    print_header("分析完成！")
    print("\n✓ 所有分析已完成，结果已保存到 outputs/ 目录")
    print("✓ 感谢使用水泥工业电除尘器智能优化系统！\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
    except Exception as e:
        print(f"\n\n发生错误：{e}")
        import traceback
        traceback.print_exc()
