# 水泥工业电除尘器智能优化控制系统

## 快速开始 🚀

### 1. 环境配置
```bash
# 克隆项目
git clone https://github.com/ZZx112131/cement-esp-optimization.git
cd cement-esp-optimization

# 安装依赖
pip install -r requirements.txt

# 如果使用conda
conda create -n esp-opt python=3.8
conda activate esp-opt
pip install -r requirements.txt
```

### 2. 数据准备
将数据文件 `Cement_ESP_Data.csv` 放到 `data/` 目录：
```
cement-esp-optimization/
├── data/
│   └── Cement_ESP_Data.csv  ← 放置您的数据文件
├── src/
├── main.py
└── ...
```

### 3. 运行分析
```bash
python main.py
```

程序将自动执行完整的分析流程，包括：
- ✓ 数据加载与预处理
- ✓ 特征工程
- ✓ 模型建立与评估
- ✓ 问题1：关系分析
- ✓ 问题2：参数优化
- ✓ 问题3：优先级分析
- ✓ 问题4：排放标准影响
- ✓ 可视化与报告生成

---

## 项目结构 📁

```
cement-esp-optimization/
│
├── data/
│   └── Cement_ESP_Data.csv          # 输入数据文件
│
├── src/                             # 源代码模块
│   ├── __init__.py
│   ├── data_loader.py               # 数据加载与预处理
│   ├── feature_engineering.py       # 特征工程
│   ├── modeling.py                  # 模型建立
│   ├── optimization.py              # 问题2：参数优化
│   ├── analysis.py                  # 问题1/3/4：综合分析
│   └── visualization.py             # 数据可视化
│
├── outputs/                         # 输出结果目录
│   ├── report.txt                   # 综合分析报告
│   ├── optimization_table.csv       # 优化参数表
│   ├── 01_correlation_heatmap.png   # 相关性热力图
│   ├── 02_input_vs_output.png       # 入出口关系图
│   ├── 03_time_series.png           # 时间序列图
│   ├── 04_parameter_distribution.png # 参数分布图
│   └── 05_efficiency_analysis.png   # 效率分析图
│
├── notebooks/                       # Jupyter分析笔记本
│   ├── 01_data_exploration.ipynb
│   ├── 02_problem1_analysis.ipynb
│   ├── 03_problem2_optimization.ipynb
│   ├── 04_problem3_comparison.ipynb
│   └── 05_problem4_emission_analysis.ipynb
│
├── requirements.txt                 # Python依赖
├── main.py                          # 主程序
└── README.md                        # 项目文档
```

---

## 核心功能说明 🎯

### 问题1：入口条件与出口浓度的关系分析
```python
from src.analysis import Problem1Analyzer

analyzer = Problem1Analyzer(data, model)
correlations = analyzer.analyze_parameter_impact()  # 参数影响度
vibration = analyzer.analyze_vibration_effect()      # 振打尘饼分析
```

**分析内容**：
- 入口浓度、温度、流量、电压、振打周期与出口浓度的相关性
- 振打周期对瞬时排放峰值的影响
- 极板积灰与排放浓度的关系

---

### 问题2：参数优化与工况划分
```python
from src.optimization import WorkingConditionAnalyzer, ParameterOptimizer

# 工况划分
analyzer = WorkingConditionAnalyzer(data)
analyzer.divide_by_clustering(n_clusters=5)

# 参数优化
optimizer = ParameterOptimizer(model, data, emission_limit=10.0)
optimal_params = optimizer.optimize_all_conditions(condition_list, names)
```

**优化问题**：
```
minimize:  P_total(U, T)
subject to:
  C_out(U, T) ≤ 10 mg/Nm³  (排放标准)
  3 ≤ Uᵢ ≤ 80 kV
  20 ≤ Tᵢ ≤ 180 s
```

---

### 问题3：典型工况对比与优先级分析
```python
from src.analysis import Problem3Analyzer

analyzer = Problem3Analyzer(optimal_params, data)
selected = analyzer.select_representative_conditions(n_select=2)
analyzer.compare_strategies(selected)
analyzer.analyze_priority_rules()
```

**优先级规律**：
| 工况类型 | 优先调节参数 | 原因 |
|--------|----------|------|
| 低浓度 (C_in < 20) | 电压 | 电压敏感性高，经济性好 |
| 高浓度 (C_in ≥ 30) | 振打周期 | 需频繁清灰保持稳定 |
| 高温 | 电压 | 高温提升电晕效率 |

---

### 问题4：排放标准提升的影响分析
```python
from src.analysis import Problem4Analyzer

analyzer = Problem4Analyzer(model, data)
result = analyzer.analyze_emission_standard_impact(
    current_limit=10.0,   # 当前标准
    new_limit=5.0,        # 新标准
    high_conc_threshold=30.0
)
```

**定量分析**：
- 排放标准从10提升到5 mg/Nm³
- 电耗增加预测：平均 ~15% (高浓度工况)
- 应对建议：提升电压+缩短振打周期+增加后级除尘

---

## 数据格式说明 📋

### 输入数据：Cement_ESP_Data.csv

| 字段名 | 单位 | 说明 |
|-------|------|------|
| timestamp | - | 时间戳（分钟级） |
| Temp_C | ℃ | 烟气入口温度 |
| C_in_gNm3 | g/Nm³ | 入口粉尘浓度 |
| Q_Nm3h | Nm³/h | 烟气流量 |
| U1_kV ~ U4_kV | kV | 各电场二次电压 |
| T1_s ~ T4_s | s | 各电场振打周期 |
| C_out_mgNm3 | mg/Nm³ | **出口粉尘浓度（目标变量）** |
| P_total_kW | kW | 总除尘电耗 |

### 输出数据：optimization_table.csv

| 字段 | 说明 |
|------|------|
| 工况 | 典型工况编号 |
| 最优电压 (kV) | 推荐的平均电压 |
| 最优振打周期 (s) | 推荐的平均振打周期 |
| 预期电耗 (kW) | 优化后的电能消耗 |
| 预期出口浓度 (mg/Nm³) | 预测的出口浓度 |

---

## 建模方法 🧠

### 特征工程
- **交互特征**：C_in×U, Temp×C_in, T×C_in, U×T
- **多项式特征**：各特征的平方项
- **比例特征**：Temp/Q, C_in/U, 振打频率
- **统计特征**：平均值、标准差

### 模型对比
系统自动对比4种模型的性能：
1. **线性回归** - 可解释性强，速度快
2. **岭回归** - 正则化，防止过拟合
3. **随机森林** - 非线性关系捕捉
4. **梯度提升** - 集成学习，性能最优

### 优化求解
- **方法**：SLSQP (Sequential Least Squares Programming)
- **约束**：出口浓度 ≤ 排放标准，参数范围限制
- **目标**：最小化总电耗

---

## 典型运行结果示例 📊

```
════════════════════════════════════════════════════════════════════════════════
  水泥工业电除尘器智能优化控制系统
════════════════════════════════════════════════════════════════════════════════

【数据概览】
  • 总样本数：10080
  • 入口浓度范围：5.12 - 95.34 g/Nm³
  • 出口浓度范围：0.52 - 42.18 mg/Nm³

【问题1：关系分析】
  • 入口浓度 vs 出口浓度相关性：0.8234 (强正相关)
  • 电压 vs 出口浓度相关性：-0.7156 (负相关)

【问题2：参数优化】
  工况1: 最优电压 45.2 kV, 振打周期 60.5 s, 预期电耗 245.3 kW
  工况2: 最优电压 62.8 kV, 振打周期 38.2 s, 预期电耗 285.6 kW
  ...

【问题4：排放标准提升影响】
  • 当前标准满足率：92.3%
  • 新标准满足率：67.8%
  • 平均电耗增幅：~15.2%
```

---

## 常见问题 ❓

### Q1: 数据文件格式不对？
A: 确保CSV文件包含所有必需字段，字段名大小写要匹配。可用以下代码检查：
```python
import pandas as pd
df = pd.read_csv('data/Cement_ESP_Data.csv')
print(df.columns)
print(df.head())
```

### Q2: 模型性能不理想？
A: 尝试以下方案：
- 增加数据量或改进数据质量
- 调整特征工程参数
- 尝试不同的模型（如GBM）
- 参数优化时调整约束条件

### Q3: 如何集成到SCADA系统？
A: 
```python
from src.optimization import ParameterOptimizer

optimizer = ParameterOptimizer(model, data)
params = optimizer.optimize_for_condition(realtime_data, "实时工况")
# 发送 params['U_opt'], params['T_opt'] 到控制器
```

### Q4: 可以预测未来的出口浓度吗？
A: 可以，使用模型的 `predict()` 方法：
```python
X_future = prepare_features(future_data)
C_out_pred = model.predict(X_future)
```

---

## 技术栈 🛠️

| 模块 | 库 |
|------|-----|
| 数据处理 | pandas, numpy |
| 机器学习 | scikit-learn, scipy |
| 可视化 | matplotlib, seaborn |
| 交互式分析 | jupyter, ipython |

---

## 许可证

MIT License

---

## 联系与支持

如有问题或建议，欢迎提Issue或PR！

**创建者**：Data Science Team  
**创建时间**：2026年7月

---

*最后更新：2026-07-08*
