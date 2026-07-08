"""
建模模块
构建出口粉尘浓度与操作参数的回归模型
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')


class ESPModel:
    """电除尘器出口浓度预测模型"""
    
    def __init__(self, model_type='linear'):
        """
        初始化模型
        
        Parameters:
        -----------
        model_type : str
            模型类型：'linear', 'ridge', 'lasso', 'rf', 'gbm'
        """
        self.model_type = model_type
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None
        self.train_score = None
        self.test_score = None
        self.cv_scores = None
        
    def _init_model(self):
        """初始化具体的模型对象"""
        if self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif self.model_type == 'lasso':
            self.model = Lasso(alpha=0.01)
        elif self.model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif self.model_type == 'gbm':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"未知模型类型：{self.model_type}")
    
    def prepare_data(self, data, target='C_out_mgNm3', test_size=0.2, random_state=42):
        """
        准备训练和测试数据
        
        Parameters:
        -----------
        data : DataFrame
            输入数据
        target : str
            目标变量列名
        test_size : float
            测试集比例
        random_state : int
            随机种子
        """
        print("\n[数据准备]...")
        
        # 选择特征
        excluded_cols = ['timestamp', target, 'U1_kV', 'U2_kV', 'U3_kV', 'U4_kV', 
                        'T1_s', 'T2_s', 'T3_s', 'T4_s', 'P_total_kW']
        X_cols = [col for col in data.columns if col not in excluded_cols]
        
        self.feature_names = X_cols
        X = data[X_cols].values
        y = data[target].values
        
        # 分割数据
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"  ✓ 特征数：{len(X_cols)}")
        print(f"  ✓ 训练集：{len(self.X_train)} 样本")
        print(f"  ✓ 测试集：{len(self.X_test)} 样本")
        
        return self
    
    def train(self):
        """训练模型"""
        print(f"\n[模型训练] 使用 {self.model_type.upper()} 模型...")
        
        self._init_model()
        self.model.fit(self.X_train, self.y_train)
        
        # 计算性能指标
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        
        self.train_score = r2_score(self.y_train, y_pred_train)
        self.test_score = r2_score(self.y_test, y_pred_test)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        
        print(f"\n  训练集指标：")
        print(f"    R² = {self.train_score:.4f}")
        print(f"    RMSE = {train_rmse:.4f} mg/Nm³")
        print(f"    MAE = {train_mae:.4f} mg/Nm³")
        
        print(f"\n  测试集指标：")
        print(f"    R² = {self.test_score:.4f}")
        print(f"    RMSE = {test_rmse:.4f} mg/Nm³")
        print(f"    MAE = {test_mae:.4f} mg/Nm³")
        
        return self
    
    def cross_validate(self, cv=5):
        """
        交叉验证
        
        Parameters:
        -----------
        cv : int
            折数
        """
        print(f"\n[交叉验证] {cv}折交叉验证...")
        
        self.cv_scores = cross_val_score(
            self.model, self.X_train, self.y_train, 
            cv=cv, scoring='r2', n_jobs=-1
        )
        
        print(f"  各折得分：{self.cv_scores}")
        print(f"  平均 R² = {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std():.4f})")
        
        return self
    
    def get_feature_importance(self):
        """获取特征重要性"""
        if self.model_type in ['rf', 'gbm']:
            importances = self.model.feature_importances_
            importance_df = pd.DataFrame({
                '特征': self.feature_names,
                '重要性': importances
            }).sort_values('重要性', ascending=False)
            
            print("\n[特征重要性]（Top 15）：")
            print(importance_df.head(15).to_string(index=False))
            
            return importance_df
        elif self.model_type in ['linear', 'ridge', 'lasso']:
            coef_df = pd.DataFrame({
                '特征': self.feature_names,
                '系数': self.model.coef_
            }).sort_values('系数', key=abs, ascending=False)
            
            print("\n[回归系数]（Top 15）：")
            print(coef_df.head(15).to_string(index=False))
            
            return coef_df
        else:
            return None
    
    def predict(self, X):
        """
        预测
        
        Parameters:
        -----------
        X : array-like
            输入特征
            
        Returns:
        --------
        array : 预测值
        """
        return self.model.predict(X)
    
    def predict_from_params(self, params_dict):
        """
        从操作参数直接预测出口浓度
        
        Parameters:
        -----------
        params_dict : dict
            参数字典，包含所有特征列
            
        Returns:
        --------
        float : 预测的出口浓度
        """
        X_new = np.array([params_dict.get(feat, 0) for feat in self.feature_names]).reshape(1, -1)
        return self.model.predict(X_new)[0]
    
    def get_model(self):
        """返回训练好的模型"""
        return self.model
    
    def save_model(self, filepath):
        """保存模型"""
        import joblib
        joblib.dump(self.model, filepath)
        print(f"✓ 模型已保存到：{filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        import joblib
        self.model = joblib.load(filepath)
        print(f"✓ 模型已加载：{filepath}")


def build_ensemble_model(data, target='C_out_mgNm3'):
    """
    构建集成模型（多个模型的平均）
    
    Parameters:
    -----------
    data : DataFrame
        输入数据
    target : str
        目标变量
        
    Returns:
    --------
    dict : 包含所有模型和结果的字典
    """
    print("\n" + "="*60)
    print("构建多模型集成")
    print("="*60)
    
    models_config = [
        ('linear', 'Linear Regression'),
        ('ridge', 'Ridge Regression'),
        ('rf', 'Random Forest'),
        ('gbm', 'Gradient Boosting'),
    ]
    
    results = {}
    
    for model_type, model_name in models_config:
        print(f"\n{model_name}:")
        model = ESPModel(model_type=model_type)
        model.prepare_data(data, target=target)
        model.train()
        model.cross_validate(cv=5)
        
        results[model_type] = {
            'model': model,
            'name': model_name,
            'train_r2': model.train_score,
            'test_r2': model.test_score,
            'cv_r2': model.cv_scores.mean()
        }
    
    # 打印对比
    print("\n" + "="*60)
    print("模型性能对比")
    print("="*60)
    comparison_df = pd.DataFrame({
        '模型': [results[k]['name'] for k in results],
        '训练 R²': [results[k]['train_r2'] for k in results],
        '测试 R²': [results[k]['test_r2'] for k in results],
        '交叉验证 R²': [results[k]['cv_r2'] for k in results],
    })
    print(comparison_df.to_string(index=False))
    
    return results


if __name__ == "__main__":
    # 使用示例
    from data_loader import load_data
    from feature_engineering import FeatureEngineer
    
    data = load_data('data/Cement_ESP_Data.csv')
    fe = FeatureEngineer(data)
    fe.execute_full_pipeline()
    processed_data = fe.get_data()
    
    # 单模型
    model = ESPModel(model_type='linear')
    model.prepare_data(processed_data)
    model.train()
    model.cross_validate()
    model.get_feature_importance()
    
    # 集成模型
    results = build_ensemble_model(processed_data)
