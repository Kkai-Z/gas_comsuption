# -*- coding: utf-8 -*-
"""

@author: kai
"""

from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# 文件路径
epc_file_path = 'F:/desktop/processed_epc_Newport.csv'
weather_file_path = 'F:/desktop/processed_weather_data.csv'

# 定义要从数据中加载的列
useful_columns = [
    'CURRENT_ENERGY_EFFICIENCY', 'POTENTIAL_ENERGY_EFFICIENCY',
    'ENVIRONMENT_IMPACT_CURRENT', 'ENVIRONMENT_IMPACT_POTENTIAL',
    'ENERGY_CONSUMPTION_CURRENT', 'ENERGY_CONSUMPTION_POTENTIAL',
    'CO2_EMISSIONS_CURRENT', 'CO2_EMISSIONS_POTENTIAL',
    'LIGHTING_COST_CURRENT', 'LIGHTING_COST_POTENTIAL',
    'HEATING_COST_CURRENT', 'HEATING_COST_POTENTIAL',
    'HOT_WATER_COST_CURRENT', 'HOT_WATER_COST_POTENTIAL',
    'TOTAL_FLOOR_AREA', 'NUMBER_HABITABLE_ROOMS', 'NUMBER_HEATED_ROOMS',
    'FLOOR_HEIGHT', 'MAIN_FUEL', 'INSPECTION_DATE', 'LODGEMENT_DATETIME'
]

def load_and_prepare_data(epc_file_path, weather_file_path):
    # Load EPC data with required columns
    use_cols = [
        'LODGEMENT_DATE', 'ENERGY_CONSUMPTION_CURRENT', 'CURRENT_ENERGY_EFFICIENCY',
        'ENVIRONMENT_IMPACT_CURRENT', 'CO2_EMISSIONS_CURRENT', 'LIGHTING_COST_CURRENT',
        'HEATING_COST_CURRENT', 'TOTAL_FLOOR_AREA', 'MAIN_FUEL'
    ]
    df_epc = pd.read_csv(epc_file_path, usecols=use_cols)
    df_epc['LODGEMENT_DATE'] = pd.to_datetime(df_epc['LODGEMENT_DATE'])
    
    # Filter out rows where MAIN_FUEL contains 'electricity'
    df_epc = df_epc[~df_epc['MAIN_FUEL'].str.contains('electricity', case=False, na=False)]
    
    # Load weather data
    df_weather = pd.read_csv(weather_file_path)
    df_weather['Date'] = pd.to_datetime(df_weather['year'].astype(str) + '-' + df_weather['month'].astype(str), format='%Y-%m')
    
   # Merge data on year and month extracted from LODGEMENT_DATE
    df_epc['YearMonth'] = df_epc['LODGEMENT_DATE'].dt.to_period('M')
    df_weather['YearMonth'] = df_weather['Date'].dt.to_period('M')
    df_combined = pd.merge(df_epc, df_weather, on='YearMonth', how='left')
    
    # Create interaction features between EPC data and weather data
    for weather_feature in ['tmax', 'tmin', 'af', 'rain', 'sun', 'hdd']:
        df_combined[f'{weather_feature}_x_CURRENT_ENERGY_EFFICIENCY'] = df_combined[weather_feature] * df_combined['CURRENT_ENERGY_EFFICIENCY']
        df_combined[f'{weather_feature}_x_ENVIRONMENT_IMPACT_CURRENT'] = df_combined[weather_feature] * df_combined['ENVIRONMENT_IMPACT_CURRENT']
        df_combined[f'{weather_feature}_x_CO2_EMISSIONS_CURRENT'] = df_combined[weather_feature] * df_combined['CO2_EMISSIONS_CURRENT']
        df_combined[f'{weather_feature}_x_LIGHTING_COST_CURRENT'] = df_combined[weather_feature] * df_combined['LIGHTING_COST_CURRENT']
        # ... Add more interaction features as needed
        
    # Drop unnecessary columns
    df_combined.drop(columns=['YearMonth', 'Date'], inplace=True)
    return df_combined

# 使用定义的函数加载数据
df_combined = load_and_prepare_data(epc_file_path, weather_file_path)

# 选择特征和目标变量
features = df_combined.select_dtypes(include=[np.number]).columns.drop('ENERGY_CONSUMPTION_CURRENT')
X = df_combined[features]
y = df_combined['ENERGY_CONSUMPTION_CURRENT']

# 数据切分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 预定义最佳参数
best_params = {'n_estimators': 50, 'min_samples_split': 6, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None}
best_params = {
    'n_estimators': 50,
    'min_samples_split': 6,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'max_depth': None
}
# 参数范围更小，减少交叉验证的折数
n_estimators_range = [30, 50, 70]
max_depth_range = [None, 20, 40]
min_samples_split_range = [4, 6, 8]
min_samples_leaf_range = [1, 3, 5]
max_features_range = ['sqrt', 'log2']

def test_param_performance(param_name, param_values):
    scores = []
    for value in param_values:
        temp_params = best_params.copy()
        temp_params[param_name] = value
        model = RandomForestRegressor(**temp_params, random_state=42)
        # 使用3折交叉验证而不是默认的5折
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='r2')
        scores.append(np.mean(cv_scores))
        
    plt.figure(figsize=(10, 5))
    plt.plot(param_values, scores, marker='o')
    plt.title(f'Effect of {param_name} on Model Performance')
    plt.xlabel(param_name)
    plt.ylabel('Average R2 Score')
    plt.grid(True)
    plt.show()

# 测试不同的参数
test_param_performance('n_estimators', n_estimators_range)
test_param_performance('max_depth', max_depth_range)
test_param_performance('min_samples_split', min_samples_split_range)
test_param_performance('min_samples_leaf', min_samples_leaf_range)
test_param_performance('max_features', max_features_range)


