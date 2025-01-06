"""
@author: kai
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

def train_and_evaluate(df_combined):
    # 定义目标变量和特征
    target = 'ENERGY_CONSUMPTION_CURRENT'
    # 从特征中排除非数值列
    numeric_cols = df_combined.select_dtypes(include=[np.number]).columns.tolist()
    features = [col for col in numeric_cols if col != target]
    
    X = df_combined[features]
    y = df_combined[target]
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 特征标准化 - 只包括数值特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train RandomForest model
    model = RandomForestRegressor(n_estimators= 50, min_samples_split= 6, min_samples_leaf=1, max_features='sqrt', max_depth= None, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"R2 Score: {r2}")
    print(f"Mean Squared Error: {mse}")
    
    # Plot results if R2 is good
    if r2 > 0.8:
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        plt.xlabel('Actual Energy Consumption')
        plt.ylabel('Predicted Energy Consumption')
        plt.title('Energy Consumption: Actual vs Predicted')
        plt.show()

# File paths
epc_file_path = 'F:/desktop/processed_epc_Cardiff.csv'
weather_file_path = 'F:/desktop/processed_weather_data.csv'

# Load, prepare and evaluate
df_combined = load_and_prepare_data(epc_file_path, weather_file_path)
train_and_evaluate(df_combined)