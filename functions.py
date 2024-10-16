import pandas as pd
import numpy as np
import re
import yfinance as yf  # type: ignore

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA # type: ignore
from statsmodels.tsa.statespace.sarimax import SARIMAX # type: ignore



import matplotlib.pyplot as plt
import seaborn as sns
# Dowload Data
def download_stock_data(ticker):
    """
    Descarga los últimos 10 años de datos históricos de una acción usando yfinance.
    """
    stock_data = yf.Ticker(ticker).history(period="10y")
    return stock_data
#----------------------------------------------------------------
# Clean Data

def data_quality_report(df):
    """
    Esta función genera un informe detallado de la calidad de los datos en un DataFrame.
    Proporciona información sobre valores nulos, duplicados, tipos de datos, outliers, y más.
    """
    
    # Verificar valores nulos
    null_values = df.isnull().sum()
    
    # Verificar filas duplicadas
    duplicates = df.duplicated().sum()
    
    # Tipos de datos
    data_types = df.dtypes
    
    # Descripción estadística de los datos numéricos
    descriptive_stats = df.describe()
    
    # Detectar outliers usando el rango intercuartílico (IQR)
    outliers = pd.DataFrame(columns=['Column', 'Outliers_Count'])
    outlier_rows = []
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outlier_condition = ((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))
        outlier_rows.append({'Column': column, 'Outliers_Count': outlier_condition.sum()})
    
    # Crear DataFrame de outliers
    outliers = pd.concat([outliers, pd.DataFrame(outlier_rows)], ignore_index=True)
    
    # Imprimir resumen
    print("----- Informe de Calidad de Datos -----")
    print("\nValores nulos por columna:\n", null_values)
    print("\nNúmero de filas duplicadas:", duplicates)
    print("\nTipos de datos:\n", data_types)
    print("\nDescripción estadística de las columnas numéricas:\n", descriptive_stats)
    print("\nNúmero de outliers por columna:\n", outliers)
    
    # Retornar el informe como un diccionario si deseas almacenarlo
    report = {
        'null_values': null_values,
        'duplicates': duplicates,
        'data_types': data_types,
        'descriptive_stats': descriptive_stats,
        'outliers': outliers
    }
    
    return report

def normalizar_nombres_columnas(df):
    df.columns = [re.sub(r'\W+', '_', col).lower() for col in df.columns]
    return df




def preprocess_data(df):
    """
    Preprocesa los datos ajustando precios por stock splits y sumando dividendos.
    También crea un conjunto de datos normalizado/estandarizado para futuros análisis.
    """
    
    # 1. Ajuste de precios y volumen por stock splits
    # Para evitar distorsiones en los datos debido a splits, ajustamos los precios y el volumen
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col] / (df['stock_splits'].replace(0, 1).cumprod())
    
    # 2. Suma acumulativa de dividendos
    df['dividend_return'] = df['dividends'].cumsum()

    # 3. Normalización (Min-Max Scaling)
    # Aplicamos MinMaxScaler solo a las columnas relevantes
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])
    
    # 4. Estandarización (StandardScaler)
    standard_scaler = StandardScaler()
    df_standardized = df.copy()
    df_standardized[['open', 'high', 'low', 'close', 'volume']] = standard_scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])
    
    print("Preprocesamiento completo. Se han generado versiones con los datos ajustados, normalizados y estandarizados.")
    
    return df, df_normalized, df_standardized

def preprocess_data_dos(df):
    """
    Preprocesa los datos ajustando precios por stock splits y sumando dividendos.
    También crea un conjunto de datos normalizado/estandarizado para futuros análisis.
    """
    
    # 1. Ajuste de precios y volumen por stock splits
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col] / (df['stock_splits'].replace(0, 1).cumprod())
    
    # 2. Suma acumulativa de dividendos
    df['dividend_return'] = df['dividends'].cumsum()
    
    # 3. Crear nuevas características antes de normalización/estandarización
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['log_return'].rolling(window=30).std() * np.sqrt(30)
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['ma100'] = df['close'].rolling(window=100).mean()
    df['volume_to_price'] = df['volume'] / df['close']
    
    # 4. Eliminar valores NaN
    df.dropna(inplace=True)
    
    # 5. Normalización (Min-Max Scaling)
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])
    
    # 6. Estandarización (StandardScaler)
    standard_scaler = StandardScaler()
    df_standardized = df.copy()
    df_standardized[['open', 'high', 'low', 'close', 'volume']] = standard_scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])
    
    print("Preprocesamiento completo. Se han generado versiones con los datos ajustados, normalizados y estandarizados.")
    
    return df, df_normalized, df_standardized


#----------------------------------------------------------------
#EDA
def perform_eda(df):
    """
    Realiza un análisis exploratorio de datos (EDA) visualizando tendencias, distribuciones y correlaciones.
    """

    # Configurar estilo de gráficos
    sns.set(style="whitegrid")
    
    # 1. Visualización de Tendencias de Precios
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='Precio de Cierre')
    plt.title('Evolución del Precio de Cierre Ajustado (AAPL)')
    plt.xlabel('Fecha')
    plt.ylabel('Precio ($)')
    plt.legend()
    plt.show()

    # 2. Análisis de Volumen de Transacciones
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['volume'], label='Volumen de Transacciones', color='orange')
    plt.title('Volumen de Transacciones (AAPL)')
    plt.xlabel('Fecha')
    plt.ylabel('Volumen')
    plt.legend()
    plt.show()

    # 3. Distribución de los Precios (histograma)
    plt.figure(figsize=(12, 6))
    sns.histplot(df['close'], bins=50, kde=True, color='blue')
    plt.title('Distribución del Precio de Cierre (AAPL)')
    plt.xlabel('Precio de Cierre ($)')
    plt.ylabel('Frecuencia')
    plt.show()

    # 4. Distribución del Volumen (histograma)
    plt.figure(figsize=(12, 6))
    sns.histplot(df['volume'], bins=50, kde=True, color='orange')
    plt.title('Distribución del Volumen de Transacciones (AAPL)')
    plt.xlabel('Volumen')
    plt.ylabel('Frecuencia')
    plt.show()

    # 5. Mapa de Calor de Correlaciones
    plt.figure(figsize=(10, 8))
    corr_matrix = df[['open', 'high', 'low', 'close', 'volume', 'dividends']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title('Mapa de Calor de Correlaciones')
    plt.show()

    # 6. Suma Acumulada de Dividendos
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['dividend_return'], label='Suma Acumulada de Dividendos', color='green')
    plt.title('Suma Acumulada de Dividendos (AAPL)')
    plt.xlabel('Fecha')
    plt.ylabel('Dividendos Acumulados ($)')
    plt.legend()
    plt.show()

    print("EDA completado.")
#----------------------------------------------------------------
# Modelos
def train_regression_model(train_data, validation_data, features, target):
    """
    Entrena un modelo de Regresión Lineal y lo evalúa.
    """
    # Entrenamiento
    X_train = train_data[features]
    y_train = train_data[target]
    
    X_validation = validation_data[features]
    y_validation = validation_data[target]
    
    # Crear y entrenar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred_train = model.predict(X_train)
    y_pred_validation = model.predict(X_validation)
    
    # Evaluación del modelo
    train_mae = mean_absolute_error(y_train, y_pred_train)
    validation_mae = mean_absolute_error(y_validation, y_pred_validation)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    validation_rmse = np.sqrt(mean_squared_error(y_validation, y_pred_validation))

        # Calcular el Coeficiente de Determinación (R²)
    r2_train = r2_score(y_train, y_pred_train)
    r2_validation = r2_score(y_validation, y_pred_validation)
    
    
    return model, y_pred_validation, validation_mae, validation_rmse, r2_validation


def train_random_forest_model(train_data, validation_data, features, target):
    """
    Entrena un modelo de Random Forest y lo evalúa.
    """
    # Separar características y objetivo
    X_train = train_data[features]
    y_train = train_data[target]
    
    X_validation = validation_data[features]
    y_validation = validation_data[target]
    
    # Crear y entrenar el modelo de Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred_train = model.predict(X_train)
    y_pred_validation = model.predict(X_validation)
    
    # Evaluación del modelo
    train_mae = mean_absolute_error(y_train, y_pred_train)
    validation_mae = mean_absolute_error(y_validation, y_pred_validation)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    validation_rmse = np.sqrt(mean_squared_error(y_validation, y_pred_validation))
    
    # Calcular el Coeficiente de Determinación (R²)
    r2_train = r2_score(y_train, y_pred_train)
    r2_validation = r2_score(y_validation, y_pred_validation)
    
    return model, y_pred_validation, validation_mae, validation_rmse, r2_validation

def train_arima_model(train_data, validation_data, target, arima_order=(5, 1, 0)):
    """
    Entrena un modelo ARIMA y lo evalúa.
    """
    # Entrenar ARIMA
    model = ARIMA(train_data[target], order=arima_order)
    arima_result = model.fit()
    
    # Predicciones en validación
    y_pred_validation = arima_result.forecast(steps=len(validation_data))
    
    # Evaluación del modelo
    validation_mae = mean_absolute_error(validation_data[target], y_pred_validation)
    validation_rmse = np.sqrt(mean_squared_error(validation_data[target], y_pred_validation))
    
    return arima_result, y_pred_validation, validation_mae, validation_rmse

def train_sarima_model(train_data, validation_data, target, sarima_order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """
    Entrena un modelo SARIMA y lo evalúa.
    """
    # Entrenar SARIMA
    model = SARIMAX(train_data[target], order=sarima_order, seasonal_order=seasonal_order)
    sarima_result = model.fit()
    
    # Predicciones en validación
    y_pred_validation = sarima_result.forecast(steps=len(validation_data))
    
    # Evaluación del modelo
    validation_mae = mean_absolute_error(validation_data[target], y_pred_validation)
    validation_rmse = np.sqrt(mean_squared_error(validation_data[target], y_pred_validation))
    
    return sarima_result, y_pred_validation, validation_mae, validation_rmse

def train_prophet_model(df, target):
    from fbprophet import Prophet # type: ignore
    # Formatear los datos para Prophet
    df_prophet = df.reset_index()[['date', target]]
    df_prophet.columns = ['ds', 'y']  # Prophet usa estas columnas por defecto
    
    # Crear y entrenar el modelo
    model = Prophet()
    model.fit(df_prophet)
    
    # Crear predicciones hasta el 2030
    future = model.make_future_dataframe(periods=365*6)  # Predicción para los próximos 6 años (hasta 2030)
    forecast = model.predict(future)
    
    return model, forecast    

# def model_stock_price(data):
#     # Aplicar modelos aquí (regresión, ARIMA, etc.)
#     # Retornar predicciones y modelo entrenado
#     return predictions

# Evaluación de modelos
# def evaluate_model(predictions, actuals):
#     # Calcular MAE, RMSE u otras métricas
#     return mae, rmse
