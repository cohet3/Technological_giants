# Gigantes Tecnológicos

<div align="center">
  <img src="https://th.bing.com/th/id/OIG1.HIPOpMwWCqRYlyP92s1A?pid=ImgGn" width="400"/>
</div>

Este proyecto titulado **Gigantes Tecnológicos** representa el dominio de las grandes empresas tecnológicas mediante una metáfora visual de gigantes mitológicos en un entorno futurista. El proyecto incluye predicciones de precios para las principales empresas tecnológicas hasta el año 2030.

## Predicción de Precios de Acciones de Compañías Tecnológicas hasta 2030, [Visita la App](https://gigantstech.streamlit.app/)

Este proyecto tiene como objetivo realizar predicciones de los precios de cierre ajustados de cinco grandes compañías tecnológicas utilizando el modelo **Prophet** de Facebook, y visualizar los resultados en gráficos interactivos. Las compañías analizadas son:

- Apple (AAPL)
- Microsoft (MSFT)
- Amazon (AMZN)
- Alphabet (GOOGL)
- NVIDIA (NVDA)

## Tabla de Contenidos

- [Descripción](#descripción)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Contribución](#contribución)


## Descripción

Este proyecto aplica análisis de series temporales para predecir el comportamiento futuro de los precios de acciones de grandes compañías tecnológicas hasta el año 2030. Utiliza **Prophet** para realizar las predicciones y presenta los resultados mediante gráficos interactivos con **Plotly**.

El proyecto incluye:

- **Análisis Exploratorio de Datos (EDA)**: Limpieza y análisis de las series históricas de precios.
- **Predicción con Prophet**: Modelado de los precios futuros basado en los datos históricos.
- **Visualización interactiva**: Comparativa de las predicciones de varias compañías, con bandas de confianza y funcionalidades como zoom, desplazamiento, y selección de rango temporal.

## Tecnologías Utilizadas

Este proyecto utiliza las siguientes tecnologías y bibliotecas:

- **Python 3.8+**
- **Facebook Prophet** para modelar series temporales
- **Plotly** para gráficos interactivos
- **Pandas** para manejo y limpieza de datos
- **Matplotlib** para gráficos estáticos complementarios
- **yFinance** para obtener los datos históricos de las acciones

## Instalación

Sigue estos pasos para instalar y ejecutar el proyecto localmente:

1. Clona el repositorio:

   ```bash
   git clone https://github.com/cohet3/proyecto-tecnologicas-predicciones.git

2. Crea y activa un entorno virtual (opcional pero recomendado):
    
    En macOS y Linux:

    ```bash
    python3 -m venv venv
    source venv/bin/activate


3. Instala las dependencias necesarias:

   ```bash
   pip install -r requirements.txt


4. Asegúrate de que Prophet está correctamente instalado y configurado:

   ```bash
   pip install -r requirements.txt

## Uso

### 1. Cargar datos históricos

Descarga los datos de las compañías utilizando la API de **Yahoo Finance** a través de la librería `yFinance`. Esto se hace automáticamente en el código:

```python
import yfinance as yf

# Ejemplo para descargar los datos de Apple
data = yf.download('AAPL', start='2010-01-01', end='2024-01-01')
```

### 2. Ejecutar las predicciones

El modelo **Prophet** se ajusta a los datos y genera predicciones hasta 2030. Para cada compañía, se generan gráficos con las predicciones y rangos de confianza:

```python
from prophet import Prophet

# Ajuste del modelo
model = Prophet()
model.fit(df_prophet)

# Predicción hasta 2030
future = model.make_future_dataframe(periods=365 * 6)
forecast = model.predict(future)
```

### 3. Visualización interactiva

El proyecto incluye gráficos interactivos con **Plotly** que permiten explorar las predicciones, aplicar zoom, y cambiar el rango temporal visualizado:

```python
import plotly.graph_objects as go

# Crear gráfico interactivo
fig = go.Figure()
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines'))
fig.show()
```

### 4. Ejecutar el proyecto

Puedes ejecutar el proyecto a través de un **notebook** Jupyter o un script Python directamente. También se puede implementar en **Streamlit** para una aplicación web interactiva.

## Estructura del Proyecto

```bash
├── data/                        # Archivos CSV con los datos de predicción (si los guardas)
├── notebooks/                   # Notebooks Jupyter con EDA y modelado GRU
├── draft.ipynb                   # Notebook exploratorio que ejecuta modelos y diversas pruebas con los datos.
├── functions.py                 # Funciones reutilizables para el proyecto
├── main.ipynb                   # Notebook principal que ejecuta todo el pipeline
├── README.md                    # Documentación del proyecto
├── requirements.txt             # Dependencias del proyecto
```

## Contribución

¡Las contribuciones son bienvenidas! Si deseas contribuir al proyecto, sigue estos pasos:

1. Haz un **fork** del repositorio.
2. Crea una nueva rama con tu contribución: `git checkout -b mi-nueva-funcionalidad`.
3. Haz un **commit** con tus cambios: `git commit -m 'Añadir nueva funcionalidad'`.
4. Haz un **push** a la rama: `git push origin mi-nueva-funcionalidad`.
5. Abre una **pull request** para revisión.


---

### ¡Gracias por tu interés en este proyecto!
