import yfinance as yf
import pandas as pd

def get_stock_data(ticker, start_date="2010-01-01", end_date="2023-12-31"):
    """
    Función para obtener los datos históricos de una acción desde Yahoo Finance.
    
    Args:
    ticker (str): Símbolo del ticker de la empresa (por ejemplo, 'AAPL' para Apple).
    start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
    end_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
    
    Returns:
    pd.DataFrame: Un DataFrame con los datos históricos de la acción.
    """
    stock = yf.Ticker(ticker)
    stock_data = stock.history(start=start_date, end=end_date)
    return stock_data

def get_financials(ticker):
    """
    Función para obtener datos financieros (EPS, PER, etc.) de una empresa desde Yahoo Finance.
    
    Args:
    ticker (str): Símbolo del ticker de la empresa.
    
    Returns:
    dict: Diccionario con los datos financieros clave.
    """
    stock = yf.Ticker(ticker)
    financials = {
        "EPS": stock.info.get('trailingEps'),
        "PER": stock.info.get('trailingPE'),
        "Sector": stock.info.get('sector'),
        "MarketCap": stock.info.get('marketCap')
    }
    return financials
