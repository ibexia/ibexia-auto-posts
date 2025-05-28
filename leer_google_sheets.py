import os
import json
import smtplib
import yfinance as yf
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.oauth2 import service_account
from googleapiclient.discovery import build
import google.generativeai as genai


def leer_google_sheets():
    credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not credentials_json:
        raise Exception("No se encontró la variable de entorno GOOGLE_APPLICATION_CREDENTIALS")

    creds_dict = json.loads(credentials_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
    )

    spreadsheet_id = os.getenv('SPREADSHEET_ID')
    if not spreadsheet_id:
        raise Exception("No se encontró la variable de entorno SPREADSHEET_ID")

    range_name = os.getenv('RANGE_NAME', 'A1:A10')  # Solo tickers

    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    values = result.get('values', [])

    if not values:
        print('No se encontraron datos.')
    else:
        print('Datos leídos de la hoja:')
        for row in values:
            print(row)

    return [row[0] for row in values if row]


def calcular_smi(df, k_window=14, d_window=3):
    df = df.copy()
    hl = (df['High'] + df['Low']) / 2
    close = df['Close']

    min_low = df['Low'].rolling(k_window).min()
    max_high = df['High'].rolling(k_window).max()
    diff = max_high - min_low
    rel_close = close - (max_high + min_low) / 2

    smi = 100 * (rel_close / (diff / 2))
    smi_ma = smi.rolling(d_window).mean()
    return smi_ma.iloc[-1]  # Último valor SMI suavizado


def interpretar_smi(smi_valor):
    if smi_valor >= 60:
        return "sobrecomprado", "Vender"
    elif smi_valor <= -60:
        return "sobrevendido", "Comprar"
    else:
        return "neutral", "Mantener"


def obtener_datos_yfinance(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="30d")

    try:
        smi_valor = calcular_smi(hist)
        condicion_smi, recomendacion = interpretar_smi(smi_valor)

        datos = {
            "NOMBRE_EMPRESA": info.get("longName", ticker),
            "PRECIO_ACTUAL": round(info.get("currentPrice", 0), 2),
            "VOLUMEN": info.get("volume", 0),
            "SOPORTE": round(hist["Low"].min(), 2),
            "RESISTENCIA": round(hist["High"].max(), 2),
            "SMI_VALOR": round(smi_valor, 2),
            "CONDICION_SMI": condicion_smi,
            "RECOMENDACION": recomendacion,
            "INGRESOS": info.get("totalRevenue", "N/A"),
            "EBITDA": info.get("ebitda", "N/A"),
            "BENEFICIOS": info.get("grossProfits", "N/A"),
            "DEUDA": info.get("totalDebt", "N/A"),
            "FLUJO_CAJA": info.get("freeCashflow", "N/A"),
            "EXPANS
