import os
import json
import smtplib
import yfinance as yf
import google.generativeai as genai
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
import re
import random

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

    range_name = 'A:A'
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

def formatear_numero(numero):
    if pd.isna(numero) or numero == "N/A" or numero is None:
        return "N/A"
    try:
        num = float(numero)
        if abs(num) >= 1_000_000_000:
            return f"{num / 1_000_000_000:,.2f}B"
        elif abs(num) >= 1_000_000:
            return f"{num / 1_000_000:,.2f}M"
        elif abs(num) >= 1_000:
            return f"{num / 1_000:,.2f}K"
        else:
            return f"{num:,.2f}"
    except (ValueError, TypeError):
        return "N/A"

def calculate_smi_tv(df):
    high = df['High']
    low = df['Low']
    close = df['Close']
    length_k = 10
    length_d = 3
    ema_signal_len = 10
    smooth_period = 5
    hh = high.rolling(window=length_k).max()
    ll = low.rolling(window=length_k).min()
    diff = hh - ll
    rdiff = close - (hh + ll) / 2
    avgrel = rdiff.ewm(span=length_d, adjust=False).mean()
    avgdiff = diff.ewm(span=length_d, adjust=False).mean()
    epsilon = 1e-9
    smi_raw = np.where(
        (avgdiff / 2 + epsilon) != 0,
        (avgrel / (avgdiff / 2 + epsilon)) * 100,
        0.0
    )
    smi_raw = np.clip(smi_raw, -100, 100)
    smi_smoothed = pd.Series(smi_raw, index=df.index).rolling(window=smooth_period).mean()
    df['SMI'] = smi_smoothed
    return df

def obtener_datos_yfinance(ticker, periodo="1d", intervalo="1d"):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period=period, interval=interval)
        if hist.empty or len(hist) < 2:
            print(f"⚠️ Advertencia: No hay suficientes datos para {ticker} en el periodo {periodo} e intervalo {intervalo}. Saltando...")
            return None, None
        hist = calculate_smi_tv(hist)
        return info, hist
    except Exception as e:
        print(f"❌ Error al obtener datos de {ticker}: {e}")
        return None, None

def analizar_giro_diario(ticker):
    info, hist_diario = obtener_datos_yfinance(ticker, periodo="60d", intervalo="1d")
    if not info or hist_diario is None:
        return None

    smi_series = hist_diario['SMI'].dropna()
    if len(smi_series) < 2:
        return None

    smi_yesterday = smi_series.iloc[-2]
    smi_today = smi_series.iloc[-1]
    
    pendientes_smi = smi_series.diff()
    pendiente_hoy = pendientes_smi.iloc[-1]
    pendiente_ayer = pendientes_smi.iloc[-2] if len(pendientes_smi) > 1 else 0

    giro = "No"
    tipo_giro = "N/A"
    if pendiente_hoy > 0 and pendiente_ayer <= 0:
        giro = "Sí"
        tipo_giro = "Compra"
    elif pendiente_hoy < 0 and pendiente_ayer >= 0:
        giro = "Sí"
        tipo_giro = "Venta"

    if giro == "Sí":
        return {
            "TICKER": ticker,
            "NOMBRE_EMPRESA": info.get("longName", ticker),
            "PRECIO_ACTUAL": info.get("currentPrice", "N/A"),
            "SMI_ANTERIOR": smi_yesterday,
            "SMI_ACTUAL": smi_today,
            "TIPO_GIRO": tipo_giro
        }
    return None

def analizar_giro_horario(ticker):
    info, hist_horario = obtener_datos_yfinance(ticker, periodo="5d", intervalo="15m")
    if not info or hist_horario is None:
        return None

    smi_series = hist_horario['SMI'].dropna()
    if len(smi_series) < 16:  # 4 horas = 16 intervalos de 15m
        return None

    smi_hace_4h = smi_series.iloc[-16]
    smi_actual = smi_series.iloc[-1]
    
    pendientes_smi = smi_series.diff()
    pendiente_hace_4h_atras = pendientes_smi.iloc[-16]
    pendiente_ahora = pendientes_smi.iloc[-1]

    giro = "No"
    tipo_giro = "N/A"
    if pendiente_ahora > 0 and pendiente_hace_4h_atras <= 0:
        giro = "Sí"
        tipo_giro = "Compra"
    elif pendiente_ahora < 0 and pendiente_hace_4h_atras >= 0:
        giro = "Sí"
        tipo_giro = "Venta"
    
    return {
        "TICKER": ticker,
        "NOMBRE_EMPRESA": info.get("longName", ticker),
        "PRECIO_ACTUAL": info.get("currentPrice", "N/A"),
        "SMI_ANTERIOR": smi_hace_4h,
        "SMI_ACTUAL": smi_actual,
        "GIRO_DETECTADO": giro,
        "TIPO_GIRO": tipo_giro
    }


def enviar_email(html_body, asunto_email):
    remitente = "xumkox@gmail.com"
    destinatario = "xumkox@gmail.com"
    password = "kdgz lvdo wqvt vfkt"
    msg = MIMEMultipart("alternative")
    msg['From'] = remitente
    msg['To'] = destinatario
    msg['Subject'] = asunto_email
    msg.attach(MIMEText(html_body, 'html'))

    try:
        servidor = smtplib.SMTP('smtp.gmail.com', 587)
        servidor.starttls()
        servidor.login(remitente, password)
        servidor.sendmail(remitente, destinatario, msg.as_string())
        servidor.quit()
        print(f"✅ Correo enviado con el asunto: {asunto_email}")
    except Exception as e:
        print("❌ Error al enviar el correo:", e)

def procesar_y_alertar(tickers):
    alertas_diarias = []
    
    # PRIMERA FASE: Filtrado diario
    for ticker in tickers:
        print(f"🔎 Analizando {ticker} para giros diarios del SMI...")
        alerta_diaria = analizar_giro_diario(ticker)
        if alerta_diaria:
            alertas_diarias.append(alerta_diaria)
        time.sleep(1)

    alertas_horarias = []
    if alertas_diarias:
        # SEGUNDA FASE: Análisis horario solo de las empresas filtradas
        for alerta_diaria in alertas_diarias:
            ticker = alerta_diaria['TICKER']
            print(f"🔬 Analizando {ticker} con más detalle (horario)...")
            alerta_horaria = analizar_giro_horario(ticker)
            if alerta_horaria:
                alertas_horarias.append(alerta_horaria)
            time.sleep(1)

    # Construir el cuerpo del correo
    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .compra {{ color: #1abc9c; font-weight: bold; }}
            .venta {{ color: #e74c3c; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h2>Resumen de Alertas de Giros del SMI - {datetime.today().strftime('%d/%m/%Y %H:%M')}</h2>
        
    """
    if alertas_diarias:
        html_body += """
        <h3>Empresas con Giros Diarios (vs. Ayer)</h3>
        <table>
            <tr>
                <th>Empresa</th>
                <th>Ticker</th>
                <th>Tipo de Giro</th>
                <th>Precio Actual</th>
                <th>SMI (Ayer)</th>
                <th>SMI (Hoy)</th>
            </tr>
        """
        for alerta in alertas_diarias:
            clase_giro = "compra" if alerta['TIPO_GIRO'] == "Compra" else "venta"
            html_body += f"""
            <tr>
                <td>{alerta['NOMBRE_EMPRESA']}</td>
                <td><strong>{alerta['TICKER']}</strong></td>
                <td class="{clase_giro}">{alerta['TIPO_GIRO']}</td>
                <td>{formatear_numero(alerta['PRECIO_ACTUAL'])}€</td>
                <td>{alerta['SMI_ANTERIOR']:.2f}</td>
                <td>{alerta['SMI_ACTUAL']:.2f}</td>
            </tr>
            """
        html_body += "</table>"
    else:
        html_body += "<p>No se detectaron giros significativos diarios.</p>"

    html_body += "<br/>"

    if alertas_horarias:
        html_body += """
        <h3>Detalle Horario de las Empresas Filtradas (vs. 4 horas atrás)</h3>
        <table>
            <tr>
                <th>Empresa</th>
                <th>Ticker</th>
                <th>Tipo de Giro</th>
                <th>Precio Actual</th>
                <th>SMI (Hace 4h)</th>
                <th>SMI (Ahora)</th>
            </tr>
        """
        for alerta in alertas_horarias:
            clase_giro = "compra" if alerta['TIPO_GIRO'] == "Compra" else "venta"
            html_body += f"""
            <tr>
                <td>{alerta['NOMBRE_EMPRESA']}</td>
                <td><strong>{alerta['TICKER']}</strong></td>
                <td class="{clase_giro}">{alerta['TIPO_GIRO']}</td>
                <td>{formatear_numero(alerta['PRECIO_ACTUAL'])}€</td>
                <td>{alerta['SMI_ANTERIOR']:.2f}</td>
                <td>{alerta['SMI_ACTUAL']:.2f}</td>
            </tr>
            """
        html_body += "</table>"
    else:
        html_body += "<p>No hay datos horarios para mostrar de las empresas filtradas.</p>"

    html_body += """
        <p><strong>Recuerda:</strong> Un giro del logaritmo es una señal, no una garantía. Utiliza esta información con tu propio análisis y criterio. ¡Feliz trading!</p>
    </body>
    </html>
    """

    asunto = f"📊 Alertas IBEXIA: Giros detectados hoy {datetime.today().strftime('%d/%m/%Y')}"
    enviar_email(html_body, asunto)

def main():
    try:
        all_tickers = leer_google_sheets()[1:]
        if not all_tickers:
            print("No hay tickers para procesar.")
            return
        
        # Se elimina el filtro de 10 tickers al día y se analizan todos
        procesar_y_alertar(all_tickers)

    except Exception as e:
        print(f"❌ Error al ejecutar el script principal: {e}")

if __name__ == '__main__':
    main()
