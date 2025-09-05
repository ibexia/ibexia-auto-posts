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
    smi_signal = smi_smoothed.ewm(span=ema_signal_len, adjust=False).mean()
    df['SMI'] = smi_smoothed
    return df

def obtener_datos_yfinance(ticker, intervalo, periodo):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist_extended = stock.history(period=periodo, interval=intervalo)
        
        if hist_extended.empty:
            print(f"⚠️ Advertencia: No se encontraron datos históricos para {ticker} en el intervalo {intervalo}. Saltando...")
            return None, None
        
        hist_extended = calculate_smi_tv(hist_extended)
        smi_series = hist_extended['SMI'].dropna()
        
        if len(smi_series) < 2:
            print(f"⚠️ Advertencia: No hay suficientes datos de SMI para {ticker} en el intervalo {intervalo}. Saltando...")
            return None, None
            
        current_price = info.get("currentPrice", "N/A")
        
        smi_yesterday = smi_series.iloc[-2]
        smi_today = smi_series.iloc[-1]
        
        pendientes_smi = smi_series.diff()
        pendiente_yesterday = pendientes_smi.iloc[-1]
        
        tendencia_hoy = "alcista" if pendiente_yesterday > 0 else "bajista"
        
        giro = "No"
        tipo_giro = "N/A"
        
        if len(smi_series) >= 3:
            pendiente_anteayer = smi_series.diff().iloc[-2]
            
            if pendiente_yesterday > 0 and pendiente_anteayer <= 0:
                giro = "Sí"
                tipo_giro = "Compra"
            elif pendiente_yesterday < 0 and pendiente_anteayer >= 0:
                giro = "Sí"
                tipo_giro = "Venta"

        return {
            "TICKER": ticker,
            "NOMBRE_EMPRESA": info.get("longName", ticker),
            "PRECIO_ACTUAL": current_price,
            "SMI_AYER": smi_yesterday,
            "SMI_HOY": smi_today,
            "GIRO_DETECTADO": giro,
            "TIPO_GIRO": tipo_giro,
            "TENDENCIA_ACTUAL": tendencia_hoy,
            "INTERVALO": intervalo
        }, hist_extended
        
    except Exception as e:
        print(f"❌ Error al obtener datos de {ticker} en el intervalo {intervalo}: {e}. Saltando a la siguiente empresa...")
        return None, None

def detectar_giros_diarios(tickers):
    giros_diarios = []
    
    for ticker in tickers:
        print(f"🔎 Analizando {ticker} para giros del SMI (diario)...")
        data, _ = obtener_datos_yfinance(ticker, '1d', '60d')
        if data and data['GIRO_DETECTADO'] == "Sí":
            giros_diarios.append(data)
        time.sleep(1)
        
    return giros_diarios

def analizar_detalles_horarios(tickers_con_giro):
    detalles_horarios = []
    
    for ticker_data in tickers_con_giro:
        ticker = ticker_data['TICKER']
        print(f"📈 Analizando detalles horarios para {ticker}...")
        data, _ = obtener_datos_yfinance(ticker, '4h', '5d')
        if data:
            detalles_horarios.append(data)
        time.sleep(1)
        
    return detalles_horarios

def enviar_email(alertas_diarias, detalles_horarios):
    remitente = "xumkox@gmail.com"
    destinatario = "xumkox@gmail.com"
    password = "kdgz lvdo wqvt vfkt"
    
    if not alertas_diarias:
        asunto = f"📊 Alertas IBEXIA: Sin giros significativos hoy {datetime.today().strftime('%d/%m/%Y')}"
        html_body = f"""
        <html>
        <body>
            <h2>Resumen de Alertas de Giros del SMI - {datetime.today().strftime('%d/%m/%Y')}</h2>
            <p>No se detectaron giros significativos de compra o venta en ninguna de las empresas analizadas hoy en el análisis diario.</p>
            <p>Se mantendrá la vigilancia para futuras oportunidades.</p>
        </body>
        </html>
        """
    else:
        asunto = f"🔔 Alertas IBEXIA: Giros en {len(alertas_diarias)} empresas hoy {datetime.today().strftime('%d/%m/%Y')}"
        
        # Construir tabla de giros diarios
        html_tabla_diaria = """
        <h3>Resumen de Giros Diarios (Fase 1)</h3>
        <p>Se han detectado los siguientes giros en el análisis diario que activan el análisis granular por horas:</p>
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
            tipo_giro = alerta['TIPO_GIRO']
            clase_giro = "compra" if tipo_giro == "Compra" else "venta"
            html_tabla_diaria += f"""
            <tr>
                <td>{alerta['NOMBRE_EMPRESA']}</td>
                <td><strong>{alerta['TICKER']}</strong></td>
                <td class="{clase_giro}">{tipo_giro}</td>
                <td>{formatear_numero(alerta['PRECIO_ACTUAL'])}€</td>
                <td>{alerta['SMI_AYER']:.2f}</td>
                <td>{alerta['SMI_HOY']:.2f}</td>
            </tr>
            """
        html_tabla_diaria += "</table>"
        
        # Construir tabla de detalles horarios
        if detalles_horarios:
            html_tabla_horaria = """
            <br>
            <h3>Análisis Detallado Horario (Fase 2)</h3>
            <p>A continuación, el comportamiento del SMI en las últimas 4 horas para las empresas filtradas:</p>
            <table>
                <tr>
                    <th>Empresa</th>
                    <th>Ticker</th>
                    <th>Tendencia 4H</th>
                    <th>SMI (Último)</th>
                </tr>
            """
            for detalle in detalles_horarios:
                tipo_giro = detalle['TIPO_GIRO']
                clase_giro = "compra" if tipo_giro == "Compra" else "venta"
                html_tabla_horaria += f"""
                <tr>
                    <td>{detalle['NOMBRE_EMPRESA']}</td>
                    <td><strong>{detalle['TICKER']}</strong></td>
                    <td class="{clase_giro}">{detalle['TENDENCIA_ACTUAL']}</td>
                    <td>{detalle['SMI_HOY']:.2f}</td>
                </tr>
                """
            html_tabla_horaria += "</table>"
        else:
            html_tabla_horaria = "<br><p>No se encontraron datos horarios para el análisis detallado.</p>"

        # Unir ambas tablas en el cuerpo del email
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
                .neutral {{ color: #34495e; }}
                .header-compra {{ background-color: #d1f2eb; }}
                .header-venta {{ background-color: #fadbd8; }}
            </style>
        </head>
        <body>
            <h2>Alertas de Giros del SMI - {datetime.today().strftime('%d/%m/%Y')}</h2>
            <p>Se ha completado el análisis de dos fases. Aquí están los resultados:</p>
            {html_tabla_diaria}
            {html_tabla_horaria}
            <p><strong>Recuerda:</strong> Un giro del logaritmo es una señal, no una garantía. Utiliza esta información con tu propio análisis y criterio. ¡Feliz trading!</p>
        </body>
        </html>
        """

    msg = MIMEMultipart("alternative")
    msg['From'] = remitente
    msg['To'] = destinatario
    msg['Subject'] = asunto
    msg.attach(MIMEText(html_body, 'html'))

    try:
        servidor = smtplib.SMTP('smtp.gmail.com', 587)
        servidor.starttls()
        servidor.login(remitente, password)
        servidor.sendmail(remitente, destinatario, msg.as_string())
        servidor.quit()
        print(f"✅ Correo enviado con el asunto: {asunto}")
    except Exception as e:
        print("❌ Error al enviar el correo:", e)


def main():
    try:
        all_tickers = leer_google_sheets()[1:]
        if not all_tickers:
            print("No hay tickers para procesar.")
            return

        # Fase 1: Análisis diario para filtrar
        alertas_diarias = detectar_giros_diarios(all_tickers)

        # Fase 2: Análisis horario para las empresas filtradas
        detalles_horarios = []
        if alertas_diarias:
            detalles_horarios = analizar_detalles_horarios(alertas_diarias)

        # Enviar el único correo con ambas tablas
        enviar_email(alertas_diarias, detalles_horarios)

    except Exception as e:
        print(f"❌ Error al ejecutar el script principal: {e}")

if __name__ == '__main__':
    main()
