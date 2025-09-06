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

def calcular_precio_aplanamiento(df):
    try:
        if len(df) < 3:
            return "N/A"

        length_d = 3
        smooth_period = 5

        smi_smoothed_prev = df['SMI'].iloc[-2]

        df_prev = df.iloc[:-1]
        df_prev = calculate_smi_tv(df_prev)

        avgrel_prev_last = (df_prev['Close'] - (df_prev['High'].rolling(window=10).max() + df_prev['Low'].rolling(window=10).min()) / 2).ewm(span=length_d, adjust=False).mean().iloc[-1]
        avgdiff_prev_last = (df_prev['High'].rolling(window=10).max() - df_prev['Low'].rolling(window=10).min()).ewm(span=length_d, adjust=False).mean().iloc[-1]

        alpha_ema = 2 / (length_d + 1)
        
        df_temp = df.copy()
        df_temp['SMI'] = pd.Series(df_temp['SMI'], index=df_temp.index).rolling(window=smooth_period).mean()
        smi_raw_yesterday = df_temp['SMI'].iloc[-2]

        hh_today = df['High'].rolling(window=10).max().iloc[-1]
        ll_today = df['Low'].rolling(window=10).min().iloc[-1]
        diff_today = hh_today - ll_today
        
        avgdiff_today = (1 - alpha_ema) * avgdiff_prev_last + alpha_ema * diff_today
        
        avgrel_today_target = (smi_raw_yesterday / 100) * (avgdiff_today / 2)
        
        rdiff_today_target = (avgrel_today_target - (1 - alpha_ema) * avgrel_prev_last) / alpha_ema
        
        close_target = rdiff_today_target + (hh_today + ll_today) / 2
        
        return close_target

    except Exception as e:
        print(f"❌ Error en el cálculo de precio de aplanamiento: {e}")
        return "N/A"

def obtener_datos_yfinance(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist_extended = stock.history(period="60d", interval="1d")
        if hist_extended.empty:
            print(f"⚠️ Advertencia: No se encontraron datos históricos para {ticker}. Saltando...")
            return None
        hist_extended = calculate_smi_tv(hist_extended)

        smi_series = hist_extended['SMI'].dropna()
        if len(smi_series) < 2:
            print(f"⚠️ Advertencia: No hay suficientes datos de SMI para {ticker}. Saltando...")
            return None

        current_price = info.get("currentPrice", "N/A")
        
        smi_yesterday = smi_series.iloc[-2]
        smi_today = smi_series.iloc[-1]
        
        pendiente_hoy = smi_today - smi_yesterday
        
        tendencia_hoy = "alcista" if pendiente_hoy > 0 else "bajista"
        
        estado_smi = "Sobrecompra" if smi_today > 60 else ("Sobreventa" if smi_today < -60 else "Intermedio")
        
        giro = "No"
        tipo_giro = "N/A"
        if len(smi_series) >= 3:
            pendiente_anteayer = smi_series.iloc[-2] - smi_series.iloc[-3]
            
            if pendiente_hoy > 0 and pendiente_anteayer <= 0:
                giro = "Sí"
                tipo_giro = "Compra"
            elif pendiente_hoy < 0 and pendiente_anteayer >= 0:
                giro = "Sí"
                tipo_giro = "Venta"
        
        precio_aplanamiento = calcular_precio_aplanamiento(hist_extended)

        return {
            "TICKER": ticker,
            "NOMBRE_EMPRESA": info.get("longName", ticker),
            "PRECIO_ACTUAL": current_price,
            "SMI_AYER": smi_yesterday,
            "SMI_HOY": smi_today,
            "GIRO_DETECTADO": giro,
            "TIPO_GIRO": tipo_giro,
            "TENDENCIA_ACTUAL": tendencia_hoy,
            "ESTADO_SMI": estado_smi,
            "PRECIO_APLANAMIENTO": precio_aplanamiento,
            "PENDIENTE": pendiente_hoy
        }

    except Exception as e:
        print(f"❌ Error al obtener datos de {ticker}: {e}. Saltando a la siguiente empresa...")
        return None

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

def clasificar_fuerza_senial(variacion):
    abs_variacion = abs(variacion)
    if abs_variacion <= 0.5:
        return "Sin confirmación clara"
    elif abs_variacion <= 1.6:
        return "Señal moderada"
    else:
        return "Buena señal"
        
def generar_recomendacion(data):
    tendencia = data['TENDENCIA_ACTUAL']
    precio_actual = data['PRECIO_ACTUAL']
    precio_aplanamiento = data['PRECIO_APLANAMIENTO']
    diferencia_porcentual = (precio_aplanamiento - precio_actual) / precio_actual if precio_actual != "N/A" and precio_aplanamiento != "N/A" and precio_actual != 0 else 0

    if abs(diferencia_porcentual) <= 0.005:  # Si está dentro del 0.5%
        if tendencia == "alcista":
            return "Señal de VENTA ACTIVADA"
        else:
            return "Señal de COMPRA ACTIVADA"
    
    if tendencia == "bajista":
        # SMI bajando, necesita subir para aplanarse -> Recomendación de COMPRA si se supera un precio
        return f"Compra si supera {formatear_numero(precio_aplanamiento)}€"
    
    if tendencia == "alcista":
        # SMI subiendo, necesita bajar para aplanarse -> Recomendación de VENTA si se baja de un precio
        return f"Vende si baja de {formatear_numero(precio_aplanamiento)}€"

    return "No aplica"


def detectar_giros_y_alertar(tickers):
    alertas_giros = []
    datos_completos = []

    for ticker in tickers:
        print(f"🔎 Analizando {ticker} para giros del SMI...")
        data = obtener_datos_yfinance(ticker)
        if data:
            datos_completos.append(data)
            if data['GIRO_DETECTADO'] == "Sí":
                alertas_giros.append(data)
        time.sleep(1)

    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            h2 {{ color: #2c3e50; }}
            p {{ color: #7f8c8d; }}
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
        <h2>Resumen Diario de Alertas y Oportunidades - {datetime.today().strftime('%d/%m/%Y')}</h2>
        
        <h3>Alerta de Giros del Algoritmo</h3>
    """

    if not alertas_giros:
        html_body += """
            <p>No se detectaron giros significativos de compra o venta en las empresas analizadas hoy.</p>
        """
    else:
        for alerta in alertas_giros:
            alerta['variacion_Algoritmo'] = alerta['SMI_HOY'] - alerta['SMI_AYER']
        
        alertas_giros.sort(key=lambda x: abs(x['variacion_Algoritmo']), reverse=True)

        html_body += """
            <p>Se han detectado los siguientes giros en nuestro Algoritmo que podrían indicar posibles oportunidades de trading. Los giros están ordenados por su fuerza, siendo los primeros los más claros:</p>
            <table>
                <tr>
                    <th>Empresa</th>
                    <th>Tipo de Giro</th>
                    <th>Precio Actual</th>
                    <th>FUERZA DE LA SEÑAL</th>
                </tr>
        """
        for alerta in alertas_giros:
            tipo_giro = alerta['TIPO_GIRO']
            clase_giro = "compra" if tipo_giro == "Compra" else "venta"
            fuerza_senial = clasificar_fuerza_senial(alerta['variacion_Algoritmo'])
            html_body += f"""
                <tr>
                    <td>{alerta['NOMBRE_EMPRESA']}</td>
                    <td class="{clase_giro}">{tipo_giro}</td>
                    <td>{formatear_numero(alerta['PRECIO_ACTUAL'])}€</td>
                    <td>{fuerza_senial}</td>
                </tr>
            """
        html_body += "</table>"
    
    html_body += """
        <hr>
        <h3>Análisis de Proximidad al Giro</h3>
        <p>Esta tabla muestra la distancia del precio actual al punto de aplanamiento del SMI. Un valor positivo indica que el precio debe subir para aplanar la curva, y un valor negativo que debe bajar:</p>
        <table>
            <tr>
                <th>Empresa</th>
                <th>Precio Actual</th>
                <th>Estado del SMI</th>
                <th>TENDENCIA ACTUAL</th>
                <th>Precio para Aplanar el SMI</th>
                <th>Diferencia %</th>
                <th>Acción Recomendada</th>
            </tr>
    """

    for data in datos_completos:
        precio_actual = data['PRECIO_ACTUAL']
        precio_aplanamiento = data['PRECIO_APLANAMIENTO']
        tendencia_actual_str = "Subiendo (Alcista)" if data['TENDENCIA_ACTUAL'] == "alcista" else "Bajando (Bajista)"
        recomendacion = generar_recomendacion(data)
        
        if precio_actual != "N/A" and precio_aplanamiento != "N/A":
            try:
                diferencia_porcentual = ((precio_aplanamiento - precio_actual) / precio_actual) * 100
                diferencia_str = f"{diferencia_porcentual:.2f}%"
            except (ValueError, TypeError, ZeroDivisionError):
                diferencia_str = "N/A"
        else:
            diferencia_str = "N/A"
            
        html_body += f"""
            <tr>
                <td>{data['NOMBRE_EMPRESA']}</td>
                <td>{formatear_numero(precio_actual)}€</td>
                <td>{data['ESTADO_SMI']}</td>
                <td>{tendencia_actual_str}</td>
                <td>{formatear_numero(precio_aplanamiento)}€</td>
                <td>{diferencia_str}</td>
                <td>{recomendacion}</td>
            </tr>
        """
    
    html_body += """
        </table>
        <br>
        <p><strong>Recuerda:</strong> Un aplanamiento de la curva no garantiza un giro inmediato, pero puede señalar que la fuerza de la tendencia actual está disminuyendo. Utiliza esta información con tu propio análisis y criterio. ¡Feliz trading!</p>
    </body>
    </html>
    """
    
    asunto = f"🔔 Alertas y Proximidad IBEXIA: {len(alertas_giros)} giros detectados hoy {datetime.today().strftime('%d/%m/%Y')}"
    enviar_email(html_body, asunto)


def main():
    try:
        all_tickers = leer_google_sheets()[1:]
        if not all_tickers:
            print("No hay tickers para procesar.")
            return

        detectar_giros_y_alertar(all_tickers)

    except Exception as e:
        print(f"❌ Error al ejecutar el script principal: {e}")

if __name__ == '__main__':
    main()
