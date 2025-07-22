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

# Configuración de Google Sheets (no modificada)
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

    range_name = 'A:A'  # Se fuerza el rango a 'A:A' para leer toda la columna A

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

# Configuración de Gemini (no modificada)
def configurar_gemini():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("La variable de entorno GEMINI_API_KEY no está configurada.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

# Función para enviar correo electrónico (no modificada)
def enviar_email(subject, body, to_email):
    sender_email = os.getenv('SENDER_EMAIL')
    sender_password = os.getenv('SENDER_PASSWORD')
    smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.getenv('SMTP_PORT', 587))

    if not all([sender_email, sender_password, to_email]):
        print("Advertencia: Faltan credenciales de correo o destinatario. No se enviará el email.")
        return

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'html'))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"Email enviado a {to_email} con el asunto: {subject}")
    except Exception as e:
        print(f"Error al enviar el email: {e}")

# Función principal para obtener datos de Yahoo Finance y calcular indicadores
def obtener_datos_yfinance(ticker):
    print(f"Buscando datos para el ticker: {ticker}")
    try:
        data = yf.Ticker(ticker)
        # Obtener más datos históricos (por ejemplo, 1 año) para asegurar el cálculo del SMI
        hist_full = data.history(period="1y")

        if hist_full.empty:
            print(f"⚠️ Advertencia: No se encontraron datos históricos para {ticker}.")
            return None

        # --- Cálculo del SMI (basado en TradingView) ---
        def calculate_smi_tv(df_input, length=20, ema_length=5, signal_length=5):
            df_copy = df_input.copy() # Trabajar en una copia para evitar SettingWithCopyWarning
            
            # Asegurarse de que las columnas necesarias existan
            if not all(col in df_copy.columns for col in ['Open', 'High', 'Low', 'Close']):
                print(f"DEBUG: DataFrame de entrada a calculate_smi_tv no tiene todas las columnas OHLC necesarias.")
                return df_copy # Retorna el DF sin SMI si faltan columnas

            # Calcular el rango más bajo y más alto de 'length' períodos
            low_min = df_copy['Low'].rolling(window=length, min_periods=1).min()
            high_max = df_copy['High'].rolling(window=length, min_periods=1).max()

            # Calcular el cambio en el cierre y el rango total
            range_diff = df_copy['Close'] - ((high_max + low_min) / 2)
            total_range = high_max - low_min

            # Evitar división por cero y NaN si total_range es 0
            smi = pd.Series(np.where(total_range == 0, 0, range_diff / (total_range / 2)), index=df_copy.index)
            smi = smi.fillna(0) # Rellenar NaN resultantes de divisiones iniciales con 0

            # Primera EMA (smi_raw)
            df_copy['SMI_raw'] = smi.ewm(span=ema_length, adjust=False, min_periods=1).mean()
            df_copy['SMI_raw'] = df_copy['SMI_raw'].fillna(0) # Asegurar no NaN

            # Segunda EMA (smi_smoothed, que es el SMI principal)
            df_copy['SMI'] = df_copy['SMI_raw'].ewm(span=ema_length, adjust=False, min_periods=1).mean()
            df_copy['SMI'] = df_copy['SMI'].fillna(0) # Asegurar no NaN

            # EMA de la señal (signal_line)
            df_copy['SMI_signal'] = df_copy['SMI'].ewm(span=signal_length, adjust=False, min_periods=1).mean()
            df_copy['SMI_signal'] = df_copy['SMI_signal'].fillna(0) # Asegurar no NaN

            return df_copy

        hist = calculate_smi_tv(hist_full.tail(60)) # Usar los últimos 60 días para cálculos cercanos


        if 'Close' not in hist.columns or hist['Close'].empty:
            print(f"⚠️ Advertencia: No hay datos de cierre disponibles para {ticker}.")
            return None

        current_price = hist['Close'].iloc[-1] if not hist['Close'].empty else 0
        volume = hist['Volume'].iloc[-1] if not hist['Volume'].empty else 0
        average_volume = hist['Volume'].tail(30).mean() if not hist['Volume'].empty else 0

        # Obtener valores actuales de SMI
        smi_actual = round(hist['SMI'].iloc[-1], 2) if 'SMI' in hist.columns and not hist['SMI'].empty and pd.notna(hist['SMI'].iloc[-1]) else 0
        smi_raw_actual = round(hist['SMI_raw'].iloc[-1], 2) if 'SMI_raw' in hist.columns and not hist['SMI_raw'].empty and pd.notna(hist['SMI_raw'].iloc[-1]) else smi_actual

        if smi_actual == 0:
            print(f"⚠️ Advertencia: No hay datos de SMI suavizado válidos para {ticker}. Asignando SMI neutral.")
        if smi_raw_actual == 0:
            print(f"⚠️ Advertencia: No hay datos de SMI RAW válidos para {ticker}. Usando SMI suavizado como fallback.")

        # Determinar tendencia del SMI
        tendencia_smi = "Neutral"
        if 'SMI' in hist.columns and len(hist['SMI'].dropna()) >= 5: # Necesitamos al menos 5 puntos para una pequeña tendencia
            smi_last_5 = hist['SMI'].dropna().tail(5)
            if not smi_last_5.empty:
                if smi_last_5.iloc[-1] > smi_last_5.iloc[0]:
                    tendencia_smi = "Ascendente"
                elif smi_last_5.iloc[-1] < smi_last_5.iloc[0]:
                    tendencia_smi = "Descendente"

        # Calcular RSI - FIX para el error 'numpy.ndarray' object has no attribute 'empty'
        rsi_actual = 50.0 # Default value
        if len(hist['Close']) > 1: # Need at least 2 data points for diff() to be meaningful
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False, min_periods=1).mean()
            
            # Use pandas division to ensure Series output, then handle inf/NaN
            # Replace 0s in loss with NaN to avoid division by zero
            rs = gain / loss.replace(0, np.nan) 
            rs = rs.replace([np.inf, -np.inf], np.nan).fillna(np.inf) # Handle division by zero giving inf, then fill NaNs from division by zero with inf

            rsi = 100 - (100 / (1 + rs))
            
            if not rsi.empty and pd.notna(rsi.iloc[-1]):
                rsi_actual = float(rsi.iloc[-1])


        condicion_rsi = "Indeterminada"
        if rsi_actual >= 70:
            condicion_rsi = "Sobrecompra"
        elif rsi_actual <= 30:
            condicion_rsi = "Sobreventa"
        else:
            condicion_rsi = "Neutral"

        # Precios históricos para el gráfico (últimos 30 días para coherencia)
        cierres_para_grafico_full = hist['Close'].dropna().tolist()

        smi_history_full = hist['SMI'].dropna() # Asegúrate de que SMI tiene datos numéricos aquí
        smi_raw_history_full = hist['SMI_raw'].dropna() # Asegúrate de que SMI_raw tiene datos numéricos aquí


        # Datos para los últimos 30 días para el gráfico
        cierres_para_grafico_total = []
        if len(cierres_para_grafico_full) >= 30:
            cierres_para_grafico_total = cierres_para_grafico_full[-30:]
        elif cierres_para_grafico_full:
            cierres_para_grafico_total = cierres_para_grafico_full
            # Rellenar con el primer valor para que tenga al menos 30 puntos si es posible
            if len(cierres_para_grafico_total) < 30:
                padding_needed = 30 - len(cierres_para_grafico_total)
                cierres_para_grafico_total = [cierres_para_grafico_total[0]] * padding_needed + cierres_para_grafico_total if cierres_para_grafico_total else [0.0] * 30
        else:
            cierres_para_grafico_total = [0.0] * 30 # Rellenar con ceros si no hay datos


        smi_historico_para_grafico = []
        if len(smi_history_full) >= 30:
            smi_historico_para_grafico = smi_history_full.tail(30).tolist()
        elif not smi_history_full.empty: # Si hay datos pero menos de 30
            first_smi_val = smi_history_full.iloc[0]
            smi_historico_para_grafico = [first_smi_val] * (30 - len(smi_history_full)) + smi_history_full.tolist()
        else: # Si está completamente vacío
            smi_historico_para_grafico = [0.0] * 30

        smi_raw_historico_para_grafico = []
        if len(smi_raw_history_full) >= 30:
            smi_raw_historico_para_grafico = smi_raw_history_full.tail(30).tolist()
        elif not smi_raw_history_full.empty: # Si hay datos pero menos de 30
            first_smi_raw_val = smi_raw_history_full.iloc[0]
            smi_raw_historico_para_grafico = [first_smi_raw_val] * (30 - len(smi_raw_history_full)) + smi_raw_history_full.tolist()
        else: # Si está completamente vacío
            smi_raw_historico_para_grafico = [0.0] * 30


        # Definir niveles de soporte y resistencia de forma simplificada
        if len(hist['Close']) > 10:
            soporte_1 = round(hist['Low'].tail(10).min() * 0.98, 2)
            soporte_2 = round(hist['Low'].tail(20).min() * 0.95, 2)
            soporte_3 = round(hist['Low'].min() * 0.92, 2)
            resistencia_1 = round(hist['High'].tail(10).max() * 1.02, 2)
            resistencia_2 = round(hist['High'].tail(20).max() * 1.05, 2)
            resistencia_3 = round(hist['High'].max() * 1.08, 2)
        else:
            soporte_1, soporte_2, soporte_3 = current_price * 0.95, current_price * 0.90, current_price * 0.85
            resistencia_1, resistencia_2, resistencia_3 = current_price * 1.05, current_price * 1.10, current_price * 1.15

        # Nota de la empresa (basado en SMI y RSI, ejemplo simplificado)
        nota_empresa = round((smi_actual + rsi_actual) / 20, 1)
        if nota_empresa > 10: nota_empresa = 10
        if nota_empresa < 0: nota_empresa = 0


        # Tendencia de la nota
        tendencia_nota = "Estable"
        if smi_actual > 40 and rsi_actual > 50: # SMI y RSI en zona positiva
            tendencia_nota = "Ascendente"
        elif smi_actual < -40 and rsi_actual < 50: # SMI y RSI en zona negativa
            tendencia_nota = "Descendente"

        # Recomendación
        recomendacion = "Neutral"
        if data.info.get('regularMarketPrice', None) is not None: # Asegurar que hay precio actual
            if smi_actual > 40 and rsi_actual > 50:
                recomendacion = "Compra"
            elif smi_actual < -40 and rsi_actual < 50:
                recomendacion = "Venta"
            elif -40 <= smi_actual <= 40 and 40 <= rsi_actual <= 60:
                recomendacion = "Neutral"

        motivo_recomendacion = "Basado en una combinación de SMI y RSI. Se requiere un análisis más profundo para decisiones de inversión."
        if recomendacion == "Compra":
            motivo_recomendacion = f"Los indicadores SMI ({smi_actual:.2f}) y RSI ({rsi_actual:.2f}) muestran una fuerte señal alcista, con el precio consolidándose por encima del soporte de {soporte_1:.2f}€."
        elif recomendacion == "Venta":
            motivo_recomendacion = f"Los indicadores SMI ({smi_actual:.2f}) y RSI ({rsi_actual:.2f}) sugieren una debilidad, con el precio acercándose a la resistencia de {resistencia_1:.2f}€."
        elif recomendacion == "Neutral":
            motivo_recomendacion = f"Los indicadores SMI ({smi_actual:.2f}) y RSI ({rsi_actual:.2f}) se encuentran en zonas neutrales, indicando consolidación o falta de dirección clara."


        # Precio objetivo (simplificado)
        precio_objetivo = round(current_price * 1.05, 2) if recomendacion == "Compra" else round(current_price * 0.95, 2)
        # Precio objetivo de compra (puede ser N/A)
        precio_objetivo_compra = round(current_price * 0.98, 2) if recomendacion == "Compra" else "N/A"


        # Preparar datos para el prompt
        datos = {
            "NOMBRE_EMPRESA": data.info.get('longName', ticker),
            "TICKER": ticker,
            "PRECIO_ACTUAL": current_price,
            "VOLUMEN": volume,
            "VOLUMEN_MEDIO": average_volume,
            "SMI": smi_actual,
            "SMI_RAW": smi_raw_actual,
            "TENDENCIA_SMI": tendencia_smi,
            "RSI": rsi_actual,
            "CONDICION_RSI": condicion_rsi,
            "SOPORTE_1": soporte_1,
            "SOPORTE_2": soporte_2,
            "SOPORTE_3": soporte_3,
            "RESISTENCIA_1": resistencia_1,
            "RESISTENCIA_2": resistencia_2,
            "RESISTENCIA_3": resistencia_3,
            "RECOMENDACION": recomendacion,
            "NOTA_EMPRESA": nota_empresa,
            "PRECIO_OBJETIVO_COMPRA": precio_objetivo_compra,
            "PRECIO_OBJETIVO": precio_objetivo,
            "TENDENCIA_NOTA": tendencia_nota,
            "MOTIVO_RECOMENDACION": motivo_recomendacion,
            "CIERRES_PARA_GRAFICO_TOTAL": cierres_para_grafico_total,
            "SMI_HISTORICO_PARA_GRAFICO": smi_historico_para_grafico,
            "SMI_RAW_HISTORICO_PARA_GRAFICO": smi_raw_historico_para_grafico,
            "PROYECCION_FUTURA_DIAS": 5
        }
        return datos

    except Exception as e:
        print(f"❌ Error al obtener datos para {ticker}: {e}")
        return None

# Función para formatear números de forma segura
def formatear_numero(numero, decimales=0):
    """
    Formatea un número con separadores de miles y decimales,
    o devuelve 'N/A' si el valor es None o la cadena 'N/A'.
    """
    if numero is None or (isinstance(numero, str) and numero == "N/A"):
        return "N/A"
    try:
        num_val = float(numero) # Asegura que es un número flotante
        # Formatea con separador de miles (.), separador de decimales (,)
        return f"{num_val:,.{decimales}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        # En caso de que el 'numero' no sea un número válido ni 'N/A'
        return str(numero)

# Función para construir el prompt formateado con todos los datos y el gráfico
def construir_prompt_formateado(data):
    ticker = data.get('TICKER', 'N/A')
    titulo_post = f"Análisis Técnico Avanzado: ¡Oportunidad en {data['NOMBRE_EMPRESA']} ({ticker})!"

    volumen_analisis_text = ""
    if data['VOLUMEN'] > data['VOLUMEN_MEDIO'] * 1.5:
        volumen_analisis_text = f"El volumen de {formatear_numero(data['VOLUMEN'], 0)} acciones, un {((data['VOLUMEN'] / data['VOLUMEN_MEDIO']) - 1) * 100:.2f}% superior al promedio, indica un fuerte interés. Esto refuerza mi recomendación de {data['RECOMENDACION']}."
    elif data['VOLUMEN'] < data['VOLUMEN_MEDIO'] * 0.5:
        volumen_analisis_text = f"El volumen de {formatear_numero(data['VOLUMEN'], 0)} acciones, un {((data['VOLUMEN_MEDIO'] / data['VOLUMEN']) - 1) * 100:.2f}% inferior al promedio, sugiere cautela. Esto debilita mi recomendación de {data['RECOMENDACION']}."
    else:
        volumen_analisis_text = f"El volumen de {formatear_numero(data['VOLUMEN'], 0)} acciones, en línea con el promedio, indica un interés normal. Esto apoya mi recomendación de {data['RECOMENDACION']}."

    chart_html = "" # Inicializar chart_html
    cierres_para_grafico_total = data.get('CIERRES_PARA_GRAFICO_TOTAL', [])
    smi_historico_para_grafico = data.get('SMI_HISTORICO_PARA_GRAFICO', [])
    smi_raw_historico_para_grafico = data.get('SMI_RAW_HISTORICO_PARA_GRAFICO', [])
    PROYECCION_FUTURA_DIAS = data.get('PROYECCION_FUTURA_DIAS', 5)
    OFFSET_DIAS = 4 # Este valor se usa para alinear el SMI en el gráfico


    # Solo generar el gráfico si tenemos datos válidos y no vacíos para todos los elementos clave
    if smi_historico_para_grafico and cierres_para_grafico_total and smi_raw_historico_para_grafico and \
       len(smi_historico_para_grafico) > 0 and len(cierres_para_grafico_total) > 0 and len(smi_raw_historico_para_grafico) > 0:

        # Asegurarse de que las listas de SMI tengan al menos 30 elementos para Chart.js
        # Si tienes menos de 30 días, rellena con el primer valor para que el gráfico no falle
        # (Esta lógica ya se maneja en obtener_datos_yfinance, pero se refuerza aquí)
        if len(smi_historico_para_grafico) < 30:
            padding_needed = 30 - len(smi_historico_para_grafico)
            smi_historico_para_grafico = [smi_historico_para_grafico[0]] * padding_needed + smi_historico_para_grafico
        
        if len(smi_raw_historico_para_grafico) < 30:
            padding_needed = 30 - len(smi_raw_historico_para_grafico)
            smi_raw_historico_para_grafico = [smi_raw_historico_para_grafico[0]] * padding_needed + smi_raw_historico_para_grafico

        if len(cierres_para_grafico_total) < 30:
            padding_needed = 30 - len(cierres_para_grafico_total)
            cierres_para_grafico_total = [cierres_para_grafico_total[0]] * padding_needed + cierres_para_grafico_total


        labels_historial = [(datetime.today() - timedelta(days=29 - i)).strftime("%d/%m") for i in range(30)]
        labels_proyeccion = [(datetime.today() + timedelta(days=i)).strftime("%d/%m (fut.)") for i in range(1, PROYECCION_FUTURA_DIAS + 1)]
        labels_total = labels_historial + labels_proyeccion

        precios_reales_grafico = cierres_para_grafico_total[:30]
        data_proyectada = [None] * (len(labels_historial) - 1) + [precios_reales_grafico[-1]] + ([None] * PROYECCION_FUTURA_DIAS) # Proyección inicialmente nula


        # Desplazar el SMI para el gráfico para que coincida con el precio
        smi_desplazados_para_grafico = [None] * OFFSET_DIAS + smi_historico_para_grafico
        if len(smi_desplazados_para_grafico) < len(labels_total):
            smi_desplazados_para_grafico.extend([None] * (len(labels_total) - len(smi_desplazados_para_grafico)))

        smi_raw_desplazados_para_grafico = [None] * OFFSET_DIAS + smi_raw_historico_para_grafico # SMI sin suavizar
        if len(smi_raw_desplazados_para_grafico) < len(labels_total):
            smi_raw_desplazados_para_grafico.extend([None] * (len(labels_total) - len(smi_raw_desplazados_para_grafico)))


        chart_html += f"""
        <h2>Evolución del Stochastic Momentum Index (SMI) y Precio</h2>
        <p>Para ofrecer una perspectiva visual clara de la evolución del SMI y su relación con el precio,
        se presenta el siguiente gráfico. Es importante recordar que el SMI de hoy (D) se alinea con el precio de D+{OFFSET_DIAS},
        lo que significa que la reacción del mercado al SMI generalmente se observa
        unos pocos días después de su formación.
        </p>
        <div style="width: 100%; max-width: 800px; margin: auto;">
            <canvas id="smiPrecioChart" style="height: 600px;"></canvas>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@1.4.0"></script>
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const ctx = document.getElementById('smiPrecioChart').getContext('2d');
            const smiPrecioChart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(labels_total)},
                    datasets: [
                        {{
                            label: 'SMI (Suavizado)',
                            data: {json.dumps(smi_desplazados_para_grafico)},
                            borderColor: 'rgba(75, 192, 192, 1)',
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            yAxisID: 'y',
                            tension: 0.1,
                            fill: false
                        }},
                        {{
                            label: 'SMI (Rápido)',
                            data: {json.dumps(smi_raw_desplazados_para_grafico)},
                            borderColor: 'rgba(255, 159, 64, 1)', // Color diferente para el SMI rápido
                            backgroundColor: 'rgba(255, 159, 64, 0.2)',
                            yAxisID: 'y',
                            tension: 0.1,
                            fill: false,
                            borderDash: [2, 2] // Línea punteada para el rápido
                        }},
                        {{
                            label: 'Precio Actual',
                            data: {json.dumps(precios_reales_grafico)},
                            borderColor: 'rgba(153, 102, 255, 1)',
                            backgroundColor: 'rgba(153, 102, 255, 0.2)',
                            yAxisID: 'y1',
                            tension: 0.1,
                            fill: false
                        }},
                        {{
                            label: 'Precio Proyectado',
                            data: {json.dumps(data_proyectada)},
                            borderColor: 'rgba(255, 99, 132, 1)',
                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                            yAxisID: 'y1',
                            tension: 0.1,
                            fill: false,
                            borderDash: [5, 5]
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {{
                        mode: 'index',
                        intersect: false,
                    }},
                    stacked: false,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Evolución del SMI y Precio'
                        }},
                        annotation: {{
                            annotations: {{
                                futureZone: {{
                                    type: 'box',
                                    xMin: {30},
                                    xMax: {30 + PROYECCION_FUTURA_DIAS},
                                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                                    borderColor: 'rgba(0, 123, 255, 0.3)',
                                    borderWidth: 1,
                                    label: {{
                                        content: 'Zona Futuro',
                                        enabled: true,
                                        position: 'start',
                                        color: 'rgba(0, 123, 255, 0.7)',
                                        font: {{
                                            size: 10,
                                            weight: 'bold'
                                        }}
                                    }}
                                }},
                                zonaSobrecompra: {{
                                    type: 'box',
                                    yScaleID: 'y',
                                    yMin: 40,
                                    yMax: 100,
                                    backgroundColor: 'rgba(255, 0, 0, 0.1)',
                                    borderColor: 'rgba(255, 0, 0, 0.2)',
                                    borderWidth: 1,
                                    label: {{
                                        content: 'Zona de Sobrecompra',
                                        enabled: false,
                                        position: 'end',
                                        color: 'rgba(150, 0, 0, 0.7)',
                                        font: {{ size: 12, weight: 'bold' }}
                                    }}
                                }},
                                zonaNeutralSMI: {{
                                    type: 'box',
                                    yScaleID: 'y',
                                    yMin: -40,
                                    yMax: 40,
                                    backgroundColor: 'rgba(255, 255, 0, 0.1)',
                                    borderColor: 'rgba(255, 255, 0, 0.2)',
                                    borderWidth: 1,
                                    label: {{
                                        content: 'Zona Neutral SMI',
                                        enabled: false,
                                        position: 'center',
                                        color: 'rgba(150, 150, 0, 0.7)',
                                        font: {{ size: 12, weight: 'bold' }}
                                    }}
                                }},
                                zonaSobreventa: {{
                                    type: 'box',
                                    yScaleID: 'y',
                                    yMin: -100,
                                    yMax: -40,
                                    backgroundColor: 'rgba(0, 255, 0, 0.1)',
                                    borderColor: 'rgba(0, 255, 0, 0.2)',
                                    borderWidth: 1,
                                    label: {{
                                        content: 'Zona de Sobreventa',
                                        enabled: false,
                                        position: 'start',
                                        color: 'rgba(0, 150, 0, 0.7)',
                                        font: {{ size: 12, weight: 'bold' }}
                                    }}
                                }},
                                soporte1Line: {{
                                    type: 'line',
                                    yScaleID: 'y1',
                                    yMin: {data['SOPORTE_1']},
                                    yMax: {data['SOPORTE_1']},
                                    borderColor: 'rgba(75, 192, 192, 0.8)',
                                    borderWidth: 2,
                                    borderDash: [6, 6],
                                    label: {{
                                        content: 'Soporte 1',
                                        enabled: true,
                                        position: 'start',
                                        backgroundColor: 'rgba(75, 192, 192, 0.5)',
                                        borderColor: 'rgba(75, 192, 192, 0.8)',
                                        borderRadius: 4,
                                        borderWidth: 1,
                                        font: {{ size: 10, weight: 'bold' }},
                                        color: 'black'
                                    }}
                                }},
                                resistencia1Line: {{
                                    type: 'line',
                                    yScaleID: 'y1',
                                    yMin: {data['RESISTENCIA_1']},
                                    yMax: {data['RESISTENCIA_1']},
                                    borderColor: 'rgba(255, 99, 132, 0.8)',
                                    borderWidth: 2,
                                    borderDash: [6, 6],
                                    label: {{
                                        content: 'Resistencia 1',
                                        enabled: true,
                                        position: 'end',
                                        backgroundColor: 'rgba(255, 99, 132, 0.5)',
                                        borderColor: 'rgba(255, 99, 132, 0.8)',
                                        borderRadius: 4,
                                        borderWidth: 1,
                                        font: {{ size: 10, weight: 'bold' }},
                                        color: 'black'
                                    }}
                                }}
                            }}
                        }}
                    }}
                }});
            }});
        </script>
        """
    else:
        # Mensaje de depuración si los datos del gráfico están vacíos
        print("DEBUG: Datos para el gráfico SMI/Precio están vacíos. No se generará chart_html.")
        print(f"DEBUG: smi_historico_para_grafico vacío: {not bool(smi_historico_para_grafico)}")
        print(f"DEBUG: cierres_para_grafico_total vacío: {not bool(cierres_para_grafico_total)}")
        print(f"DEBUG: smi_raw_historico_para_grafico vacío: {not bool(smi_raw_historico_para_grafico)}")


    # --- INICIO DEL PROMPT FINAL PARA GEMINI ---
    prompt_formateado = f"""
Actúa como un trader profesional con amplia experiencia en análisis técnico y mercados financieros. Genera el análisis completo en **formato HTML**, ideal para publicaciones web. Utiliza etiquetas `<h2>` para los títulos de sección y `<p>` para cada párrafo de texto. Redacta en primera persona, con total confianza en tu criterio.

Destaca los datos importantes como precios, notas de la empresa, cifras financieras y el nombre de la empresa utilizando la etiqueta `<strong>`. Asegúrate de que no haya asteriscos u otros símbolos de marcado en el texto final, solo HTML válido. Asegurate que todo este escrito en español independientemente del idioma de donde saques los datos.

Genera un análisis técnico completo de aproximadamente 800 palabras sobre la empresa {data['NOMBRE_EMPRESA']}, utilizando los siguientes datos reales extraídos de Yahoo Finance. Presta especial atención a la **nota obtenida por la empresa**: {data['NOTA_EMPRESA']}.

¡ATENCIÓN URGENTE! Para CADA EMPRESA analizada, debes generar el CÓDIGO HTML Y JAVASCRIPT COMPLETO y Único para TODOS sus gráficos solicitados (Notas Chart, Divergencia Color Chart, Nota Variación Chart y Precios Chart). Bajo ninguna circunstancia debes omitir ningún script, resumir bloques de código o utilizar frases como 'código JavaScript idéntico al ejemplo anterior'. Cada gráfico, para cada empresa, debe tener su script completamente incrustado, funcional e independiente de otros. Asegúrate de que los datos de cada gráfico corresponden SIEMPRE a la empresa que se está analizando en ese momento

**Datos clave:**
- Precio actual: {formatear_numero(data['PRECIO_ACTUAL'], 2)}
- Volumen del último día completo: {formatear_numero(data['VOLUMEN'], 0)}
- Soporte 1: {formatear_numero(data['SOPORTE_1'], 2)}
- Soporte 2: {formatear_numero(data['SOPORTE_2'], 2)}
- Soporte 3: {formatear_numero(data['SOPORTE_3'], 2)}
- Resistencia clave: {formatear_numero(data['RESISTENCIA_1'], 2)}
- Recomendación general: {data['RECOMENDACION']}
- Nota de la empresa (0-10): {formatear_numero(data['NOTA_EMPRESA'], 1)}
- Precio objetivo de compra: {formatear_numero(data['PRECIO_OBJETIVO_COMPRA'], 2)}€
- Tendencia de la nota: {data['TENDENCIA_NOTA']}


Importante: si algún dato no está disponible ("N/A", "No disponibles", "No disponible"), no lo menciones ni digas que falta. No expliques que la recomendación proviene de un indicador o dato específico. La recomendación debe presentarse como una conclusión personal basada en tu experiencia y criterio profesional como analista. Al redactar el análisis, haz referencia a la **nota obtenida por la empresa ({formatear_numero(data['NOTA_EMPRESA'], 1)})** en al menos dos de los párrafos principales (Recomendación General, Análisis a Corto Plazo o Predicción a Largo Plazo) como un factor clave para tu valoración.

---
<h1>{titulo_post}</h1>


<h2>Análisis Inicial y Recomendación</h2>
<p><strong>{data['NOMBRE_EMPRESA']} ({data['TICKER']})</strong> cotiza actualmente a <strong>{formatear_numero(data['PRECIO_ACTUAL'], 2)}€</strong>. Mi precio objetivo de compra se sitúa en <strong>{formatear_numero(data['PRECIO_OBJETIVO_COMPRA'], 2)}€</strong>. El volumen negociado recientemente, alcanzó las <strong>{formatear_numero(data['VOLUMEN'], 0)} acciones</strong>.</p>

<p>Asignamos una <strong>nota técnica de {formatear_numero(data['NOTA_EMPRESA'], 1)} sobre 10</strong>. Esta puntuación, combinada con el análisis de la **{data['TENDENCIA_NOTA']}** de la nota, la proximidad a soportes y resistencias, y el volumen, nos lleva a una recomendación de <strong>{data['RECOMENDACION']}</strong>. Actualmente, el mercado se encuentra en una situación de <strong>{data['CONDICION_RSI']}</strong>. Esto refleja {'una excelente fortaleza técnica y baja volatilidad esperada a corto plazo, lo que indica un bajo riesgo técnico en relación con el potencial de crecimiento.' if "Compra" in data['RECOMENDACION'] and data['NOTA_EMPRESA'] >= 7 else ''}
{'una fortaleza técnica moderada, con un equilibrio entre potencial y riesgo, sugiriendo una oportunidad que requiere seguimiento.' if "Compra Moderada" in data['RECOMENDACION'] or (data['NOTA_EMPRESA'] >= 5 and data['NOTA_EMPRESA'] < 7) else ''}
{'una situación técnica neutral, donde el gráfico no muestra un patrón direccional claro, indicando que es un momento para la observación y no para la acción inmediata.' if "Neutral" in data['RECOMENDACION'] else ''}
{'cierta debilidad técnica, con posibles señales de corrección o continuación bajista, mostrando una pérdida de impulso alcista y un aumento de la presión vendedora.' if "Venta" in data['RECOMENDACION'] or (data['NOTA_EMPRESA'] < 5 and data['NOTA_EMPRESA'] >= 3) else ''}
{'una debilidad técnica significativa y una posible sobrecompra en el gráfico, lo que sugiere un alto riesgo de corrección.' if "Venta Fuerte" in data['RECOMENDACION'] or data['NOTA_EMPRESA'] < 3 else ''} </p>
{chart_html}


<h2>Estrategia de Inversión y Gestión de Riesgos</h2>
<p>Mi evaluación profesional indica que la tendencia actual de nuestra nota técnica es **{data['TENDENCIA_NOTA']}**, lo que, en combinación con el resto de nuestros indicadores, se alinea con una recomendación de <strong>{data['RECOMENDACION']}</strong>.</p>
<p><strong>Motivo de la Recomendación:</strong> {data['MOTIVO_RECOMENDACION']}</p>

<p>{volumen_analisis_text}</p>


<h2>Predicción a Largo Plazo y Conclusión</h2>
<p>Considerando la nota técnica actual de <strong>{formatear_numero(data['NOTA_EMPRESA'], 1)}</strong> y la dirección de su tendencia (<strong>{data['TENDENCIA_NOTA']}</strong>), mi pronóstico a largo plazo para <strong>{data['NOMBRE_EMPRESA']}</strong> es {("optimista. La empresa muestra una base sólida para un crecimiento sostenido, respaldada por indicadores técnicos favorables y una gestión financiera prudente. Si los planes de expansión y los acuerdos estratégicos se materializan, podríamos ver una apreciación significativa del valor en el futuro." if data['NOTA_EMPRESA'] >= 7 else "")}
{("cauteloso. Si bien no hay señales inmediatas de alarma, la nota técnica sugiere que la empresa podría enfrentar desafíos en el corto y mediano plazo. Es crucial monitorear de cerca los riesgos identificados y cualquier cambio en el sentimiento del mercado para ajustar la estrategia." if data['NOTA_EMPRESA'] < 7 and data['NOTA_EMPRESA'] >=4 else "")}
{("pesimista. La debilidad técnica persistente y los factores de riesgo sugieren que la empresa podría experimentar una presión bajista considerable. Se recomienda extrema cautela y considerar estrategias de protección de capital." if data['NOTA_EMPRESA'] < 4 else "")}.</p>

<h2>Conclusión General y Descargo de Responsabilidad</h2>
<p>Para cerrar este análisis de <strong>{data['NOMBRE_EMPRESA']}</strong>, considero que las claras señales técnicas que apuntan a {('un rebote desde una zona de sobreventa extrema, configurando una oportunidad atractiva' if data['NOTA_EMPRESA'] >= 7 else 'una posible corrección, lo que exige cautela')}, junto con aspectos fundamentales que requieren mayor claridad, hacen de esta empresa un activo para mantener bajo estricta vigilancia. </p>
<p>Descargo de responsabilidad: Este contenido tiene una finalidad exclusivamente informativa y educativa. No constituye ni debe interpretarse como una recomendación de inversión, asesoramiento financiero o una invitación a comprar o vender ningún activo. </p>

"""
    return prompt_formateado, titulo_post


# Función para generar contenido con Gemini (no modificada, excepto por la llamada a construir_prompt_formateado)
def generar_contenido_con_gemini(tickers):
    model = configurar_gemini()
    for ticker in tickers:
        print(f"\n📊 Procesando ticker: {ticker}")
        data = obtener_datos_yfinance(ticker)

        if data is None:
            print(f"❌ No se pudieron obtener datos para {ticker}. Saltando al siguiente ticker.")
            continue

        prompt, titulo_post = construir_prompt_formateado(data) # Ahora devuelve el prompt y el título

        # Configuración de reintentos para la llamada a la API de Gemini
        max_retries = 3
        initial_delay = 5  # segundos
        retries = 0

        while retries < max_retries:
            try:
                print(f"✉️ Enviando prompt a Gemini para {ticker} (Intento {retries + 1}/{max_retries})...")
                # print("DEBUG - Prompt enviado:\n", prompt) # Descomentar para depurar el prompt completo

                response = model.generate_content(prompt)
                
                # Acceder al texto de la respuesta
                contenido_generado = response.text
                print(f"✅ Contenido generado con éxito para {ticker}.")

                # Nombre del archivo para guardar el post
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"post_{ticker}_{timestamp}.html"
                
                # Guarda el contenido en un archivo HTML en la carpeta 'posts'
                posts_dir = "posts"
                os.makedirs(posts_dir, exist_ok=True)
                file_path = os.path.join(posts_dir, nombre_archivo)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(contenido_generado)
                print(f"💾 Post guardado en: {file_path}")

                # Enviar email
                to_email = os.getenv('RECIPIENT_EMAIL')
                if to_email:
                    enviar_email(f"Post de Análisis Técnico: {titulo_post}", contenido_generado, to_email)
                else:
                    print("Advertencia: No se encontró RECIPIENT_EMAIL en las variables de entorno. No se enviará el email.")
                break # Salir del bucle si es exitoso

            except Exception as e:
                if "quota" in str(e).lower():
                    server_suggested_delay = 0
                    try:
                        # Intenta extraer el retraso sugerido por el servidor de Gemini
                        match = re.search(r"retry_delay \{\s*seconds: (\d+)", str(e))
                        if match:
                            server_suggested_delay = int(match.group(1))
                    except:
                        pass

                    current_delay = max(initial_delay * (2 ** retries), server_suggested_delay + 1)

                    jitter = random.uniform(0.5, 1.5)
                    delay_with_jitter = current_delay * jitter

                    print(f"❌ Cuota de Gemini excedida al generar contenido. Reintentando en {delay_with_jitter:.2f} segundos... (Intento {retries + 1}/{max_retries})")
                    time.sleep(delay_with_jitter)
                    retries += 1
                else:
                    print(f"❌ Error al generar contenido con Gemini (no de cuota): {e}")
                    break
        else:  
            print(f"❌ Falló la generación de contenido para {ticker} después de {max_retries} reintentos.")
            
        print(f"⏳ Esperando 180 segundos antes de procesar el siguiente ticker...")
        time.sleep(180)


def main():
    # Define el ticker que quieres analizar
    ticker_deseado = "AMS.MC"  # <-- ¡CAMBIA "AMS.MC" por el Ticker que quieras analizar!
                                # Por ejemplo: "REP.MC", "TSLA", etc.

    # Prepara la lista de tickers para la función generar_contenido_con_gemini
    # que espera una lista de tickers
    tickers_for_today = [ticker_deseado]

    if tickers_for_today:
        print(f"\nAnalizando el ticker solicitado: {ticker_deseado}")
        generar_contenido_con_gemini(tickers_for_today)
    else:
        print(f"No se especificó ningún ticker deseado. Saliendo.")

if __name__ == "__main__":
    main()
