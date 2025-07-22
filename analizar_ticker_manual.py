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

def calculate_smi_tv(df, window=20, smooth_window=5):
    """
    Calcula el Stochastic Momentum Index (SMI) y su señal para un DataFrame.
    Añade un campo TV (True Value) para normalizar el volumen.
    """
    if 'High' not in df.columns or 'Low' not in df.columns or 'Close' not in df.columns or 'Open' not in df.columns or 'Volume' not in df.columns:
        # print("Advertencia: Columnas necesarias (High, Low, Close, Open, Volume) no encontradas en el DataFrame. Saltando cálculo de SMI.")
        df['SMI'] = np.nan
        df['SMI_signal'] = np.nan
        df['TV'] = np.nan
        return df

    # Calcular SMI al estilo TradingView
    hh = df['High'].rolling(window=window).max()
    ll = df['Low'].rolling(window=window).min()

    diff = hh - ll
    rdiff = df['Close'] - (hh + ll) / 2

    avgrel = rdiff.ewm(span=3, adjust=False).mean()
    avgdiff = diff.ewm(span=3, adjust=False).mean()

    smi = np.where(avgdiff != 0, (avgrel / (avgdiff / 2)) * 100, 0)
    smi_smoothed = pd.Series(smi, index=df.index).rolling(window=smooth_window).mean()
    smi_signal = smi_smoothed.ewm(span=10, adjust=False).mean()

    df['SMI'] = smi_smoothed
    df['SMI_signal'] = smi_signal

    # Calcular True Value (TV) para el volumen
    df['TR'] = np.maximum(df['High'] - df['Low'],
                          np.maximum(abs(df['High'] - df['Close'].shift()),
                                     abs(df['Low'] - df['Close'].shift())))
    df['ATR'] = df['TR'].rolling(window=window).mean()
    df['TV'] = df['Volume'] / df['ATR'] # Normalización simple de volumen
    df['TV'] = df['TV'].replace([np.inf, -np.inf], np.nan).fillna(0) # Manejar infinitos y NaN

    return df

def obtener_datos_yfinance(ticker):
    print(f"Buscando datos para el ticker: {ticker}")
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="60d") # Suficientes datos para SMI y un historial de 30 días + proyección
        hist_extended = data.history(period="1y") # Más datos para cálculos si es necesario

        if hist.empty or hist_extended.empty:
            print(f"⚠️ Advertencia: No se encontraron datos históricos para {ticker}.")
            return None

        # --- Cálculo del SMI (basado en TradingView) ---
        def calculate_smi_tv(df, length=20, ema_length=5, signal_length=5):
            # Calcular el rango más bajo y más alto de 'length' períodos
            low_min = df['Low'].rolling(window=length).min()
            high_max = df['High'].rolling(window=length).max()

            # Calcular el cambio en el cierre y el rango total
            range_diff = df['Close'] - ((high_max + low_min) / 2)
            total_range = high_max - low_min

            # Evitar división por cero
            smi = pd.Series(np.where(total_range == 0, 0, range_diff / (total_range / 2)))

            # Primera EMA (smi_raw)
            df['SMI_raw'] = smi.ewm(span=ema_length, adjust=False).mean()

            # Segunda EMA (smi_smoothed, que es el SMI principal)
            df['SMI'] = df['SMI_raw'].ewm(span=ema_length, adjust=False).mean()

            # EMA de la señal (signal_line)
            df['SMI_signal'] = df['SMI'].ewm(span=signal_length, adjust=False).mean()
            return df

        hist_extended = calculate_smi_tv(hist_extended)
        hist = calculate_smi_tv(hist) # Aplicar a los datos de 60 días también para los valores actuales

        if 'Close' not in hist.columns or hist['Close'].empty:
            print(f"⚠️ Advertencia: No hay datos de cierre disponibles para {ticker}.")
            return None

        current_price = hist['Close'].iloc[-1]
        volume = hist['Volume'].iloc[-1]
        average_volume = hist['Volume'].tail(30).mean()

        # Obtener valores actuales de SMI
        smi_actual_series = hist['SMI'].dropna() # smoothed SMI (main one)
        smi_raw_actual_series = hist['SMI_raw'].dropna() # raw SMI

        smi_actual = 0
        if not smi_actual_series.empty:
            smi_actual = round(smi_actual_series.iloc[-1], 2)
        else:
            print(f"⚠️ Advertencia: No hay datos de SMI suavizado válidos para {ticker}. Asignando SMI neutral.")

        smi_raw_actual = 0
        if not smi_raw_actual_series.empty:
            smi_raw_actual = round(smi_raw_actual_series.iloc[-1], 2)
        else:
            smi_raw_actual = smi_actual # Fallback to smoothed if raw not available
            print(f"⚠️ Advertencia: No hay datos de SMI RAW válidos para {ticker}. Usando SMI suavizado como fallback.")


        # Determinar tendencia del SMI
        tendencia_smi = "Neutral"
        if len(smi_actual_series) >= 5: # Necesitamos al menos 5 puntos para una pequeña tendencia
            smi_last_5 = smi_actual_series.tail(5)
            if smi_last_5.iloc[-1] > smi_last_5.iloc[0]:
                tendencia_smi = "Ascendente"
            elif smi_last_5.iloc[-1] < smi_last_5.iloc[0]:
                tendencia_smi = "Descendente"

        # Calcular RSI (ejemplo básico, ajustar según necesidad)
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_actual = rsi.iloc[-1] if not rsi.empty else 50

        condicion_rsi = "Indeterminada"
        if rsi_actual >= 70:
            condicion_rsi = "Sobrecompra"
        elif rsi_actual <= 30:
            condicion_rsi = "Sobreventa"
        else:
            condicion_rsi = "Neutral"

        # Precios históricos para el gráfico
        cierres_para_grafico_full = hist['Close'].dropna().tolist()

        smi_history_full = hist['SMI'].dropna()
        smi_raw_history_full = hist['SMI_raw'].dropna()


        # Datos para los últimos 30 días para el gráfico
        # Asumiendo que quieres 30 puntos para el historial en el gráfico
        cierres_para_grafico_total = []
        if len(cierres_para_grafico_full) >= 30:
            cierres_para_grafico_total = cierres_para_grafico_full[-30:]
        elif cierres_para_grafico_full: # Si hay menos de 30 pero hay datos, los usamos
            cierres_para_grafico_total = cierres_para_grafico_full
        else:
            cierres_para_grafico_total = [0.0] * 30 # Rellenar con ceros si no hay datos

        smi_historico_para_grafico = []
        if len(smi_history_full) >= 30:
            smi_historico_para_grafico = smi_history_full.tail(30).tolist()
        elif smi_history_full.empty:
            smi_historico_para_grafico = [0.0] * 30 # Asegurarse de que no esté vacío
        else: # Si hay menos de 30, rellenar el principio con el primer valor para mantener la longitud de 30
            first_smi_val = smi_history_full.iloc[0]
            smi_historico_para_grafico = [first_smi_val] * (30 - len(smi_history_full)) + smi_history_full.tolist()

        smi_raw_historico_para_grafico = []
        if len(smi_raw_history_full) >= 30:
            smi_raw_historico_para_grafico = smi_raw_history_full.tail(30).tolist()
        elif smi_raw_history_full.empty:
            smi_raw_historico_para_grafico = [0.0] * 30 # Asegurarse de que no esté vacío
        else: # Si hay menos de 30, rellenar el principio con el primer valor para mantener la longitud de 30
            first_smi_raw_val = smi_raw_history_full.iloc[0]
            smi_raw_historico_para_grafico = [first_smi_raw_val] * (30 - len(smi_raw_history_full)) + smi_raw_history_full.tolist()


        # Definir niveles de soporte y resistencia de forma simplificada
        # Esto es un ejemplo, puedes necesitar una lógica más sofisticada
        if len(hist['Close']) > 10:
            soporte_1 = round(hist['Low'].tail(10).min() * 0.98, 2)
            soporte_2 = round(hist['Low'].tail(20).min() * 0.95, 2)
            soporte_3 = round(hist['Low'].min() * 0.92, 2)
            resistencia_1 = round(hist['High'].tail(10).max() * 1.02, 2)
            resistencia_2 = round(hist['High'].tail(20).max() * 1.05, 2)
            resistencia_3 = round(hist['High'].max() * 1.08, 2)
        else: # Valores por defecto si no hay suficientes datos
            soporte_1, soporte_2, soporte_3 = current_price * 0.95, current_price * 0.90, current_price * 0.85
            resistencia_1, resistencia_2, resistencia_3 = current_price * 1.05, current_price * 1.10, current_price * 1.15

        # Nota de la empresa (simplificado para el ejemplo)
        # Esto debería venir de otra fuente o cálculo
        nota_empresa = round((smi_actual + rsi_actual) / 20, 1) # Ejemplo simplificado

        # Tendencia de la nota (ejemplo simplificado)
        tendencia_nota = "Estable"
        if smi_actual > 60 and rsi_actual > 60:
            tendencia_nota = "Ascendente"
        elif smi_actual < -60 and rsi_actual < 40:
            tendencia_nota = "Descendente"

        # Recomendación (simplificado)
        recomendacion = "Neutral"
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

        # Precio objetivo (simplificado)
        precio_objetivo = round(current_price * 1.05, 2) if recomendacion == "Compra" else round(current_price * 0.95, 2)
        precio_objetivo_compra = round(current_price * 0.98, 2) if recomendacion == "Compra" else "N/A"

        # Preparar datos para el prompt
        datos = {
            "NOMBRE_EMPRESA": data.info.get('longName', ticker),
            "TICKER": ticker,
            "PRECIO_ACTUAL": current_price,
            "VOLUMEN": volume,
            "VOLUMEN_MEDIO": average_volume,
            "SMI": smi_actual,
            "SMI_RAW": smi_raw_actual, # Asegúrate de que este dato se pasa
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
            "SMI_RAW_HISTORICO_PARA_GRAFICO": smi_raw_historico_para_grafico, # Asegúrate de que este dato se pasa
            "PROYECCION_FUTURA_DIAS": 5 # Número de días para proyectar en el gráfico
        }
        return datos

    except Exception as e:
        print(f"❌ Error al obtener datos para {ticker}: {e}")
        return None

def generar_recomendacion_avanzada(data, cierres_para_grafico_total, smi_historico_para_grafico): # Cambio de nombre de la variable
    # Extraer los últimos 30 días de SMI para el análisis de tendencias
    smi_historico = smi_historico_para_grafico[-30:] if len(smi_historico_para_grafico) >= 30 else smi_historico_para_grafico

    # Calcular la pendiente de los últimos N SMI para la tendencia
    n_trend = min(7, len(smi_historico)) # Últimos 7 días o menos si no hay tantos
    if n_trend > 1:
        x_trend = np.arange(n_trend)
        y_trend = np.array(smi_historico[-n_trend:])
        # Filtrar NaN para calcular la pendiente
        valid_indices = ~np.isnan(y_trend)
        if np.any(valid_indices): # Solo calcular si hay datos válidos
            slope, _ = np.polyfit(x_trend[valid_indices], y_trend[valid_indices], 1)
        else:
            slope = 0 # No hay datos válidos para la pendiente
    else:
        slope = 0 # No hay suficientes datos para calcular la pendiente

    if slope > 0.1:
        tendencia_smi = "mejorando (alcista)"
    elif slope < -0.1:
        tendencia_smi = "empeorando (bajista)"
    else:
        tendencia_smi = "estable (lateral)"

    # Determinar si el volumen es alto (ej. > 1.5 veces el volumen medio de los últimos 20 días)
    volumen_alto = False
    if data['VOLUMEN_MEDIO'] and data['VOLUMEN'] is not None and data['VOLUMEN_MEDIO'] > 0:
        if data['VOLUMEN'] > (data['VOLUMEN_MEDIO'] * 1.5):
            volumen_alto = True

    # Determinar proximidad a soportes y resistencias
    proximidad_soporte = False
    proximidad_resistencia = False
    if data['PRECIO_ACTUAL'] is not None:
        if data['SOPORTE_1'] is not None and data['PRECIO_ACTUAL'] != 0:
            if abs(data['PRECIO_ACTUAL'] - data['SOPORTE_1']) / data['PRECIO_ACTUAL'] < 0.02: # 2% de proximidad
                proximidad_soporte = True
        if data['RESISTENCIA_1'] is not None and data['PRECIO_ACTUAL'] != 0:
            if abs(data['PRECIO_ACTUAL'] - data['RESISTENCIA_1']) / data['PRECIO_ACTUAL'] < 0.02: # 2% de proximidad
                proximidad_resistencia = True

    recomendacion = "Neutral"
    condicion_mercado = "En observación"
    motivo_recomendacion = "La situación actual no presenta señales claras de compra ni venta."

    # Lógica de Giro Alcista (Compra)
    # Basamos la recomendación en el SMI directamente
    if tendencia_smi == "mejorando (alcista)" and data['SMI'] < 0 and volumen_alto: # SMI en zona negativa y mejorando con volumen
        recomendacion = "Fuerte Compra"
        condicion_mercado = "Impulso alcista con confirmación de volumen desde zona de sobreventa"
        motivo_recomendacion = "El SMI está mejorando y se encuentra en zona de sobreventa, con un volumen significativo, indicando un fuerte impulso alcista."

    # Lógica de Giro Bajista (Venta Condicional)
    elif tendencia_smi == "empeorando (bajista)" and data['SMI'] > 0 and volumen_alto: # SMI en zona positiva y empeorando con volumen
        if not proximidad_soporte:
            recomendacion = "Venta Condicional / Alerta"
            condicion_mercado = "Debilidad confirmada por volumen desde zona de sobrecompra, considerar salida"
            motivo_recomendacion = "El SMI está empeorando y se encuentra en zona de sobrecompra, con volumen alto y sin soporte cercano, sugiriendo debilidad."
        else:
            recomendacion = "Neutral / Cautela"
            condicion_mercado = "Debilidad pero cerca de soporte clave, observar rebote"
            motivo_recomendacion = "El SMI está empeorando, pero la proximidad a un soporte clave sugiere cautela antes de vender."

    # Detección de Patrones de Reversión desde Extremos:
    # Reversión de Compra (SMI saliendo de sobrecompra/extremo negativo)
    if len(smi_historico) >= 2 and smi_historico[-1] > smi_historico[-2] and \
       smi_historico[-2] <= -40 and smi_historico[-1] > -40: # SMI estaba muy bajo y empieza a subir
        if recomendacion not in ["Fuerte Compra", "Oportunidad de Compra (Reversión)"]:
            recomendacion = "Oportunidad de Compra (Reversión)"
            condicion_mercado = "Posible inicio de rebote tras sobreventa extrema, punto de entrada"
            motivo_recomendacion = "Reversión de compra: El SMI está ascendiendo desde una zona de sobreventa extrema, indicando una oportunidad de entrada."

    # Reversión de Venta (SMI saliendo de sobreventa/extremo positivo)
    elif len(smi_historico) >= 2 and smi_historico[-1] < smi_historico[-2] and \
         smi_historico[-2] >= 40 and smi_historico[-1] < 40: # SMI estaba muy alto y empieza a bajar
        if recomendacion not in ["Venta Condicional / Alerta", "Señal de Venta (Reversión)"]:
            recomendacion = "Señal de Venta (Reversión)"
            condicion_mercado = "Posible inicio de corrección tras sobrecompra extrema, punto de salida"
            motivo_recomendacion = "Señal de venta: El SMI está descendiendo desde una zona de sobrecompra extrema, indicando un punto de salida."

    # Lógica para "Neutral" si ninguna de las condiciones anteriores se cumple con fuerza
    if recomendacion == "Neutral":
        if tendencia_smi == "estable (lateral)":
            condicion_mercado = "Consolidación o lateralidad sin dirección clara."
            motivo_recomendacion = "El SMI se mantiene estable, indicando una fase de consolidación o lateralidad sin dirección clara."
        elif data['SMI'] < 20 and tendencia_smi == "mejorando (alcista)" and not volumen_alto:
            recomendacion = "Neutral / Observación"
            condicion_mercado = "SMI moderadamente bajo con mejora, pero falta confirmación de volumen."
            motivo_recomendacion = "El SMI es moderadamente bajo y muestra una mejora, pero la falta de volumen significativo sugiere una fase de observación."
        elif data['SMI'] > -20 and tendencia_smi == "empeorando (bajista)" and not volumen_alto:
            recomendacion = "Neutral / Observación"
            condicion_mercado = "SMI moderadamente alto con empeoramiento, pero falta confirmación de volumen."
            motivo_recomendacion = "El SMI es moderadamente alto y empeora, pero la falta de volumen significativo sugiere una fase de observación."

    data['RECOMENDACION'] = recomendacion
    data['CONDICION_RSI'] = condicion_mercado # Aunque el nombre es RSI, el concepto es la condición del mercado
    data['MOTIVO_RECOMENDACION'] = motivo_recomendacion

    return data


def construir_prompt_formateado(data):
    def formatear_numero(numero):
        if numero is None:
            return "N/A"
        return f"{numero:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")

    ticker = data.get('TICKER', 'N/A')
    titulo_post = f"Análisis Técnico Avanzado: ¡Oportunidad en {data['NOMBRE_EMPRESA']} ({ticker})!"

    volumen_analisis_text = ""
    if data['VOLUMEN'] > data['VOLUMEN_MEDIO'] * 1.5:
        volumen_analisis_text = f"El volumen de {formatear_numero(data['VOLUMEN'])} acciones, un {((data['VOLUMEN'] / data['VOLUMEN_MEDIO']) - 1) * 100:.2f}% superior al promedio, indica un fuerte interés. Esto refuerza mi recomendación de {data['RECOMENDACION']}."
    elif data['VOLUMEN'] < data['VOLUMEN_MEDIO'] * 0.5:
        volumen_analisis_text = f"El volumen de {formatear_numero(data['VOLUMEN'])} acciones, un {((data['VOLUMEN_MEDIO'] / data['VOLUMEN']) - 1) * 100:.2f}% inferior al promedio, sugiere cautela. Esto debilita mi recomendación de {data['RECOMENDACION']}."
    else:
        volumen_analisis_text = f"El volumen de {formatear_numero(data['VOLUMEN'])} acciones, en línea con el promedio, indica un interés normal. Esto apoya mi recomendación de {data['RECOMENDACION']}."

    chart_html = "" # Inicializar chart_html
    cierres_para_grafico_total = data.get('CIERRES_PARA_GRAFICO_TOTAL', [])
    smi_historico_para_grafico = data.get('SMI_HISTORICO_PARA_GRAFICO', [])
    smi_raw_historico_para_grafico = data.get('SMI_RAW_HISTORICO_PARA_GRAFICO', [])
    PROYECCION_FUTURA_DIAS = data.get('PROYECCION_FUTURA_DIAS', 5)
    OFFSET_DIAS = 4 # Este valor se usa para alinear el SMI en el gráfico


    # Solo generar el gráfico si tenemos datos válidos para todos los elementos clave
    if smi_historico_para_grafico and cierres_para_grafico_total and smi_raw_historico_para_grafico:
        # Asegurarse de que las listas de SMI tengan al menos 30 elementos para Chart.js
        # Si tienes menos de 30 días, rellena con el primer valor para que el gráfico no falle
        if len(smi_historico_para_grafico) < 30:
            padding_needed = 30 - len(smi_historico_para_grafico)
            smi_historico_para_grafico = [smi_historico_para_grafico[0]] * padding_needed + smi_historico_para_grafico if smi_historico_para_grafico else [0.0] * 30
        
        if len(smi_raw_historico_para_grafico) < 30:
            padding_needed = 30 - len(smi_raw_historico_para_grafico)
            smi_raw_historico_para_grafico = [smi_raw_historico_para_grafico[0]] * padding_needed + smi_raw_historico_para_grafico if smi_raw_historico_para_grafico else [0.0] * 30

        if len(cierres_para_grafico_total) < 30:
            padding_needed = 30 - len(cierres_para_grafico_total)
            cierres_para_grafico_total = [cierres_para_grafico_total[0]] * padding_needed + cierres_para_grafico_total if cierres_para_grafico_total else [0.0] * 30


        labels_historial = [(datetime.today() - timedelta(days=29 - i)).strftime("%d/%m") for i in range(30)]
        labels_proyeccion = [(datetime.today() + timedelta(days=i)).strftime("%d/%m (fut.)") for i in range(1, PROYECCION_FUTURA_DIAS + 1)]
        labels_total = labels_historial + labels_proyeccion

        precios_reales_grafico = cierres_para_grafico_total[:30]
        data_proyectada = [None] * (len(labels_historial) - 1) + [precios_reales_grafico[-1]] + ([None] * PROYECCION_FUTURA_DIAS) # Proyección inicialmente nula


        # Desplazar el SMI para el gráfico
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


    # --- INICIO DEL PROMPT FINAL PARA GEMINI ---
    # Asegúrate de que este bloque de 'return' esté indentado al mismo nivel que las definiciones de variables principales
    # dentro de la función (ej. 'volumen_analisis_text = ""')
    prompt_formateado = f"""
Actúa como un trader profesional con amplia experiencia en análisis técnico y mercados financieros. Genera el análisis completo en **formato HTML**, ideal para publicaciones web. Utiliza etiquetas `<h2>` para los títulos de sección y `<p>` para cada párrafo de texto. Redacta en primera persona, con total confianza en tu criterio.

Destaca los datos importantes como precios, notas de la empresa, cifras financieras y el nombre de la empresa utilizando la etiqueta `<strong>`. Asegúrate de que no haya asteriscos u otros símbolos de marcado en el texto final, solo HTML válido. Asegurate que todo este escrito en español independientemente del idioma de donde saques los datos.

Genera un análisis técnico completo de aproximadamente 800 palabras sobre la empresa {data['NOMBRE_EMPRESA']}, utilizando los siguientes datos reales extraídos de Yahoo Finance. Presta especial atención a la **nota obtenida por la empresa**: {data['NOTA_EMPRESA']}.

¡ATENCIÓN URGENTE! Para CADA EMPRESA analizada, debes generar el CÓDIGO HTML Y JAVASCRIPT COMPLETO y Único para TODOS sus gráficos solicitados (Notas Chart, Divergencia Color Chart, Nota Variación Chart y Precios Chart). Bajo ninguna circunstancia debes omitir ningún script, resumir bloques de código o utilizar frases como 'código JavaScript idéntico al ejemplo anterior'. Cada gráfico, para cada empresa, debe tener su script completamente incrustado, funcional e independiente de otros. Asegúrate de que los datos de cada gráfico corresponden SIEMPRE a la empresa que se está analizando en ese momento

**Datos clave:**
- Precio actual: {data['PRECIO_ACTUAL']}
- Volumen del último día completo: {data['VOLUMEN']}
- Soporte 1: {data['SOPORTE_1']}
- Soporte 2: {data['SOPORTE_2']}
- Soporte 3: {data['SOPORTE_3']}
- Resistencia clave: {data['RESISTENCIA_1']}
- Recomendación general: {data['RECOMENDACION']}
- Nota de la empresa (0-10): {data['NOTA_EMPRESA']}
- Precio objetivo de compra: {data['PRECIO_OBJETIVO_COMPRA']}€
- Tendencia de la nota: {data['TENDENCIA_NOTA']}


Importante: si algún dato no está disponible ("N/A", "No disponibles", "No disponible"), no lo menciones ni digas que falta. No expliques que la recomendación proviene de un indicador o dato específico. La recomendación debe presentarse como una conclusión personal basada en tu experiencia y criterio profesional como analista. Al redactar el análisis, haz referencia a la **nota obtenida por la empresa ({data['NOTA_EMPRESA']})** en al menos dos de los párrafos principales (Recomendación General, Análisis a Corto Plazo o Predicción a Largo Plazo) como un factor clave para tu valoración.

---
<h1>{titulo_post}</h1>


<h2>Análisis Inicial y Recomendación</h2>
<p><strong>{data['NOMBRE_EMPRESA']} ({data['TICKER']})</strong> cotiza actualmente a <strong>{data['PRECIO_ACTUAL']:.2f}€</strong>. Mi precio objetivo de compra se sitúa en <strong>{data['PRECIO_OBJETIVO_COMPRA']:.2f}€</strong>. El volumen negociado recientemente, alcanzó las <strong>{formatear_numero(data['VOLUMEN'])} acciones</strong>.</p>

<p>Asignamos una <strong>nota técnica de {data['NOTA_EMPRESA']} sobre 10</strong>. Esta puntuación, combinada con el análisis de la **{data['TENDENCIA_NOTA']}** de la nota, la proximidad a soportes y resistencias, y el volumen, nos lleva a una recomendación de <strong>{data['RECOMENDACION']}</strong>. Actualmente, el mercado se encuentra en una situación de <strong>{data['CONDICION_RSI']}</strong>. Esto refleja {'una excelente fortaleza técnica y baja volatilidad esperada a corto plazo, lo que indica un bajo riesgo técnico en relación con el potencial de crecimiento.' if "Compra" in data['RECOMENDACION'] and data['NOTA_EMPRESA'] >= 7 else ''}
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
<p>Considerando la nota técnica actual de <strong>{data['NOTA_EMPRESA']}</strong> y la dirección de su tendencia (<strong>{data['TENDENCIA_NOTA']}</strong>), mi pronóstico a largo plazo para <strong>{data['NOMBRE_EMPRESA']}</strong> es {("optimista. La empresa muestra una base sólida para un crecimiento sostenido, respaldada por indicadores técnicos favorables y una gestión financiera prudente. Si los planes de expansión y los acuerdos estratégicos se materializan, podríamos ver una apreciación significativa del valor en el futuro." if data['NOTA_EMPRESA'] >= 7 else "")}
{("cauteloso. Si bien no hay señales inmediatas de alarma, la nota técnica sugiere que la empresa podría enfrentar desafíos en el corto y mediano plazo. Es crucial monitorear de cerca los riesgos identificados y cualquier cambio en el sentimiento del mercado para ajustar la estrategia." if data['NOTA_EMPRESA'] < 7 and data['NOTA_EMPRESA'] >=4 else "")}
{("pesimista. La debilidad técnica persistente y los factores de riesgo sugieren que la empresa podría experimentar una presión bajista considerable. Se recomienda extrema cautela y considerar estrategias de protección de capital." if data['NOTA_EMPRESA'] < 4 else "")}.</p>

<h2>Conclusión General y Descargo de Responsabilidad</h2>
<p>Para cerrar este análisis de <strong>{data['NOMBRE_EMPRESA']}</strong>, considero que las claras señales técnicas que apuntan a {('un rebote desde una zona de sobreventa extrema, configurando una oportunidad atractiva' if data['NOTA_EMPRESA'] >= 7 else 'una posible corrección, lo que exige cautela')}, junto con aspectos fundamentales que requieren mayor claridad, hacen de esta empresa un activo para mantener bajo estricta vigilancia. </p>
<p>Descargo de responsabilidad: Este contenido tiene una finalidad exclusivamente informativa y educativa. No constituye ni debe interpretarse como una recomendación de inversión, asesoramiento financiero o una invitación a comprar o vender ningún activo. </p>

"""
    return prompt_formateado, titulo_post


def enviar_email(texto_generado, asunto_email, nombre_archivo):
    import os
    from email.mime.base import MIMEBase
    from email import encoders

    remitente = "xumkox@gmail.com"
    destinatario = "xumkox@gmail.com"
    password = "kdgz lvdo wqvt vfkt"  # RECOMENDADO: usar variable de entorno

    # Guardar el HTML en un archivo temporal
    ruta_archivo = f"{nombre_archivo}.html"
    with open(ruta_archivo, "w", encoding="utf-8") as f:
        f.write(texto_generado)

    # Crear el email
    msg = MIMEMultipart()
    msg['From'] = remitente
    msg['To'] = destinatario
    msg['Subject'] = asunto_email
    msg.attach(MIMEText("Adjunto el análisis en formato HTML.", 'plain'))

    # Adjuntar el archivo HTML
    with open(ruta_archivo, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename= {nombre_archivo}.html")
    msg.attach(part)

    # Enviar el correo
    try:
        servidor = smtplib.SMTP('smtp.gmail.com', 587)
        servidor.starttls()
        servidor.login(remitente, password)
        servidor.sendmail(remitente, destinatario, msg.as_string())
        servidor.quit()
        print(f"✅ Correo enviado con el adjunto: {ruta_archivo}")
    except Exception as e:
        print("❌ Error al enviar el correo:", e)

def generar_contenido_con_gemini(tickers):
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise Exception("No se encontró la variable de entorno GEMINI_API_KEY")

    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")  

    for ticker in tickers:
        print(f"\n📊 Procesando ticker: {ticker}")
        data = obtener_datos_yfinance(ticker)
        if not data:
            print(f"⏩ Saltando {ticker} debido a un error al obtener datos.")
            continue
        
        # ACCESO A LAS VARIABLES DESDE EL DICCIONARIO 'data'
        # ANTES ERAN INDEFINIDAS, AHORA SE OBTIENEN DE 'data'
        cierres_para_grafico_total = data.get('CIERRES_PARA_GRAFICO_TOTAL', [])
        # Cambio aquí para usar 'SMI_HISTORICO_PARA_GRAFICO'
        smi_historico_para_grafico = data.get('SMI_HISTORICO_PARA_GRAFICO', [])

        # Ahora pasa estas variables a la función generar_recomendacion_avanzada
        data = generar_recomendacion_avanzada(data, cierres_para_grafico_total, smi_historico_para_grafico)
        

        prompt, titulo_post = construir_prompt_formateado(data)

        max_retries = 1
        initial_delay = 10  
        retries = 0
        delay = initial_delay

        while retries < max_retries:
            try:
                response = model.generate_content(prompt)
                print(f"\n🧠 Contenido generado para {ticker}:\n")
                print(response.text)
                asunto_email = f"Análisis: {data['NOMBRE_EMPRESA']} ({data['TICKER']}) - {data['RECOMENDACION']}"
                nombre_archivo = f"analisis_{ticker}_{datetime.today().strftime('%Y%m%d')}"
                enviar_email(response.text, asunto_email, nombre_archivo)

                break  
            except Exception as e:
                if "429 You exceeded your current quota" in str(e):
                    server_suggested_delay = 0 
                    try:
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
    ticker_deseado = "AMS.MC"

    tickers_for_today = [ticker_deseado]

    if tickers_for_today:
        print(f"\nAnalizando el ticker solicitado: {ticker_deseado}")
        generar_contenido_con_gemini(tickers_for_today)
    else:
        print(f"No se especificó ningún ticker para analizar.")

if __name__ == '__main__':
    main()
