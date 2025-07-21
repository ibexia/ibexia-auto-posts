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

    # Calcular SMI
    highest_high = df['High'].rolling(window=window).max()
    lowest_low = df['Low'].rolling(window=window).min()

    range_hl = highest_high - lowest_low
    # Evitar división por cero
    range_hl = range_hl.replace(0, np.nan)

    relative_close = df['Close'] - lowest_low

    smi = ((relative_close - (range_hl / 2)) / (range_hl / 2)) * 100
    smi = smi.fillna(0) # Rellenar NaN que puedan quedar por división por cero o datos insuficientes

    df['SMI'] = smi
    df['SMI_signal'] = df['SMI'].ewm(span=smooth_window, adjust=False).mean()

    # Calcular True Value (TV) para el volumen
    df['TR'] = np.maximum(df['High'] - df['Low'],
                          np.maximum(abs(df['High'] - df['Close'].shift()),
                                     abs(df['Low'] - df['Close'].shift())))
    df['ATR'] = df['TR'].rolling(window=window).mean()
    df['TV'] = df['Volume'] / df['ATR'] # Normalización simple de volumen
    df['TV'] = df['TV'].replace([np.inf, -np.inf], np.nan).fillna(0) # Manejar infinitos y NaN

    return df

def obtener_datos_yfinance(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Ampliar periodo si es necesario para el retraso y proyecciones
        hist_extended = stock.history(period="90d", interval="1d")
        hist_extended = calculate_smi_tv(hist_extended)

        # Usar un historial más corto para obtener la tendencia de la nota actual (últimos 30 días)
        hist = stock.history(period="30d", interval="1d")
        hist = calculate_smi_tv(hist)

        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Ampliar periodo si es necesario para el retraso y proyecciones
        hist_extended = stock.history(period="90d", interval="1d")
        hist_extended = calculate_smi_tv(hist_extended)

        # Usar un historial más corto para obtener la tendencia de la nota actual (últimos 30 días)
        hist = stock.history(period="30d", interval="1d")
        hist = calculate_smi_tv(hist)

        # Obtener el precio actual y volumen
        current_price = round(info["currentPrice"], 2)
        current_volume = info.get("volume", "N/A")

        # Get last valid SMI signal and calculate nota_empresa safely
        smi_actual_series = hist['SMI_signal'].dropna() # Obtener las señales SMI sin NaN

        if not smi_actual_series.empty:
            smi_actual = round(smi_actual_series.iloc[-1], 2)
            # La nota técnica actual de la empresa
            nota_empresa = round((-(max(min(smi_actual, 60), -60)) + 60) * 10 / 120, 1)
        else:
            # Si no hay datos SMI válidos, asignar un valor por defecto
            print(f"⚠️ Advertencia: No hay datos de SMI válidos para calcular la nota de {ticker}. Asignando nota neutral.")
            smi_actual = 0  # Un valor por defecto para smi_actual
            nota_empresa = 5.0 # Nota neutral por defecto (entre 0 y 10)


        # Calcular soportes y resistencia
        # Asegurarse de tener al menos 30 días para un cálculo significativo
        if len(hist) < 30:
            highs_lows = hist[['High', 'Low', 'Close']].values.flatten()
        else:
            highs_lows = hist[['High', 'Low', 'Close']].iloc[-30:].values.flatten()
        

        # Calculamos soportes y resistencias como listas ordenadas
        # Soportes: de menor a mayor
        soportes_raw = np.unique(highs_lows)
        soportes = np.sort(soportes_raw).tolist()

        # Resistencias: de mayor a menor
        resistencias_raw = np.unique(highs_lows)
        resistencias = np.sort(resistencias_raw)[::-1].tolist() # Orden inverso para tener las más altas primero


        # Definir los 3 soportes
        if len(soportes) >= 3:
            soporte_1 = round(soportes[0], 2)
            soporte_2 = round(soportes[1], 2)
            soporte_3 = round(soportes[2], 2)
        elif len(soportes) == 2:
            soporte_1 = round(soportes[0], 2)
            soporte_2 = round(soportes[1], 2)
            soporte_3 = soporte_2 # Usar el mismo si no hay 3 distintos
        elif len(soportes) == 1:
            soporte_1 = round(soportes[0], 2)
            soporte_2 = soporte_1
            soporte_3 = soporte_1
        else:
            soporte_1, soporte_2, soporte_3 = round(current_price * 0.95, 2), round(current_price * 0.9, 2), round(current_price * 0.85, 2) # Default si no hay datos

        # Definir las 3 resistencias (similar a soportes)
        if len(resistencias) >= 3:
            resistencia_1 = round(resistencias[0], 2)
            resistencia_2 = round(resistencias[1], 2)
            resistencia_3 = round(resistencias[2], 2)
        elif len(resistencias) == 2:
            resistencia_1 = round(resistencias[0], 2)
            resistencia_2 = round(resistencias[1], 2)
            resistencia_3 = resistencia_2
        elif len(resistencias) == 1:
            resistencia_1 = round(resistencias[0], 2)
            resistencia_2 = resistencia_1
            resistencia_3 = resistencia_1
        else:
            resistencia_1, resistencia_2, resistencia_3 = round(current_price * 1.05, 2), round(current_price * 1.1, 2), round(current_price * 1.15, 2) # Default si no hay datos

        # --- LÓGICA MEJORADA PARA EL PRECIO OBJETIVO ---
        # Primero, asegurarnos de que tenemos soportes y resistencias válidos para la interpolación
        # Si no hay 3 soportes/resistencias, usaremos los disponibles o un porcentaje del precio actual.

        # Puntos de referencia para la interpolación del precio objetivo
        # Puedes ajustar estos porcentajes o la selección de soportes/resistencias
        # para que se adapten mejor a tu estrategia.
        
        # Extremo inferior: Un soporte más bajo o un porcentaje de caída
        # Se podría usar el soporte 3, o un valor más agresivo si la nota es muy baja.
        referencia_min = soportes[2] if len(soportes) >= 3 else (soportes[1] if len(soportes) >= 2 else (soportes[0] if len(soportes) >= 1 else round(current_price * 0.80, 2)))
        
        # Extremo superior: Una resistencia más alta o un porcentaje de subida
        # Se podría usar la resistencia 3, o un valor más agresivo si la nota es muy alta.
        referencia_max = resistencias[2] if len(resistencias) >= 3 else (resistencias[1] if len(resistencias) >= 2 else (resistencias[0] if len(resistencias) >= 1 else round(current_price * 1.20, 2)))

        # Asegurarse de que referencia_min sea menor que referencia_max
        if referencia_min >= referencia_max:
            # Si los soportes/resistencias están muy juntos o son incorrectos,
            # usamos un rango por defecto alrededor del precio actual
            referencia_min = round(current_price * 0.85, 2)
            referencia_max = round(current_price * 1.15, 2)
        
        # Interpolación lineal del precio objetivo basada en la nota_empresa (escalada de 0 a 10)
        # La nota se normaliza de 0 a 1.
        normalised_nota = nota_empresa / 10.0
        
        # El precio objetivo se calcula como una interpolación entre referencia_min y referencia_max
        # Un normalised_nota de 0 daría referencia_min, y un normalised_nota de 1 daría referencia_max.
        precio_objetivo = referencia_min + (referencia_max - referencia_min) * normalised_nota
        
        # Asegurarse de que el precio objetivo no sea irracionalmente bajo si la nota es 0
        # y referencia_min es muy alto por falta de soportes.
        if nota_empresa == 0 and precio_objetivo > current_price * 0.9:
            precio_objetivo = round(min(precio_objetivo, current_price * 0.85), 2) # Limitar a un 15% de caída
        elif nota_empresa == 10 and precio_objetivo < current_price * 1.1:
            precio_objetivo = round(max(precio_objetivo, current_price * 1.15), 2) # Limitar a un 15% de subida


        precio_objetivo = round(precio_objetivo, 2)
        # --- FIN DE LA LÓGICA MEJORADA PARA EL PRECIO OBJETIVO ---

        # Precio objetivo de compra (ejemplo simple, puedes refinarlo)
        # Este 'precio_objetivo_compra' es diferente al 'precio_objetivo' general
        precio_objetivo_compra = round(current_price * 0.98, 2) # Un 2% por debajo del precio actual como ejemplo

        

        # Inicializar recomendacion y condicion_rsi como temporales, se recalcularán después
        recomendacion = "Pendiente de análisis avanzado"
        condicion_rsi = "Pendiente"


        # Nuevas variables para los gráficos con offset y proyección
        OFFSET_DIAS = 4 # La nota de hoy (D) se alinea con el precio de D+4
        PROYECCION_FUTURA_DIAS = 5 # Días a proyectar después del último precio real

        # Aseguramos tener suficientes datos para el historial, el offset y la proyección
        smi_history_full = hist_extended['SMI_signal'].dropna()
        cierres_history_full = hist_extended['Close'].dropna()

        # Calcula el volumen promedio de los últimos 30 días usando hist_extended
        volumen_promedio_30d = hist_extended['Volume'].tail(30).mean()

        # Notas para los 30 días del gráfico (serán las que se visualicen)
        # Serán las 30 notas más recientes disponibles
        notas_historicas_para_grafico = []
        if len(smi_history_full) >= 30:
            notas_historicas_para_grafico = [round((-(max(min(smi, 60), -60)) + 60) * 10 / 120, 1)
                                             for smi in smi_history_full.tail(30).tolist()]
        elif smi_history_full.empty:
            notas_historicas_para_grafico = [5.0] * 30 # Default neutral if no data
        else:
            # Fill with first available note if less than 30
            first_note_val = round((-(max(min(smi_history_full.iloc[0], 60), -60)) + 60) * 10 / 120, 1)
            notas_historicas_para_grafico = [first_note_val] * (30 - len(smi_history_full)) + \
                                            [round((-(max(min(smi, 60), -60)) + 60) * 10 / 120, 1)
                                             for smi in smi_history_full.tolist()]


        # Precios para el gráfico: 30 días DESPLAZADOS + PROYECCIÓN
        # Necesitamos los últimos (30 + OFFSET_DIAS) precios reales para tener el rango completo
        precios_reales_para_grafico = []
        if len(cierres_history_full) >= (30 + OFFSET_DIAS):
            # Tomamos los 30 precios que se alinearán con las 30 notas (considerando el offset)
            precios_reales_para_grafico = cierres_history_full.iloc[-(30 + OFFSET_DIAS):-OFFSET_DIAS].tolist()
        elif len(cierres_history_full) > OFFSET_DIAS: # Si tenemos menos de 30 pero más que el offset
            # Tomamos lo que tengamos después del offset y rellenamos al principio
            temp_prices = cierres_history_full.iloc[OFFSET_DIAS:].tolist()
            first_price_val = temp_prices[0] if temp_prices else current_price
            precios_reales_para_grafico = [first_price_val] * (30 - len(temp_prices)) + temp_prices
        else: # Muy pocos datos históricos
             precios_reales_para_grafico = [current_price] * 30 # Default to current price if no historical data
            
        # Asegúrate de mantener la misma indentación (los espacios) que las líneas de arriba.
        smi_history_last_30 = hist['SMI_signal'].dropna().tail(30).tolist()
        notas_historicas_ultimos_30_dias_tendencia = [round((-(max(min(smi, 60), -60)) + 60) * 10 / 120, 1) for smi in smi_history_last_30]
        
        # Determinar si la nota ha sido estable en 0 o 10 en la ventana de estabilidad
        NOTE_STABILITY_WINDOW = 5 # Días para considerar la estabilidad de la nota
        
        is_note_stable_at_zero = False
        is_note_stable_at_ten = False

        # Usamos 'notas_historicas_ultimos_30_dias_tendencia' porque contiene los valores numéricos de las notas.
        if len(notas_historicas_ultimos_30_dias_tendencia) >= NOTE_STABILITY_WINDOW:
            # Tomamos las últimas 5 notas
            last_n_notes = notas_historicas_ultimos_30_dias_tendencia[-NOTE_STABILITY_WINDOW:]
            # Verificamos si todas las notas en esa ventana son 0.0 (o muy cercanas)
            if all(abs(note - 0.0) < 0.1 for note in last_n_notes): # Tolerancia de 0.1 para notas muy cercanas a 0
                is_note_stable_at_zero = True
            # Verificamos si todas las notas en esa ventana son 10.0 (o muy cercanas)
            elif all(abs(note - 10.0) < 0.1 for note in last_n_notes): # Tolerancia de 0.1 para notas muy cercanas a 10
                is_note_stable_at_ten = True
        # Si no hay suficientes datos históricos para la ventana, pero la nota actual es 0 o 10
        elif abs(nota_empresa - 0.0) < 0.1:
             is_note_stable_at_zero = True
        elif abs(nota_empresa - 10.0) < 0.1:
             is_note_stable_at_ten = True

        # Versión con LÍMITE ABSOLUTO: no puede perforar resistencias ni soportes
        precios_proyectados = []
        ultimo_precio_conocido = precios_reales_para_grafico[-1] if precios_reales_para_grafico else current_price

        # Determinar techo y suelo absoluto para la proyección
        resistencias_ordenadas = sorted([r for r in resistencias if r > ultimo_precio_conocido])
        soportes_ordenados = sorted([s for s in soportes if s < ultimo_precio_conocido], reverse=True)

        # Elegimos la más próxima como límite duro
        limite_resistencia = resistencias_ordenadas[0] if resistencias_ordenadas else None
        limite_soporte = soportes_ordenados[0] if soportes_ordenados else None

        for _ in range(PROYECCION_FUTURA_DIAS):
            # Calcular intento de movimiento
            if nota_empresa < 2.1:
                siguiente_precio = ultimo_precio_conocido * (1 - 0.015)
            elif nota_empresa > 7.9:
                siguiente_precio = ultimo_precio_conocido * (1 + 0.015)
            else:
                daily_rate_of_change = (nota_empresa - 5.0) * (0.005 / 2.9)
                siguiente_precio = ultimo_precio_conocido * (1 + daily_rate_of_change)

            # 🔒 Aplicar LÍMITE DURO
            if limite_resistencia and siguiente_precio > limite_resistencia:
                siguiente_precio = limite_resistencia * (1 - 0.001)
            elif limite_soporte and siguiente_precio < limite_soporte:
                siguiente_precio = limite_soporte * (1 + 0.001)

            siguiente_precio = round(siguiente_precio, 2)
            precios_proyectados.append(siguiente_precio)
            ultimo_precio_conocido = siguiente_precio



        # Unir precios reales y proyectados
        cierres_para_grafico_total = precios_reales_para_grafico + precios_proyectados






        tendencia_smi = "No disponible"
        dias_estimados_accion = "No disponible"

        if len(notas_historicas_ultimos_30_dias_tendencia) >= 2:
            x = np.arange(len(notas_historicas_ultimos_30_dias_tendencia))
            y = np.array(notas_historicas_ultimos_30_dias_tendencia)
            if np.std(y) > 0.01:
                slope, intercept = np.polyfit(x, y, 1)
            else:
                slope = 0.0

            if slope > 0.1:
                tendencia_smi = "mejorando"
            elif slope < -0.1:
                tendencia_smi = "empeorando"
            else:
                tendencia_smi = "estable"

            # Lógica inicial para días estimados para acción (se puede refinar en la nueva función de decisión)
            if nota_empresa <= 2:
                dias_estimados_accion = "Ya en zona de posible venta"
            elif nota_empresa >= 8:
                dias_estimados_accion = "Ya en zona de posible compra"
            elif abs(slope) < 0.01:
                dias_estimados_accion = "Tendencia estable, sin acción inmediata clara"
            elif slope < 0:
                dias = (nota_empresa - 2.0) / abs(slope)
                dias_estimados_accion = f"aprox. {int(max(1, dias))} días para alcanzar zona de venta"
            elif slope > 0:
                dias = (8.0 - nota_empresa) / abs(slope)
                dias_estimados_accion = f"aprox. {int(max(1, dias))} días para alcanzar zona de compra"


        datos = {
            "TICKER": ticker,
            "NOMBRE_EMPRESA": info.get("longName", ticker),
            "PRECIO_ACTUAL": current_price,
            "VOLUMEN": current_volume,
            "VOLUMEN_MEDIO": round(volumen_promedio_30d, 2) if not pd.isna(volumen_promedio_30d) else "N/A", # <-- AÑADE ESTA LÍNEA
            "SOPORTE_1": soporte_1,
            "SOPORTE_2": soporte_2,
            "SOPORTE_3": soporte_3,
            "RESISTENCIA": resistencia_1,
            "CONDICION_RSI": condicion_rsi, # Se llenará en la nueva función
            "RECOMENDACION": recomendacion, # Se llenará en la nueva función
            "SMI": smi_actual,
            "NOTA_EMPRESA": nota_empresa,
            "PRECIO_OBJETIVO_COMPRA": precio_objetivo_compra,
            "TENDENCIA_NOTA": tendencia_smi,            
            "NOTAS_HISTORICAS_30_DIAS_ANALISIS": notas_historicas_ultimos_30_dias_tendencia, # Para cálculo de la nueva recomendación
            "CIERRES_30_DIAS": hist['Close'].dropna().tail(30).tolist(), # Cierres reales de los últimos 30 días para análisis histórico
            "NOTAS_HISTORICAS_PARA_GRAFICO": notas_historicas_para_grafico, # Solo los 30 días para el gráfico
            "CIERRES_PARA_GRAFICO_TOTAL": cierres_para_grafico_total, # 30 días reales + 5 días proyectados
            "OFFSET_DIAS_GRAFICO": OFFSET_DIAS,
            "RESISTENCIA_1": resistencia_1, # Ahora usamos resistencia_1
            "RESISTENCIA_2": resistencia_2,
            "RESISTENCIA_3": resistencia_3,
            "PRECIO_OBJETIVO": precio_objetivo, # Añadimos el precio objetivo calculado
            "PROYECCION_FUTURA_DIAS_GRAFICO": PROYECCION_FUTURA_DIAS
        }

        return datos

    except Exception as e:
        print(f"❌ Error al obtener datos de {ticker}: {e}. Saltando a la siguiente empresa...")
        return None

def generar_recomendacion_avanzada(data, cierres_para_grafico_total, notas_historicas_para_grafico):
    # Asegúrate de que notas_historicas_para_grafico y cierres_para_grafico_total son listas válidas.
    # Si no lo son, o están vacías, esto podría causar errores.
    # Se asume que data['NOTA_TECNICA'], data['VOLUMEN'], data['VOLUMEN_MEDIO'],
    # data['PRECIO_ACTUAL'], data['SOPORTE_1'], data['RESISTENCIA_1'] ya están poblados.

    # Extraer los últimos 30 días de notas para el análisis de tendencias
    notas_historicas = notas_historicas_para_grafico[-30:] if len(notas_historicas_para_grafico) >= 30 else notas_historicas_para_grafico

    # Calcular la pendiente de las últimas N notas para la tendencia
    n_trend = min(7, len(notas_historicas)) # Últimos 7 días o menos si no hay tantos
    if n_trend > 1:
        x_trend = np.arange(n_trend)
        y_trend = np.array(notas_historicas[-n_trend:])
        # Filtrar NaN para calcular la pendiente
        valid_indices = ~np.isnan(y_trend)
        if np.any(valid_indices): # Solo calcular si hay datos válidos
            slope, _ = np.polyfit(x_trend[valid_indices], y_trend[valid_indices], 1)
        else:
            slope = 0 # No hay datos válidos para la pendiente
    else:
        slope = 0 # No hay suficientes datos para calcular la pendiente

    if slope > 0.1:
        tendencia_nota = "mejorando"
    elif slope < -0.1:
        tendencia_nota = "empeorando"
    else:
        tendencia_nota = "neutral"

    # Determinar si el volumen es alto (ej. > 1.5 veces el volumen medio de los últimos 20 días)
    volumen_alto = False
    if data['VOLUMEN_MEDIO'] and data['VOLUMEN'] is not None and data['VOLUMEN_MEDIO'] > 0: # Añadida verificación para evitar división por cero
        if data['VOLUMEN'] > (data['VOLUMEN_MEDIO'] * 1.5):
            volumen_alto = True

    # Determinar proximidad a soportes y resistencias
    proximidad_soporte = False
    proximidad_resistencia = False
    # Asegurarse de que los valores existen y no son None para evitar errores
    if data['PRECIO_ACTUAL'] is not None:
        if data['SOPORTE_1'] is not None and data['PRECIO_ACTUAL'] != 0:
            if abs(data['PRECIO_ACTUAL'] - data['SOPORTE_1']) / data['PRECIO_ACTUAL'] < 0.02: # 2% de proximidad
                proximidad_soporte = True
        if data['RESISTENCIA_1'] is not None and data['PRECIO_ACTUAL'] != 0:
            if abs(data['PRECIO_ACTUAL'] - data['RESISTENCIA_1']) / data['PRECIO_ACTUAL'] < 0.02: # 2% de proximidad
                proximidad_resistencia = True

    recomendacion = "Neutral"
    condicion_mercado = "En observación"
    motivo_recomendacion = "La situación actual no presenta señales claras de compra ni venta." # NUEVA VARIABLE: Motivo por defecto

    # Lógica de Giro Alcista (Compra)
    if tendencia_nota == "mejorando" and data['NOTA_TECNICA'] >= 5 and volumen_alto:
        recomendacion = "Fuerte Compra"
        condicion_mercado = "Impulso alcista con confirmación de volumen"
        motivo_recomendacion = "La nota técnica está mejorando con un volumen significativo, indicando un fuerte impulso alcista."

    # Lógica de Giro Bajista (Venta Condicional)
    elif tendencia_nota == "empeorando" and data['NOTA_TECNICA'] <= 6 and volumen_alto:
        if not proximidad_soporte:
            recomendacion = "Venta Condicional / Alerta"
            condicion_mercado = "Debilidad confirmada por volumen, considerar salida"
            motivo_recomendacion = "La nota técnica está empeorando con volumen alto y sin soporte cercano, sugiriendo debilidad."
        else:
            recomendacion = "Neutral / Cautela"
            condicion_mercado = "Debilidad pero cerca de soporte clave, observar rebote"
            motivo_recomendacion = "La nota técnica está empeorando, pero la proximidad a un soporte clave sugiere cautela antes de vender."

    # Detección de Patrones de Reversión desde Extremos:
    # Reversión de Compra
    if len(notas_historicas) >= 2 and notas_historicas[-1] < notas_historicas[-2] and \
       notas_historicas[-2] >= 9 and notas_historicas[-1] >= 8:
        if recomendacion not in ["Fuerte Compra", "Oportunidad de Compra (Reversión)"]: # No sobrescribir si ya es una compra fuerte
            recomendacion = "Oportunidad de Compra (Reversión)"
            condicion_mercado = "Posible inicio de corrección tras sobreventa extrema, punto de entrada"
            motivo_recomendacion = "Reversión de compra: La nota técnica está descendiendo desde una zona de sobreventa extrema (cerca de 10), indicando una oportunidad de entrada."

    # Reversión de Venta
    elif len(notas_historicas) >= 2 and notas_historicas[-1] > notas_historicas[-2] and \
         notas_historicas[-2] <= 1 and notas_historicas[-1] <= 2:
        if recomendacion not in ["Venta Condicional / Alerta", "Señal de Venta (Reversión)"]: # No sobrescribir si ya es una venta fuerte
            recomendacion = "Señal de Venta (Reversión)"
            condicion_mercado = "Posible inicio de corrección tras sobrecompra extrema, punto de salida"
            motivo_recomendacion = "Señal de venta: La nota técnica está ascendiendo desde una zona de sobrecompra extrema (cerca de 0), indicando un punto de salida."


    # Lógica para "Neutral" si ninguna de las condiciones anteriores se cumple con fuerza
    # Esta lógica se ejecuta si no se ha establecido una recomendación más fuerte
    if recomendacion == "Neutral":
        if tendencia_nota == "neutral":
            condicion_mercado = "Consolidación o lateralidad sin dirección clara."
            motivo_recomendacion = "La nota técnica se mantiene neutral, indicando una fase de consolidación o lateralidad sin dirección clara."
        elif data['NOTA_TECNICA'] >= 7 and tendencia_nota == "mejorando" and not volumen_alto:
            recomendacion = "Neutral / Observación"
            condicion_mercado = "Nota alta con mejora, pero falta confirmación de volumen."
            motivo_recomendacion = "La nota técnica es alta y muestra una mejora, pero la falta de volumen significativo sugiere una fase de observación."
        elif data['NOTA_TECNICA'] <= 4 and tendencia_nota == "empeorando" and not volumen_alto:
            recomendacion = "Neutral / Observación"
            condicion_mercado = "Nota baja con empeoramiento, pero falta confirmación de volumen."
            motivo_recomendacion = "La nota técnica es baja y empeora, pero la falta de volumen significativo sugiere una fase de observación."

    data['RECOMENDACION'] = recomendacion
    data['CONDICION_RSI'] = condicion_mercado
    data['MOTIVO_RECOMENDACION'] = motivo_recomendacion # AÑADIR ESTA LÍNEA

    return data




def construir_prompt_formateado(data):
    # Generación de la recomendación de volumen
    volumen_analisis_text = ""
    if data['VOLUMEN'] != "N/A":
        volumen_actual = data['VOLUMEN']
        # Obtener volumen histórico para el promedio
        try:
            ticker_obj = yf.Ticker(data['TICKER'])
            hist_vol = ticker_obj.history(period="90d") # Más periodo para un promedio más robusto
            if not hist_vol.empty and 'Volume' in hist_vol.columns:
                volumen_promedio_30d = hist_vol['Volume'].tail(30).mean()
                if volumen_promedio_30d > 0:
                    cambio_porcentual_volumen = ((volumen_actual - volumen_promedio_30d) / volumen_promedio_30d) * 100
                    if cambio_porcentual_volumen > 50: # Volumen significativamente más alto
                        volumen_analisis_text = f"El volumen negociado de <strong>{volumen_actual:,.0f} acciones</strong> es notablemente superior al promedio reciente, indicando un fuerte interés del mercado y validando la actual tendencia de la nota técnica ({data['TENDENCIA_NOTA']})."
                    elif cambio_porcentual_volumen < -30: # Volumen significativamente más bajo
                        volumen_analisis_text = f"El volumen de <strong>{volumen_actual:,.0f} acciones</strong> es inferior a lo habitual, lo que podría sugerir cautela en la actual tendencia. Una confirmación de la señal de la nota técnica ({data['TENDENCIA_NOTA']}) requeriría un aumento en la participación del mercado."
                    else:
                        volumen_analisis_text = f"El volumen de <strong>{volumen_actual:,.0f} acciones</strong> se mantiene en línea con el promedio. Es un volumen adecuado, pero no excepcional, para confirmar de manera contundente la señal de la nota técnica ({data['TENDENCIA_NOTA']})."
                else:
                    volumen_analisis_text = f"El volumen de <strong>{volumen_actual:,.0f} acciones</strong> es importante para confirmar cualquier movimiento. "
            else:
                volumen_analisis_text = f"El volumen de <strong>{volumen_actual:,.0f} acciones</strong> es importante para confirmar cualquier movimiento. "
        except Exception as e:
            volumen_analisis_text = f"El volumen de <strong>{volumen_actual:,.0f} acciones</strong> es importante para confirmar cualquier movimiento. No fue posible comparar con el volumen promedio: {e}"
    else:
        volumen_analisis_text = "El volumen de negociación no está disponible en este momento."

    titulo_post = f"Análisis Técnico: {data['NOMBRE_EMPRESA']} ({data['TICKER']}) - Recomendación de {data['RECOMENDACION']}"

    # Datos para el gráfico principal de Notas y Precios
    notas_historicas_para_grafico = data.get('NOTAS_HISTORICAS_PARA_GRAFICO', [])
    cierres_para_grafico_total = data.get('CIERRES_PARA_GRAFICO_TOTAL', [])
    OFFSET_DIAS = data.get('OFFSET_DIAS_GRAFICO', 4)
    PROYECCION_FUTURA_DIAS = data.get('PROYECCION_FUTURA_DIAS_GRAFICO', 5)

    chart_html = ""
    if notas_historicas_para_grafico and cierres_para_grafico_total:
        # Los labels deben cubrir los 30 días de historial + los días de proyección
        labels_historial = [(datetime.today() - timedelta(days=29 - i)).strftime("%d/%m") for i in range(30)]
        labels_proyeccion = [(datetime.today() + timedelta(days=i)).strftime("%d/%m (fut.)") for i in range(1, PROYECCION_FUTURA_DIAS + 1)]
        labels_total = labels_historial + labels_proyeccion

        # Precios históricos reales para el gráfico (son los últimos 30 días, ya desplazados)
        precios_reales_grafico = cierres_para_grafico_total[:30]

        # Precios proyectados (dashed line)
        # Esto asegura que la parte de los precios proyectados empiece en el día 30 del dataset
        # y que el resto del array esté relleno con None hasta el punto de inicio de la proyección real.
        data_proyectada = [None] * (len(labels_historial) - 1) + [precios_reales_grafico[-1]] + cierres_para_grafico_total[len(labels_historial):]

        # Las notas históricas ya están calculadas para los 30 días
        # Para la visualización de "nota D vs precio D+4", las líneas se desplazan visualmente, no los datos.
        # Insertamos Nones al principio de la nota para desplazarla visualmente 4 días a la derecha.
        notas_desplazadas_para_grafico = [None] * OFFSET_DIAS + notas_historicas_para_grafico
        # La longitud del array de notas desplazadas debe coincidir con la de labels_total
        # Si notas_desplazadas_para_grafico es más corto que labels_total, rellenar con None
        if len(notas_desplazadas_para_grafico) < len(labels_total):
            notas_desplazadas_para_grafico.extend([None] * (len(labels_total) - len(notas_desplazadas_para_grafico)))

        chart_html += f"""
        <h2>Evolución de la Nota Técnica y Precio</h2>
        <p>Para ofrecer una perspectiva visual clara de la evolución de la nota técnica...
        Es importante recordar que la nota de hoy (D) se alinea con el precio de D+4,
        lo que significa que la reacción del mercado a la nota SMI generalmente se observa
        unos pocos días después de su formación.
        </p>
        <div style="width: 100%; max-width: 800px; margin: auto;">
            <canvas id="notaPrecioChart" style="height: 600px;"></canvas>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@1.4.0"></script>
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const ctx = document.getElementById('notaPrecioChart').getContext('2d');
            const notaPrecioChart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(labels_total)},
                    datasets: [
                        {{
                            label: 'Nota Técnica SMI',
                            data: {json.dumps(notas_desplazadas_para_grafico)},
                            borderColor: 'rgba(75, 192, 192, 1)',
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            yAxisID: 'y',
                            tension: 0.1,
                            fill: false
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
                            text: 'Evolución de Nota Técnica y Precio'
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
                                // Zonas de color en el eje Y de la Nota Técnica (0-10)
                                zonaCompra: {{
                                    type: 'box',
                                    yScaleID: 'y',
                                    yMin: 8,
                                    yMax: 10,
                                    backgroundColor: 'rgba(0, 255, 0, 0.1)', // Verde claro para compra
                                    borderColor: 'rgba(0, 255, 0, 0.2)',
                                    borderWidth: 1,
                                    label: {{
                                        content: 'Zona de Compra',
                                        enabled: false,
                                        position: 'end',
                                        color: 'rgba(0, 150, 0, 0.7)',
                                        font: {{ size: 12, weight: 'bold' }}
                                    }}
                                }},
                                zonaNeutral: {{
                                    type: 'box',
                                    yScaleID: 'y',
                                    yMin: 2,
                                    yMax: 8,
                                    backgroundColor: 'rgba(255, 255, 0, 0.1)', // Amarillo claro para neutral
                                    borderColor: 'rgba(255, 255, 0, 0.2)',
                                    borderWidth: 1,
                                    label: {{
                                        content: 'Zona Neutral',
                                        enabled: false,
                                        position: 'center',
                                        color: 'rgba(150, 150, 0, 0.7)',
                                        font: {{ size: 12, weight: 'bold' }}
                                    }}
                                }},
                                zonaVenta: {{
                                    type: 'box',
                                    yScaleID: 'y',
                                    yMin: 0,
                                    yMax: 2,
                                    backgroundColor: 'rgba(255, 0, 0, 0.1)', // Rojo claro para venta
                                    borderColor: 'rgba(255, 0, 0, 0.2)',
                                    borderWidth: 1,
                                    label: {{
                                        content: 'Zona de Venta',
                                        enabled: false,
                                        position: 'start',
                                        color: 'rgba(150, 0, 0, 0.7)',
                                        font: {{ size: 12, weight: 'bold' }}
                                    }}
                                }},
                                // Líneas de soporte
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
                                        font: {{ size: 10 }}
                                    }}
                                }},
                                soporte2Line: {{
                                    type: 'line',
                                    yScaleID: 'y1',
                                    yMin: {data['SOPORTE_2']},
                                    yMax: {data['SOPORTE_2']},
                                    borderColor: 'rgba(75, 192, 192, 0.8)',
                                    borderWidth: 2,
                                    borderDash: [6, 6],
                                    label: {{
                                        content: 'Soporte 2',
                                        enabled: true,
                                        position: 'start',
                                        font: {{ size: 10 }}
                                    }}
                                }},
                                soporte3Line: {{
                                    type: 'line',
                                    yScaleID: 'y1',
                                    yMin: {data['SOPORTE_3']},
                                    yMax: {data['SOPORTE_3']},
                                    borderColor: 'rgba(75, 192, 192, 0.8)',
                                    borderWidth: 2,
                                    borderDash: [6, 6],
                                    label: {{
                                        content: 'Soporte 3',
                                        enabled: true,
                                        position: 'start',
                                        font: {{ size: 10 }}
                                    }}
                                }},
                                // Líneas de resistencia
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
                                        font: {{ size: 10 }}
                                    }}
                                }},
                                resistencia2Line: {{
                                    type: 'line',
                                    yScaleID: 'y1',
                                    yMin: {data['RESISTENCIA_2']},
                                    yMax: {data['RESISTENCIA_2']},
                                    borderColor: 'rgba(255, 99, 132, 0.8)',
                                    borderWidth: 2,
                                    borderDash: [6, 6],
                                    label: {{
                                        content: 'Resistencia 2',
                                        enabled: true,
                                        position: 'end',
                                        font: {{ size: 10 }}
                                    }}
                                }},
                                resistencia3Line: {{
                                    type: 'line',
                                    yScaleID: 'y1',
                                    yMin: {data['RESISTENCIA_3']},
                                    yMax: {data['RESISTENCIA_3']},
                                    borderColor: 'rgba(255, 99, 132, 0.8)',
                                    borderWidth: 2,
                                    borderDash: [6, 6],
                                    label: {{
                                        content: 'Resistencia 3',
                                        enabled: true,
                                        position: 'end',
                                        font: {{ size: 10 }}
                                    }}
                                }}
                                // Icono de Entrada/Salida en el último día de historial
                                // (último día del dataset de precios reales)
                                signalPoint:{{
                                    type: 'point',
                                    xValue: {len(labels_historial) - 1}, // Último día del historial real
                                    yValue: {precios_reales_grafico[-1] if precios_reales_grafico else 'null'}, // Precio del último día real
                                    yScaleID: 'y1', // En el eje del precio
                                    radius: 10,
                                    pointStyle: {'"triangle"' if 'Compra' in data['RECOMENDACION'] else ('"triangle"' if 'Venta' in data['RECOMENDACION'] else '"circle"')}, // Triángulo para compra/venta
                                    rotation: {{0 if 'Compra' in data['RECOMENDACION'] else (180 if 'Venta' in data['RECOMENDACION'] else 0)}}, // Girar para venta
                                    backgroundColor: {'"rgba(0, 200, 0, 0.8)"' if 'Compra' in data['RECOMENDACION'] else ('"rgba(200, 0, 0, 0.8)"' if 'Venta' in data['RECOMENDACION'] else '"rgba(100, 100, 100, 0.8)"')}, // Verde para compra, Rojo para venta, Gris para neutral
                                    borderColor: 'white',
                                    borderWidth: 2,
                                    display: {('true' if 'Compra' in data['RECOMENDACION'] or 'Venta' in data['RECOMENDACION'] else 'false')}, // Mostrar solo si es compra o venta
                                    label: {{
                                        content: '{data["RECOMENDACION"]}',
                                        enabled: true,
                                        position: 'top', // O 'bottom'
                                        font: {{ size: 10, weight: 'bold' }},
                                        color: {'"rgba(0, 200, 0, 0.8)"' if 'Compra' in data['RECOMENDACION'] else ('"rgba(200, 0, 0, 0.8)"' if 'Venta' in data['RECOMENDACION'] else '"rgba(100, 100, 100, 0.8)"')},
                                        backgroundColor: 'rgba(255, 255, 255, 0.7)',
                                        borderRadius: 4,
                                        padding: 4
                                    }}
                                }}                                
                            }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {{
                                display: true,
                                text: 'Nota Técnica (0-10)'
                            }},
                            min: 0,
                            max: 10
                        }},
                        y1: {{
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {{
                                display: true,
                                text: 'Precio'
                            }},
                            grid: {{
                                drawOnChartArea: false,
                            }},
                        }}
                    }}
                }}
            }});
        }});
        </script>
        """

    # Pre-procesamiento de soportes para agruparlos si son muy cercanos
    soportes_unicos = []
    temp_soportes = sorted([data['SOPORTE_1'], data['SOPORTE_2'], data['SOPORTE_3']], reverse=True)
    
    if len(temp_soportes) > 0:
        soportes_unicos.append(temp_soportes[0])
        for i in range(1, len(temp_soportes)):
            if abs(temp_soportes[i] - soportes_unicos[-1]) / soportes_unicos[-1] > 0.005: # Tolerancia del 0.5%
                soportes_unicos.append(temp_soportes[i])
    
    # Asegurarse de que soportes_unicos tenga al menos un elemento para la tabla
    if not soportes_unicos:
        soportes_unicos.append(0.0) # Valor por defecto si no se encontraron soportes

    # Construcción del texto de soportes
    soportes_texto = ""
    if len(soportes_unicos) == 1:
        soportes_texto = f"un soporte clave en <strong>{soportes_unicos[0]:,.2f}€</strong>."
    elif len(soportes_unicos) == 2:
        soportes_texto = f"dos soportes importantes en <strong>{soportes_unicos[0]:,.2f}€</strong> y <strong>{soportes_unicos[1]:,.2f}€</strong>."
    elif len(soportes_unicos) >= 3:
        soportes_texto = (f"tres soportes relevantes: el primero en <strong>{soportes_unicos[0]:,.2f}€</strong>, "
                          f"el segundo en <strong>{soportes_unicos[1]:,.2f}€</strong>, y el tercero en <strong>{soportes_unicos[2]:,.2f}€</strong>.")
    else:
        soportes_texto = "no presenta soportes claros en el análisis reciente, requiriendo un seguimiento cauteloso."

    # Construcción de la tabla de resumen de puntos clave
    tabla_resumen = f"""
<h2>Resumen de Puntos Clave</h2>
<table border="1" style="width:100%; border-collapse: collapse;">
    <tr>
        <th style="padding: 8px; text-align: left; background-color: #f2f2f2;">Métrica</th>
        <th style="padding: 8px; text-align: left; background-color: #f2f2f2;">Valor</th>
    </tr>
    <tr>
        <td style="padding: 8px;">Precio Actual</td>
        <td style="padding: 8px;"><strong>{data['PRECIO_ACTUAL']:,}€</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Volumen</td>
        <td style="padding: 8px;"><strong>{data['VOLUMEN']:,} acciones</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Soporte Clave</td>
        <td style="padding: 8px;"><strong>{soportes_unicos[0]:,.2f}€</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Resistencia Clave</td>
        <td style="padding: 8px;"><strong>{data['RESISTENCIA']:,}€</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Recomendación</td>
        <td style="padding: 8px;"><strong>{data['RECOMENDACION']}</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Nota Técnica (0-10)</td>
        <td style="padding: 8px;"><strong>{data['NOTA_EMPRESA']:,}</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Precio Objetivo de Compra</td>
        <td style="padding: 8px;"><strong>{data['PRECIO_OBJETIVO_COMPRA']:,}€</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Tendencia de la Nota</td>
        <td style="padding: 8px;"><strong>{data['TENDENCIA_NOTA']}</strong></td>
    </tr>
</table>
<br/>
"""

    prompt = f"""
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
- Resistencia clave: {data['RESISTENCIA']}
- Recomendación general: {data['RECOMENDACION']}
- Nota de la empresa (0-10): {data['NOTA_EMPRESA']}
- Precio objetivo de compra: {data['PRECIO_OBJETIVO_COMPRA']}€
- Tendencia de la nota: {data['TENDENCIA_NOTA']}


Importante: si algún dato no está disponible ("N/A", "No disponibles", "No disponible"), no lo menciones ni digas que falta. No expliques que la recomendación proviene de un indicador o dato específico. La recomendación debe presentarse como una conclusión personal basada en tu experiencia y criterio profesional como analista. Al redactar el análisis, haz referencia a la **nota obtenida por la empresa ({data['NOTA_EMPRESA']})** en al menos dos de los párrafos principales (Recomendación General, Análisis a Corto Plazo o Predicción a Largo Plazo) como un factor clave para tu valoración.

---
<h1>{titulo_post}</h1>


<h2>Análisis Inicial y Recomendación</h2>
<p><strong>{data['NOMBRE_EMPRESA']} ({data['TICKER']})</strong> cotiza actualmente a <strong>{data['PRECIO_ACTUAL']:,}€</strong>. Mi precio objetivo de compra se sitúa en <strong>{data['PRECIO_OBJETIVO_COMPRA']:,}€</strong>. El volumen negociado recientemente, alcanzó las <strong>{data['VOLUMEN']:,} acciones</strong>.</p>

<p>Asignamos una <strong>nota técnica de {data['NOTA_EMPRESA']} sobre 10</strong>. Esta puntuación, combinada con el análisis de la **{data['TENDENCIA_NOTA']}** de la nota, la proximidad a soportes y resistencias, y el volumen, nos lleva a una recomendación de <strong>{data['RECOMENDACION']}</strong>. Actualmente, el mercado se encuentra en una situación de <strong>{data['CONDICION_RSI']}</strong>. Esto refleja {'una excelente fortaleza técnica y baja volatilidad esperada a corto plazo, lo que indica un bajo riesgo técnico en relación con el potencial de crecimiento.' if "Compra" in data['RECOMENDACION'] and data['NOTA_EMPRESA'] >= 7 else ''}
{'una fortaleza técnica moderada, con un equilibrio entre potencial y riesgo, sugiriendo una oportunidad que requiere seguimiento.' if "Compra Moderada" in data['RECOMENDACION'] or (data['NOTA_EMPRESA'] >= 5 and data['NOTA_EMPRESA'] < 7) else ''}
{'una situación técnica neutral, donde el gráfico no muestra un patrón direccional claro, indicando que es un momento para la observación y no para la acción inmediata.' if "Neutral" in data['RECOMENDACION'] else ''}
{'cierta debilidad técnica, con posibles señales de corrección o continuación bajista, mostrando una pérdida de impulso alcista y un aumento de la presión vendedora.' if "Venta" in data['RECOMENDACION'] or (data['NOTA_EMPRESA'] < 5 and data['NOTA_EMPRESA'] >= 3) else ''}
{'una debilidad técnica significativa y una posible sobrecompra en el gráfico, lo que sugiere un alto riesgo de corrección.' if "Venta Fuerte" in data['RECOMENDACION'] or data['NOTA_EMPRESA'] < 3 else ''} </p>
{chart_html}



{tabla_resumen}

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

    return prompt, titulo_post


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
        notas_historicas_para_grafico = data.get('NOTAS_HISTORICAS_PARA_GRAFICO', [])

        # Ahora pasa estas variables a la función generar_recomendacion_avanzada
        data = generar_recomendacion_avanzada(data, cierres_para_grafico_total, notas_historicas_para_grafico)
        

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
    ticker_deseado = "IBE.MC"  # <-- ¡CAMBIA "BBVA.MC" por el Ticker que quieras analizar!
                                # Por ejemplo: "REP.MC", "TSLA", etc.

    # Prepara la lista de tickers para la función generar_contenido_con_gemini
    # que espera una lista de tickers
    tickers_for_today = [ticker_deseado]

    if tickers_for_today:
        print(f"\nAnalizando el ticker solicitado: {ticker_deseado}")
        generar_contenido_con_gemini(tickers_for_today)
    else:
        print(f"No se especificó ningún ticker para analizar.")

if __name__ == '__main__':
    main()
