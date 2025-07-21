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


        smi_actual = round(hist['SMI_signal'].iloc[-1], 2)
        current_price = round(info["currentPrice"], 2)
        current_volume = info.get("volume", "N/A")


        # Calcular soportes y resistencia
        # Asegurarse de tener al menos 30 días para un cálculo significativo
        if len(hist) < 30:
            # print(f"Advertencia: Menos de 30 días de historial para {ticker}. Los soportes pueden ser menos precisos.")
            highs_lows = hist[['High', 'Low', 'Close']].values.flatten()
        else:
            highs_lows = hist[['High', 'Low', 'Close']].iloc[-30:].values.flatten()

        soportes = np.sort(np.unique(highs_lows))
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
            soporte_1, soporte_2, soporte_3 = current_price * 0.95, current_price * 0.9, current_price * 0.85 # Default si no hay datos

        # Precio objetivo de compra (ejemplo simple, puedes refinarlo)
        precio_objetivo_compra = round(current_price * 0.98, 2) # Un 2% por debajo del precio actual como ejemplo

        # Calculamos la nota técnica actual
        nota_empresa = round((-(max(min(smi_actual, 60), -60)) + 60) * 10 / 120, 1)

        # Inicializar recomendacion y condicion_rsi como temporales, se recalcularán después
        recomendacion = "Pendiente de análisis avanzado"
        condicion_rsi = "Pendiente"


        # Nuevas variables para los gráficos con offset y proyección
        OFFSET_DIAS = 4 # La nota de hoy (D) se alinea con el precio de D+4
        PROYECCION_FUTURA_DIAS = 5 # Días a proyectar después del último precio real

        # Aseguramos tener suficientes datos para el historial, el offset y la proyección
        smi_history_full = hist_extended['SMI_signal'].dropna()
        cierres_history_full = hist_extended['Close'].dropna()

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

        # Proyección para los próximos 5 días (ej. usando el último precio real)
        ultimo_precio_conocido = precios_reales_para_grafico[-1] if precios_reales_para_grafico else current_price
        precios_proyectados = [ultimo_precio_conocido] * PROYECCION_FUTURA_DIAS

        # Unir precios reales y proyectados
        cierres_para_grafico_total = precios_reales_para_grafico + precios_proyectados

        # La tendencia de la nota y días estimados (para la explicación)
        # Esto se mantiene igual, calculada sobre los 30 días más recientes del 'hist'
        smi_history_last_30 = hist['SMI_signal'].dropna().tail(30).tolist()
        notas_historicas_ultimos_30_dias_tendencia = [round((-(max(min(smi, 60), -60)) + 60) * 10 / 120, 1) for smi in smi_history_last_30]

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
            "SOPORTE_1": soporte_1,
            "SOPORTE_2": soporte_2,
            "SOPORTE_3": soporte_3,
            "RESISTENCIA": round(hist_extended["High"].max(), 2), # Usar el high máximo de todo el historial extendido
            "CONDICION_RSI": condicion_rsi, # Se llenará en la nueva función
            "RECOMENDACION": recomendacion, # Se llenará en la nueva función
            "SMI": smi_actual,
            "NOTA_EMPRESA": nota_empresa,
            "PRECIO_OBJETIVO_COMPRA": precio_objetivo_compra,
            "INGRESOS": info.get("totalRevenue", "N/A"),
            "EBITDA": info.get("ebitda", "N/A"),
            "BENEFICIOS": info.get("grossProfits", "N/A"),
            "DEUDA": info.get("totalDebt", "N/A"),
            "FLUJO_CAJA": info.get("freeCashflow", "N/A"),
            "EXPANSION_PLANES": "Información de planes de expansión no disponible o no traducible en este momento.",
            "ACUERDOS": "Información sobre acuerdos no disponible o no traducible en este momento.",
            "SENTIMIENTO_ANALISTAS": "Sentimiento de analistas no disponible o no traducible.",
            "TENDENCIA_SOCIAL": "No disponible",
            "EMPRESAS_SIMILARES": ", ".join(info.get("category", "").split(",")) if info.get("category") else "No disponibles",
            "RIESGOS_OPORTUNIDADES": "No disponibles",
            "TENDENCIA_NOTA": tendencia_smi,
            "DIAS_ESTIMADOS_ACCION": dias_estimados_accion, # Se ajustará en la nueva función
            "NOTAS_HISTORICAS_30_DIAS_ANALISIS": notas_historicas_ultimos_30_dias_tendencia, # Para cálculo de la nueva recomendación
            "CIERRES_30_DIAS": hist['Close'].dropna().tail(30).tolist(), # Cierres reales de los últimos 30 días para análisis histórico
            "NOTAS_HISTORICAS_PARA_GRAFICO": notas_historicas_para_grafico, # Solo los 30 días para el gráfico
            "CIERRES_PARA_GRAFICO_TOTAL": cierres_para_grafico_total, # 30 días reales + 5 días proyectados
            "OFFSET_DIAS_GRAFICO": OFFSET_DIAS,
            "PROYECCION_FUTURA_DIAS_GRAFICO": PROYECCION_FUTURA_DIAS
        }

        return datos

    except Exception as e:
        print(f"❌ Error al obtener datos de {ticker}: {e}. Saltando a la siguiente empresa...")
        return None

def generar_recomendacion_avanzada(data):
    nota_empresa = data['NOTA_EMPRESA']
    tendencia_nota = data['TENDENCIA_NOTA']
    current_price = data['PRECIO_ACTUAL']
    soporte_1 = data['SOPORTE_1']
    resistencia = data['RESISTENCIA']
    volumen = data['VOLUMEN']
    notas_historicas = data['NOTAS_HISTORICAS_30_DIAS_ANALISIS'] # Usar las notas para análisis de tendencia
    cierres_historicos_analisis = data['CIERRES_30_DIAS'] # Para comparar volumen con histórico

    # Calcular volumen promedio (simplificado, puedes mejorarlo con un periodo más largo)
    # Asegurarse de que hay suficientes datos para calcular el volumen_historico_reciente
    volumen_historico_reciente = 0
    if cierres_historicos_analisis and len(cierres_historicos_analisis) > 1:
        try:
            # Obtener el volumen histórico directamente de yfinance para un periodo más largo para el promedio
            ticker_obj = yf.Ticker(data['TICKER'])
            hist_vol = ticker_obj.history(period="90d")
            if not hist_vol.empty and 'Volume' in hist_vol.columns:
                volumen_historico_reciente = hist_vol['Volume'].tail(30).mean()
        except Exception as e:
            print(f"Advertencia: No se pudo obtener volumen histórico para promedio: {e}")
            volumen_historico_reciente = 0 # Fallback

    volumen_alto = volumen > (volumen_historico_reciente * 1.2) if volumen_historico_reciente else False # 20% por encima del promedio
    volumen_bajo = volumen < (volumen_historico_reciente * 0.8) if volumen_historico_reciente else False # 20% por debajo del promedio

    recomendacion = "Neutral"
    condicion_mercado = "En observación"
    dias_accion = data['DIAS_ESTIMADOS_ACCION'] # Valor inicial

    # Distancia a soportes y resistencias
    distancia_soporte = abs(current_price - soporte_1) / soporte_1 if soporte_1 else float('inf')
    distancia_resistencia = abs(current_price - resistencia) / resistencia if resistencia else float('inf')
    proximidad_soporte = distancia_soporte < 0.03 # A menos del 3% del soporte
    proximidad_resistencia = distancia_resistencia < 0.03 # A menos del 3% de la resistencia


    # Lógica de Giro Alcista (Compra)
    # Si la nota está mejorando significativamente Y el volumen lo acompaña
    if tendencia_nota == "mejorando" and nota_empresa >= 5 and volumen_alto:
        recomendacion = "Fuerte Compra"
        condicion_mercado = "Impulso alcista con confirmación de volumen"
        if "aprox." in dias_accion: # Ajustar si la nota ya es alta y la tendencia la lleva a compra
             if nota_empresa >= 7: # Si ya estamos en zona buena, la acción es "inmediata"
                 dias_accion = "Acción de compra inminente o ya activa"
        elif nota_empresa < 8 and ("Ya en zona de posible compra" in dias_accion or "inminente" in dias_accion):
            dias_accion = "Acción de compra inminente o ya activa" # Sobreescribir si el giro es muy fuerte

    # Lógica de Giro Bajista (Venta Condicional)
    # Si la nota está empeorando significativamente Y el volumen es alto (confirmación de debilidad)
    elif tendencia_nota == "empeorando" and nota_empresa <= 6 and volumen_alto:
        if not proximidad_soporte: # No vender si ya estamos en soporte fuerte
            recomendacion = "Venta Condicional / Alerta"
            condicion_mercado = "Debilidad confirmada por volumen, considerar salida"
            if "aprox." in dias_accion: # Ajustar si la nota ya es baja y la tendencia la lleva a venta
                 if nota_empresa <= 3: # Si ya estamos en zona mala, la acción es "inmediata"
                     dias_accion = "Acción de venta inminente o ya activa"
            elif nota_empresa > 2 and ("Ya en zona de posible venta" in dias_accion or "inminente" in dias_accion):
                dias_accion = "Acción de venta inminente o ya activa" # Sobreescribir si el giro es muy fuerte
        else:
            recomendacion = "Neutral / Cautela"
            condicion_mercado = "Debilidad pero cerca de soporte clave, observar rebote"

    # Detección de Patrones de Reversión desde Extremos (histórico reciente)
    if len(notas_historicas) >= 3:
        # Reversión de Compra (Nota subió a extremo, empieza a descender desde sobreventa)
        # Por ejemplo: ...10, 9, 8.5 (giro hacia abajo desde sobreventa)
        # Si la nota actual ha bajado un poco desde un pico extremo de sobreventa (nota alta)
        if notas_historicas[-1] < notas_historicas[-2] and \
           notas_historicas[-2] >= 9 and notas_historicas[-1] >= 8: # Bajando de 9/10 a 8/9
            recomendacion = "Oportunidad de Compra (Reversión)"
            condicion_mercado = "Posible inicio de corrección tras sobreventa extrema, punto de entrada"
            dias_accion = "Acción de compra (reversión) inminente"

        # Reversión de Venta (Nota bajó a extremo, empieza a subir desde sobrecompra)
        # Por ejemplo: ...0, 1, 1.5 (giro hacia arriba desde sobrecompra)
        # Si la nota actual ha subido un poco desde un valle extremo de sobrecompra (nota baja)
        elif notas_historicas[-1] > notas_historicas[-2] and \
             notas_historicas[-2] <= 1 and notas_historicas[-1] <= 2: # Subiendo de 0/1 a 1/2
            recomendacion = "Señal de Venta (Reversión)"
            condicion_mercado = "Posible inicio de corrección tras sobrecompra extrema, punto de salida"
            dias_accion = "Acción de venta (reversión) inminente"

    # Lógica basada en Nivel Absoluto de la Nota (como fallback o refuerzo)
    if recomendacion == "Neutral" or recomendacion == "En observación" or recomendacion == "Pendiente de análisis avanzado": # Si no se activó una señal fuerte por giros/reversiones
        if nota_empresa >= 8:
            recomendacion = "Compra Fuerte"
            condicion_mercado = "Nota técnica muy sólida"
        elif 6 <= nota_empresa < 8:
            recomendacion = "Compra Moderada"
            condicion_mercado = "Nota técnica favorable"
        elif 4 <= nota_empresa < 6:
            recomendacion = "Neutral"
            condicion_mercado = "Nota técnica balanceada"
        elif 2 < nota_empresa < 4:
            recomendacion = "Evitar / Venta Moderada"
            condicion_mercado = "Nota técnica débil"
        elif nota_empresa <= 2:
            recomendacion = "Venta Fuerte"
            condicion_mercado = "Nota técnica muy débil / sobrecompra"


    # Consideraciones de proximidad a soportes/resistencias para ajustar la recomendación
    if proximidad_soporte and ("Venta" in recomendacion or "Alerta" in recomendacion):
        recomendacion = "Neutral / Observar Soporte"
        condicion_mercado = "Cerca de soporte clave, no forzar venta. Posible rebote."
    elif proximidad_resistencia and ("Compra" in recomendacion):
        recomendacion = "Neutral / Observar Resistencia"
        condicion_mercado = "Cerca de resistencia clave, observar posible reversión."


    data['RECOMENDACION'] = recomendacion
    data['CONDICION_RSI'] = condicion_mercado # Renombramos a condición_mercado para ser más general
    data['DIAS_ESTIMADOS_ACCION'] = dias_accion # Actualizar con la nueva lógica

    return data

def analizar_oportunidades_historicas(data):
    inversion_base = 10000.0
    comision_por_operacion_porcentual = 0.001
    
    cierres = data.get("CIERRES_30_DIAS", []) # Usar los cierres reales de los últimos 30 días
    notas = data.get("NOTAS_HISTORICAS_30_DIAS_ANALISIS", []) # Usar las notas para análisis

    mejor_compra = None
    mejor_venta = None
    mejor_punto_giro_compra = None
    mejor_punto_giro_venta = None

    if cierres and notas and len(cierres) == len(notas):
        for i in range(len(notas)):
            nota = notas[i]
            cierre = cierres[i]

            # Lógica de "Mejor Compra" (nota alta)
            if nota >= 8: # Umbral de compra fuerte
                # Asegurarse de que el índice no exceda el límite al calcular max_post
                max_post = max(cierres[i:], default=cierre)
                if max_post > cierre: # Solo considerar si hubo ganancia
                    pct = ((max_post - cierre) / cierre) * 100
                    ganancia_bruta = (max_post - cierre) * (inversion_base / cierre)
                    comision_compra = inversion_base * comision_por_operacion_porcentual
                    comision_venta = (inversion_base + ganancia_bruta) * comision_por_operacion_porcentual
                    ganancia_neta_compra_actual = ganancia_bruta - (comision_compra + comision_venta)

                    if (not mejor_compra) or (ganancia_neta_compra_actual > (mejor_compra[4] if mejor_compra else -float('inf'))):
                        mejor_compra = (i, cierre, max_post, pct, ganancia_neta_compra_actual)

            # Lógica de "Mejor Venta" (nota baja)
            elif nota <= 2: # Umbral de venta fuerte
                min_post = min(cierres[i:], default=cierre)
                if min_post < cierre: # Solo considerar si hubo caída (pérdida evitada)
                    pct = ((min_post - cierre) / cierre) * 100 # Será negativo
                    perdida_bruta_evitada = (cierre - min_post) * (inversion_base / cierre)
                    comision_compra_imaginaria = inversion_base * comision_por_operacion_porcentual
                    comision_venta_simulada = (inversion_base - (inversion_base * abs(pct)/100)) * comision_por_operacion_porcentual
                    perdida_evitada_neta_actual = perdida_bruta_evitada - (comision_compra_imaginaria + comision_venta_simulada)

                    if (not mejor_venta) or (perdida_evitada_neta_actual > (mejor_venta[4] if mejor_venta else -float('inf'))):
                        mejor_venta = (i, cierre, min_post, pct, perdida_evitada_neta_actual)

        # Lógica de Puntos de Giro (debe ocurrir para al menos 3 puntos en notas históricas)
        if len(notas) >= 3:
            for i in range(1, len(notas) - 1):
                nota_anterior = notas[i-1]
                nota_actual = notas[i]
                nota_siguiente = notas[i+1]
                cierre_actual = cierres[i] if i < len(cierres) else None # Ensure cierre_actual exists

                if cierre_actual is None: continue # Skip if no price data

                # Giro Alcista (Nota disminuye y luego aumenta - posible compra)
                # nota_anterior > nota_actual (baja), nota_siguiente > nota_actual (luego sube)
                # Y nota_actual está en zona baja (sobrecompra o neutral hacia sobrecompra)
                if nota_anterior > nota_actual and nota_siguiente > nota_actual and nota_actual <= 4: # Nota baja y gira hacia arriba
                    max_post = max(cierres[i:], default=cierre_actual)
                    if max_post > cierre_actual:
                        pct_giro = ((max_post - cierre_actual) / cierre_actual) * 100
                        ganancia_bruta = (max_post - cierre_actual) * (inversion_base / cierre_actual)
                        comision_compra = inversion_base * comision_por_operacion_porcentual
                        comision_venta = (inversion_base + ganancia_bruta) * comision_por_operacion_porcentual
                        ganancia_neta_actual = ganancia_bruta - (comision_compra + comision_venta)

                        if (not mejor_punto_giro_compra) or (ganancia_neta_actual > (mejor_punto_giro_compra[4] if mejor_punto_giro_compra else -float('inf'))):
                            mejor_punto_giro_compra = (i, cierre_actual, max_post, pct_giro, ganancia_neta_actual)

                # Giro Bajista (Nota aumenta y luego disminuye - posible venta)
                # nota_anterior < nota_actual (sube), nota_siguiente < nota_actual (luego baja)
                # Y nota_actual está en zona alta (sobreventa o neutral hacia sobreventa)
                if nota_anterior < nota_actual and nota_siguiente < nota_actual and nota_actual >= 6: # Nota alta y gira hacia abajo
                    min_post = min(cierres[i:], default=cierre_actual)
                    if min_post < cierre_actual:
                        pct_giro = ((min_post - cierre_actual) / cierre_actual) * 100 # Será negativo
                        perdida_bruta_evitada = (cierre_actual - min_post) * (inversion_base / cierre_actual)
                        comision_compra_imaginaria = inversion_base * comision_por_operacion_porcentual
                        comision_venta_simulada = (inversion_base - (inversion_base * abs(pct_giro)/100)) * comision_por_operacion_porcentual
                        perdida_evitada_neta_actual = perdida_bruta_evitada - (comision_compra_imaginaria + comision_venta_simulada)

                        if (not mejor_punto_giro_venta) or (perdida_evitada_neta_actual > (mejor_punto_giro_venta[4] if mejor_punto_giro_venta else -float('inf'))):
                            mejor_punto_giro_venta = (i, cierre_actual, min_post, pct_giro, perdida_evitada_neta_actual)

    data['mejor_compra_historica'] = mejor_compra
    data['mejor_venta_historica'] = mejor_venta
    data['mejor_punto_giro_compra_historica'] = mejor_punto_giro_compra
    data['mejor_punto_giro_venta_historica'] = mejor_punto_giro_venta
    data['inversion_base'] = inversion_base
    data['comision_por_operacion_porcentual'] = comision_por_operacion_porcentual

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
        data_proyectada = [None] * 30 + cierres_para_grafico_total[30:]


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
<p>Para ofrecer una perspectiva visual clara de la evolución de la nota técnica de <strong>{data['NOMBRE_EMPRESA']}</strong> y su relación con el precio, mostramos un gráfico que **desplaza nuestra nota técnica {OFFSET_DIAS} días hacia el futuro** en relación al precio. Esto significa que la nota que observas hoy (día D) se alinea con el precio que esperamos ver en el día D+{OFFSET_DIAS}. Esta calificación es una herramienta exclusiva de <strong>ibexia.es</strong> y representa el histórico entre nuestra valoración técnica (línea azul) sobre el precio de cotización (línea roja). La escala de la nota va de 0 (venta o cautela) a 10 (oportunidad de compra).</p>
<p>Además, hemos extendido la línea de precio con una **proyección punteada para los próximos {PROYECCION_FUTURA_DIAS} días**, basada en el último precio real. Esto te ofrece una estimación visual del futuro inmediato del valor.</p>
<p>Para comprender mejor cómo interpretar estos gráficos y tomar decisiones informadas, visita nuestro enlace explicativo: <a href="https://ibexia.es/como-interpretar-los-graficos-para-comprar-o-vender/" target="_blank">Cómo interpretar los gráficos para comprar o vender</a>. </p>

<div style="width: 100%; height: 500px;">
    <canvas id="notasChart"></canvas>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@1.1.0"></script>
<script>
    document.addEventListener('DOMContentLoaded', function() {{

        var cierres = {json.dumps(precios_reales_grafico)}; // Solo la parte real
        var cierresProyectados = {json.dumps(cierres_para_grafico_total)}; // Real + Proyectado (para la línea completa y luego dashed)
        var objetivo = {data['PRECIO_OBJETIVO_COMPRA']};
        var notas = {json.dumps(notas_desplazadas_para_grafico)}; // Notas ya desplazadas con Nones
        var labels = {json.dumps(labels_total)}; // Labels con días futuros

        var minPrecio = Math.min(...cierres.filter(n => n !== null), objetivo);
        var maxPrecio = Math.max(...cierres.filter(n => n !== null), objetivo);

        if (cierres.length > 0) {{
             minPrecio = Math.min(minPrecio, ...cierres.filter(n => n !== null));
             maxPrecio = Math.max(maxPrecio, ...cierres.filter(n => n !== null));
        }}

        if (minPrecio === maxPrecio) {{
            minPrecio *= 0.95;
            maxPrecio *= 1.05;
        }} else {{
            minPrecio *= 0.98;
            maxPrecio *= 1.02;
        }}

        var ctx = document.getElementById('notasChart').getContext('2d');
        var notasChart = new Chart(ctx, {{
            type: 'line', // CAMBIADO A LINE
            data: {{
                labels: labels,
                datasets: [
                    {{
                        label: 'Nota Técnica (Desplazada {OFFSET_DIAS} días)',
                        data: notas, // Usamos las notas ya desplazadas
                        backgroundColor: 'rgba(0, 128, 255, 0.2)', // Área bajo la línea
                        borderColor: 'rgba(0, 128, 255, 1)',
                        borderWidth: 2,
                        type: 'line',
                        fill: false, // Ahora es una línea
                        yAxisID: 'y'
                    }},
                    {{
                        label: 'Precio de Cierre',
                        data: cierresProyectados, // Datos completos (reales + proyectados)
                        type: 'line',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y1',
                        segment: {{ // Para la línea punteada futura
                            borderDash: ctx => {{(ctx.p1DataIndex >= {30 - 1}) ? [6, 6] : undefined}}
                        }}
                    }}
                ]
            }},
            options: {{
                plugins: {{
                    annotation: {{
                        annotations: {{
                            soporte1: {{
                                type: 'line',
                                yMin: {data['SOPORTE_1']},
                                yMax: {data['SOPORTE_1']},
                                yScaleID: 'y1',
                                borderColor: 'rgba(0, 255, 0, 0.8)',
                                borderWidth: 2,
                                label: {{
                                    enabled: true,
                                    content: 'Soporte 1 ({data["SOPORTE_1"]}€)',
                                    position: 'end'
                                }}
                            }},
                            soporte2: {{
                                type: 'line',
                                yMin: {data['SOPORTE_2']},
                                yMax: {data['SOPORTE_2']},
                                yScaleID: 'y1',
                                borderColor: 'rgba(0, 200, 0, 0.6)',
                                borderWidth: 2,
                                label: {{
                                    enabled: true,
                                    content: 'Soporte 2 ({data["SOPORTE_2"]}€)',
                                    position: 'end'
                                }}
                            }},
                            soporte3: {{
                                type: 'line',
                                yMin: {data['SOPORTE_3']},
                                yMax: {data['SOPORTE_3']},
                                yScaleID: 'y1',
                                borderColor: 'rgba(0, 150, 0, 0.6)',
                                borderWidth: 2,
                                label: {{
                                    enabled: true,
                                    content: 'Soporte 3 ({data["SOPORTE_3']}€)',
                                    position: 'end'
                                }}
                            }},
                            resistencia: {{
                                type: 'line',
                                yMin: {data['RESISTENCIA']},
                                yMax: {data['RESISTENCIA']},
                                yScaleID: 'y1',
                                borderColor: 'rgba(255, 99, 132, 1)',
                                borderWidth: 2,
                                label: {{
                                    enabled: true,
                                    content: 'Resistencia ({data["RESISTENCIA"]}€)',
                                    position: 'end'
                                }}
                            }},
                            objetivo: {{
                                type: 'line',
                                yMin: {data['PRECIO_OBJETIVO_COMPRA']},
                                yMax: {data['PRECIO_OBJETIVO_COMPRA']},
                                yScaleID: 'y1',
                                borderColor: 'rgba(255, 206, 86, 1)',
                                borderWidth: 2,
                                borderDash: [6, 6],
                                label: {{
                                    enabled: true,
                                    content: 'Objetivo ({data["PRECIO_OBJETIVO_COMPRA"]}€)',
                                    position: 'end',
                                    backgroundColor: 'rgba(255, 206, 86, 0.8)'
                                }}
                            }}
                        }}
                    }},
                    legend: {{
                        display: true
                    }},
                    tooltip: {{
                        mode: 'index',
                        intersect: false,
                        callbacks: {{
                            label: function(context) {{
                                let label = context.dataset.label || '';
                                if (label) {{
                                    label += ': ';
                                }}
                                // Si es un punto futuro proyectado, no mostrar valor para la nota
                                if (context.dataset.label.includes('Nota Técnica') && context.parsed.y === null) {{
                                    return label + 'N/A (Proyectado)';
                                }}
                                label += context.parsed.y !== null ? context.parsed.y.toFixed(2) : 'N/A';
                                return label;
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 10,
                        title: {{
                            display: true,
                            text: 'Nota Técnica (0-10)'
                        }}
                    }},
                    y1: {{
                        position: 'right',
                        beginAtZero: false,
                        title: {{
                            display: true,
                            text: 'Precio de Cierre (€)'
                        }},
                        grid: {{
                            drawOnChartArea: false
                        }},
                        ticks: {{
                            padding: 5
                        }},
                        suggestedMin: minPrecio,
                        suggestedMax: maxPrecio
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Historial y Proyección de {PROYECCION_FUTURA_DIAS} Días (ibexia.es)'
                        }}
                    }}
                }},
                responsive: true,
                maintainAspectRatio: false
            }}
        }});
    }});
</script>
<br/>
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
    <tr>
        <td style="padding: 8px;">Días Estimados para Acción</td>
        <td style="padding: 8px;"><strong>{data['DIAS_ESTIMADOS_ACCION']}</strong></td>
    </tr>
</table>
<br/>
"""

    # Lógica Condicional para la Sección "Ganaríamos"
    ganancia_seccion_contenido = ""
    inversion_base = data.get('inversion_base', 10000.0)
    comision_por_operacion_porcentual = data.get('comision_por_operacion_porcentual', 0.001)

    mejor_compra = data.get('mejor_compra_historica')
    mejor_venta = data.get('mejor_venta_historica')
    mejor_punto_giro_compra = data.get('mejor_punto_giro_compra_historica')
    mejor_punto_giro_venta = data.get('mejor_punto_giro_venta_historica')

    if mejor_compra:
        idx, inicio, maximo, pct, ganancia_neta = mejor_compra
        fecha = (datetime.today() - timedelta(days=29 - idx)).strftime("%d/%m")
        ganancia_seccion_contenido += f"<p>En nuestra mejor recomendación de <strong>compra</strong>, el día {fecha}, con el precio a <strong>{inicio:.2f}€</strong>, el valor alcanzó un máximo de <strong>{maximo:.2f}€</strong>. Con una inversión de <strong>{inversion_base:,.2f}€</strong>, esto habría generado una ganancia neta estimada de <strong>{ganancia_neta:,.2f}€</strong> (tras descontar las comisiones del {comision_por_operacion_porcentual*100:.1f}% por operación). Este acierto demuestra la potencia de nuestras señales para capturar el potencial alcista del mercado.</p>"

    if mejor_venta:
        idx, inicio, minimo, pct, perdida_evitada_neta = mejor_venta
        fecha = (datetime.today() - timedelta(days=29 - idx)).strftime("%d/%m")
        ganancia_seccion_contenido += f"<p>En cuanto a nuestras señales de <strong>venta</strong>, la más destacada ocurrió el día {fecha}, con un precio de <strong>{inicio:.2f}€</strong>. Si hubiéramos invertido <strong>{inversion_base:,.2f}€</strong> y seguido nuestra señal para evitar la caída hasta <strong>{minimo:.2f}€</strong>, habríamos evitado una pérdida neta estimada de <strong>{abs(perdida_evitada_neta):,.2f}€</strong> (tras descontar comisiones). Esto subraya la capacidad de nuestros análisis para proteger tu capital en momentos de debilidad del mercado.</p>"

    if mejor_punto_giro_compra:
        idx, inicio, maximo, pct, ganancia_neta = mejor_punto_giro_compra
        fecha = (datetime.today() - timedelta(days=29 - idx)).strftime("%d/%m")
        ganancia_seccion_contenido += f"<p>También, identificamos un punto de inflexión alcista el día {fecha}, cuando el precio era de <strong>{inicio:.2f}€</strong>. Si hubiéramos aprovechado este giro de la nota técnica, el valor alcanzó <strong>{maximo:.2f}€</strong>, lo que podría haber generado una ganancia neta estimada de <strong>{ganancia_neta:,.2f}€</strong> con una inversión de {inversion_base:,.2f}€.</p>"

    if mejor_punto_giro_venta:
        idx, inicio, minimo, pct, perdida_evitada_neta = mejor_punto_giro_venta
        fecha = (datetime.today() - timedelta(days=29 - idx)).strftime("%d/%m")
        ganancia_seccion_contenido += f"<p>De manera similar, un punto de inflexión bajista se observó el día {fecha} con el precio a <strong>{inicio:.2f}€</strong>. Al anticipar esta caída hasta <strong>{minimo:.2f}€</strong>, se habría podido evitar una pérdida neta estimada de <strong>{abs(perdida_evitada_neta):,.2f}€</strong> con una inversión de {inversion_base:,.2f}€.</p>"

    # Solo añadir la sección si hay contenido de ganancia/pérdida
    if ganancia_seccion_contenido:
        chart_html += f"""
        <div style="margin-top:20px;">
            <h2>Impacto de Nuestras Señales Históricas (Inversión Base: {inversion_base:,.2f}€)</h2>
            {ganancia_seccion_contenido}
        </div>
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
- Resultados financieros recientes: {data['INGRESOS']}, {data['EBITDA']}, {data['BENEFICIOS']}
- Nivel de deuda y flujo de caja: {data['DEUDA']}, {data['FLUJO_CAJA']}
- Información estratégica: {data['EXPANSION_PLANES']}, {data['ACUERDOS']}
- Sentimiento del mercado: {data['SENTIMIENTO_ANALISTAS']}, {data['TENDENCIA_SOCIAL']}
- ativa sectorial: {data['EMPRESAS_SIMILARES']}
- Riesgos y oportunidades: {data['RIESGOS_OPORTUNIDADES']}
- Tendencia de la nota: {data['TENDENCIA_NOTA']}
- Días estimados para acción: {data['DIAS_ESTIMADOS_ACCION']}

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

<h2>Visión a Largo Plazo y Fundamentales</h2>

<p>En el último ejercicio, los ingresos declarados fueron de <strong>{formatear_numero(data['INGRESOS'])}</strong>, el EBITDA alcanzó <strong>{formatear_numero(data['EBITDA'])}</strong>, y los beneficios netos se situaron en torno a <strong>{formatear_numero(data['BENEFICIOS'])}</strong>.
En cuanto a su posición financiera, la deuda asciende a <strong>{formatear_numero(data['DEUDA'])}</strong>, y el flujo de caja operativo es de <strong>{formatear_numero(data['FLUJO_CAJA'])}</strong>.</p>

{tabla_resumen}

<h2>Estrategia de Inversión y Gestión de Riesgos</h2>
<p>Mi evaluación profesional indica que la tendencia actual de nuestra nota técnica es **{data['TENDENCIA_NOTA']}**, lo que, en combinación con el resto de nuestros indicadores, se alinea con una recomendación de <strong>{data['RECOMENDACION']}</strong>.
Esto sugiere {('un rebote inminente, dado que los indicadores muestran una sobreventa extrema, lo que significa que la acción ha sido \'castigada\' en exceso y hay una alta probabilidad de que los compradores tomen el control, impulsando el precio alza. Esta situación de sobreventa, sumada al impulso alcista subyacente, nos sugiere que estamos ante el inicio de un rebote significativo.' if "Compra" in data['RECOMENDACION'] and data['TENDENCIA_NOTA'] == 'mejorando' else '')}
{('una potencial continuación bajista, con los indicadores técnicos mostrando una sobrecompra significativa o una pérdida de impulso alcista. Esto sugiere que la acción podría experimentar una corrección. Es un momento para la cautela y la vigilancia de los niveles de soporte.' if "Venta" in data['RECOMENDACION'] and data['TENDENCIA_NOTA'] == 'empeorando' else '')}
{('una fase de consolidación o lateralidad, donde los indicadores técnicos no muestran una dirección clara. Es un momento para esperar la confirmación de una nueva tendencia antes de tomar decisiones.' if "Neutral" in data['RECOMENDACION'] else '')}
{f" Estimamos que el punto de acción óptimo se encuentra en aproximadamente **{data['DIAS_ESTIMADOS_ACCION']}**." if "No disponible" not in data['DIAS_ESTIMADOS_ACCION'] and "Ya en zona" not in data['DIAS_ESTIMADOS_ACCION'] and "inminente" not in data['DIAS_ESTIMADOS_ACCION'] else ("La nota ya se encuentra en una zona de acción clara, lo que sugiere una oportunidad {('de compra' if 'Compra' in data['RECOMENDACION'] else 'de venta')} inmediata, y por tanto, no se estima un plazo de días adicional, o la acción es inminente." if "Ya en zona" in data['DIAS_ESTIMADOS_ACCION'] or "inminente" in data['DIAS_ESTIMADOS_ACCION'] else "")}</p>

<p>{volumen_analisis_text}</p>


<h2>Predicción a Largo Plazo y Conclusión</h2>
<p>Considerando la nota técnica actual de <strong>{data['NOTA_EMPRESA']}</strong> y la dirección de su tendencia (<strong>{data['TENDENCIA_NOTA']}</strong>), mi pronóstico a largo plazo para <strong>{data['NOMBRE_EMPRESA']}</strong> es {("optimista. La empresa muestra una base sólida para un crecimiento sostenido, respaldada por indicadores técnicos favorables y una gestión financiera prudente. Si los planes de expansión y los acuerdos estratégicos se materializan, podríamos ver una apreciación significativa del valor en el futuro." if data['NOTA_EMPRESA'] >= 7 else "")}
{("cauteloso. Si bien no hay señales inmediatas de alarma, la nota técnica sugiere que la empresa podría enfrentar desafíos en el corto y mediano plazo. Es crucial monitorear de cerca los riesgos identificados y cualquier cambio en el sentimiento del mercado para ajustar la estrategia." if data['NOTA_EMPRESA'] < 7 and data['NOTA_EMPRESA'] >=4 else "")}
{("pesimista. La debilidad técnica persistente y los factores de riesgo sugieren que la empresa podría experimentar una presión bajista considerable. Se recomienda extrema cautela y considerar estrategias de protección de capital." if data['NOTA_EMPRESA'] < 4 else "")}.</p>

<h2>Conclusión General y Descargo de Responsabilidad</h2>
<p>Para cerrar este análisis de <strong>{data['NOMBRE_EMPRESA']}</strong>, considero que las claras señales técnicas que apuntan a {('un rebote desde una zona de sobreventa extrema, configurando una oportunidad atractiva' if data['NOTA_EMPRESA'] >= 7 else 'una posible corrección, lo que exige cautela')}, junto con {f"sus sólidos ingresos de <strong>{formatear_numero(data['INGRESOS'])}</strong> y un flujo de caja positivo de <strong>{formatear_numero(data['FLUJO_CAJA'])}</strong>," if data['INGRESOS'] != 'N/A' else "aspectos fundamentales que requieren mayor claridad,"} hacen de esta empresa un activo para mantener bajo estricta vigilancia. </p>
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
        
        # LLAMADAS A LAS NUEVAS FUNCIONES DE RECOMENDACIÓN Y ANÁLISIS HISTÓRICO
        data = generar_recomendacion_avanzada(data)
        data = analizar_oportunidades_historicas(data)

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
                    server_suggested_delay = 0 # Inicializamos a 0
                    try:
                        match = re.search(r"retry_delay \{\s*seconds: (\d+)", str(e))
                        if match:
                            server_suggested_delay = int(match.group(1))
                    except:
                        pass

                    # Calcula el retraso actual basado en la retirada exponencial o el sugerido por el servidor
                    current_delay = max(initial_delay * (2 ** retries), server_suggested_delay + 1)

                    # Añade jitter (aleatoriedad) para evitar colisiones con otras solicitudes
                    jitter = random.uniform(0.5, 1.5) # Factor aleatorio entre 0.5 y 1.5
                    delay_with_jitter = current_delay * jitter

                    print(f"❌ Cuota de Gemini excedida al generar contenido. Reintentando en {delay_with_jitter:.2f} segundos... (Intento {retries + 1}/{max_retries})")
                    time.sleep(delay_with_jitter) # Usa el retraso con jitter
                    retries += 1
                    # La variable 'delay' ya no se necesita mantener persistente ni multiplicar
                    # porque 'current_delay' se calcula de nuevo en cada intento
                else:
                    print(f"❌ Error al generar contenido con Gemini (no de cuota): {e}")
                    break
        else:  
            print(f"❌ Falló la generación de contenido para {ticker} después de {max_retries} reintentos.")
            
        # --- PAUSA DE 3 MINUTO DESPUÉS DE CADA TICKER ---
        print(f"⏳ Esperando 180 segundos antes de procesar el siguiente ticker...")
        time.sleep(180) # Pausa de 180 segundos entre cada ticker


def main():
    # Define el ticker que quieres analizar
    ticker_deseado = "MSFT"  # <-- ¡CAMBIA "BBVA.MC" por el Ticker que quieras analizar!
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
