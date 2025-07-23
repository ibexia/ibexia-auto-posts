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

# Definición de la cantidad de días para la proyección futura
PROYECCION_FUTURA_DIAS = 5

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

    # Manejo de división por cero y clipado para SMI
    # Se añade un pequeño epsilon al denominador para mayor robustez
    epsilon = 1e-9
    # np.where permite definir el valor cuando la condición es True/False
    smi_raw = np.where(
        (avgdiff / 2 + epsilon) != 0, # Si el denominador (más epsilon) no es cero
        (avgrel / (avgdiff / 2 + epsilon)) * 100, # Realiza el cálculo
        0.0 # Si es cero, asigna 0.0
    )
    smi_raw = np.clip(smi_raw, -100, 100) # Asegurar que esté entre -100 y 100

    smi_smoothed = pd.Series(smi_raw, index=df.index).rolling(window=smooth_period).mean()
    smi_signal = smi_smoothed.ewm(span=ema_signal_len, adjust=False).mean()

    df['SMI'] = smi_smoothed # Asignamos directamente la señal SMI suavizada al DataFrame
    return df

def obtener_datos_yfinance(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Ampliar periodo si es necesario para el retraso y ciones
        hist_extended = stock.history(period="90d", interval="1d")
        hist_extended = calculate_smi_tv(hist_extended)

        # Usar un historial más corto para obtener la tendencia de la nota actual (últimos 30 días)
        hist = stock.history(period="30d", interval="1d")
        hist = calculate_smi_tv(hist)

        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Ampliar periodo si es necesario para el retraso y ciones
        hist_extended = stock.history(period="90d", interval="1d")
        hist_extended = calculate_smi_tv(hist_extended)

        # Usar un historial más corto para obtener la tendencia de la nota actual (últimos 30 días)
        hist = stock.history(period="30d", interval="1d")
        hist = calculate_smi_tv(hist)

        # Obtener el precio actual y volumen
        current_price = round(info["currentPrice"], 2)
        current_volume = info.get("volume", "N/A")

        # Get last valid SMI signal
        smi_actual_series = hist['SMI'].dropna() # Obtener las señales SMI sin NaN

        if not smi_actual_series.empty:
            smi_actual = round(smi_actual_series.iloc[-1], 2)
        else:
            # Si no hay datos SMI válidos, asignar un valor por defecto
            print(f"⚠️ Advertencia: No hay datos de SMI válidos para {ticker}. Asignando SMI neutral.")
            smi_actual = 0  # Un valor por defecto para smi_actual


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
        # --- NUEVA LÓGICA DE PRECIO OBJETIVO BASADA EN PENDIENTE DEL SMI ---
        # Asegúrate de tener historial completo para calcular SMI reciente
        smi_history_full = hist_extended['SMI'].dropna()

        # Calcular pendiente de los últimos 5 días del SMI
        smi_ultimos_5 = smi_history_full.tail(5).dropna()
        if len(smi_ultimos_5) >= 2:
            x = np.arange(len(smi_ultimos_5))
            y = smi_ultimos_5.values
            pendiente_smi, _ = np.polyfit(x, y, 1)
        else:
            pendiente_smi = 0

        # Precio objetivo basado en dirección del SMI
        if pendiente_smi > 0.1:
            # Tendencia alcista → subir hasta resistencia más próxima
            precio_objetivo = next((r for r in sorted(resistencias) if r > current_price), current_price * 1.05)
        elif pendiente_smi < -0.1:
            # Tendencia bajista → bajar hasta soporte más próximo
            precio_objetivo = next((s for s in sorted(soportes, reverse=True) if s < current_price), current_price * 0.95)
        else:
            # SMI sin dirección clara → mantener precio actual
            precio_objetivo = current_price

        precio_objetivo = round(precio_objetivo, 2)
        # --- FIN NUEVA LÓGICA ---
        # --- FIN DE LA LÓGICA MEJORADA PARA EL PRECIO OBJETIVO ---

        # Precio objetivo de compra (ejemplo simple, puedes refinarlo)
        # Este 'precio_objetivo_compra' es diferente al 'precio_objetivo' general
        precio_objetivo_compra = round(current_price * 0.98, 2) # Un 2% por debajo del precio actual como ejemplo

        

        # Inicializar recomendacion y condicion_rsi como temporales, se recalcularán después
        recomendacion = "Pendiente de análisis avanzado"
        condicion_rsi = "Pendiente"


        # --- NUEVAS VARIABLES Y LÓGICA PARA GRÁFICOS Y PROYECCIÓN ---
        # Definiciones para los gráficos
        OFFSET_DIAS = 0
        
        # Aseguramos tener suficientes datos para el historial y la proyección
        smi_history_full = hist_extended['SMI'].dropna()
        cierres_history_full = hist_extended['Close'].dropna()

        # Calcula el volumen promedio de los últimos 30 días
        volumen_promedio_30d = hist_extended['Volume'].tail(30).mean()

        # Fechas reales de cotización para los últimos 30 días
        fechas_historial = hist_extended['Close'].dropna().tail(30).index.strftime("%d/%m").tolist()
        ultima_fecha_historial = hist_extended['Close'].dropna().tail(1).index[0]
        # Creamos las fechas de proyección con la variable global PROYECCION_FUTURA_DIAS
        fechas_proyeccion = [(ultima_fecha_historial + timedelta(days=i)).strftime("%d/%m (fut.)") for i in range(1, PROYECCION_FUTURA_DIAS + 1)]
        
        # SMI para los 30 días del gráfico
        smi_historico_para_grafico = []
        if len(smi_history_full) >= 30:
            smi_historico_para_grafico = smi_history_full.tail(30).tolist()
        elif smi_history_full.empty:
            smi_historico_para_grafico = [0.0] * 30
        else:
            first_smi_val = smi_history_full.iloc[0]
            smi_historico_para_grafico = [first_smi_val] * (30 - len(smi_history_full)) + smi_history_full.tolist()

        # Precios para el gráfico: 30 días reales
        precios_reales_para_grafico = []
        if len(cierres_history_full) >= (30 + OFFSET_DIAS):
            precios_reales_para_grafico = cierres_history_full.tail(30).tolist()
        elif len(cierres_history_full) > OFFSET_DIAS:
            temp_prices = cierres_history_full.iloc[OFFSET_DIAS:].tolist()
            first_price_val = temp_prices[0] if temp_prices else current_price
            precios_reales_para_grafico = [first_price_val] * (30 - len(temp_prices)) + temp_prices
        else:
             precios_reales_para_grafico = [current_price] * 30
            
        smi_history_last_30 = hist['SMI'].dropna().tail(30).tolist()
        
        # --- NUEVA LÓGICA DE PROYECCIÓN DE PRECIOS BASADA EN TENDENCIA DEL SMI ---
        precios_proyectados = []
        ultimo_precio_conocido = precios_reales_para_grafico[-1] if precios_reales_para_grafico else current_price
        smi_history_full = hist_extended['SMI'].dropna()
        smi_proyectado = smi_history_full.tail(1).iloc[0] if not smi_history_full.empty else 0
        TOLERANCIA_CRUCE = 0.005

        # Proyectamos el SMI para la zona del futuro
        smi_proyectado_futuro = []
        for i in range(PROYECCION_FUTURA_DIAS):
            smi_step = pendiente_smi  # La base del movimiento es la pendiente actual

            # Ajustamos el paso si el SMI se acerca a las zonas de sobrecompra/sobreventa
            if smi_proyectado > 70:  # Si está en sobrecompra fuerte (>70)
                smi_step *= (100 - smi_proyectado) / 30  # Disminuye el paso a medida que se acerca a 100
                if smi_step > 0:
                    smi_step = -abs(smi_step)  # Si sigue subiendo, fuerza el descenso
            elif smi_proyectado < -70:  # Si está en sobreventa fuerte (<-70)
                smi_step *= (smi_proyectado + 100) / 30  # Disminuye el paso a medida que se acerca a -100
                if smi_step < 0:
                    smi_step = abs(smi_step)  # Si sigue bajando, fuerza el ascenso
            elif smi_proyectado > 40:  # En sobrecompra (40-70)
                if smi_step > 0:
                    smi_step *= 0.5  # Suaviza el ascenso
            elif smi_proyectado < -40:  # En sobreventa (-40 - -70)
                if smi_step < 0:
                    smi_step *= 0.5  # Suaviza el descenso

            # Añadimos una pequeña aleatoriedad para evitar una línea demasiado recta
            smi_step += random.uniform(-0.5, 0.5)

            smi_proyectado += smi_step
            smi_proyectado = np.clip(smi_proyectado, -100, 100)  # Mantenemos el SMI dentro de [-100, 100]
            smi_proyectado_futuro.append(smi_proyectado)

        # Usar los SMI proyectados para calcular los precios proyectados
        for smi_futuro in smi_proyectado_futuro:
            factor_cambio = (smi_futuro / 100) * (-0.01)
            siguiente_precio = ultimo_precio_conocido * (1 + factor_cambio)

            if siguiente_precio > ultimo_precio_conocido:
                for r in sorted(resistencias):
                    if ultimo_precio_conocido < r < siguiente_precio:
                        distancia_relativa = abs(siguiente_precio - r) / r
                        if distancia_relativa > TOLERANCIA_CRUCE:
                            siguiente_precio = round(r * (1 - 0.001), 2)
                        break
            elif siguiente_precio < ultimo_precio_conocido:
                for s in sorted(soportes, reverse=True):
                    if siguiente_precio < s < ultimo_precio_conocido:
                        distancia_relativa = abs(siguiente_precio - s) / s
                        if distancia_relativa > TOLERANCIA_CRUCE:
                            siguiente_precio = round(s * (1 + 0.001), 2)
                        break

            siguiente_precio = round(siguiente_precio, 2)
            precios_proyectados.append(siguiente_precio)
            ultimo_precio_conocido = siguiente_precio
        # --- FIN DE LA NUEVA LÓGICA ---
     
        # Unir precios reales y proyectados
        cierres_para_grafico_total = precios_reales_para_grafico + precios_proyectados

        tendencia_smi = "No disponible"
        
        if len(smi_history_last_30) >= 2:
            x = np.arange(len(smi_history_last_30))
            y = np.array(smi_history_last_30)
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

        datos = {
            "TICKER": ticker,
            "NOMBRE_EMPRESA": info.get("longName", ticker),
            "PRECIO_ACTUAL": current_price,
            "VOLUMEN": current_volume,
            "VOLUMEN_MEDIO": round(volumen_promedio_30d, 2) if not pd.isna(volumen_promedio_30d) else "N/A",
            "SOPORTE_1": soporte_1,
            "SOPORTE_2": soporte_2,
            "SOPORTE_3": soporte_3,
            "RESISTENCIA": resistencia_1,
            "CONDICION_RSI": condicion_rsi,
            "RECOMENDACION": recomendacion,
            "SMI": smi_actual,
            "PRECIO_OBJETIVO_COMPRA": precio_objetivo_compra,
            "TENDENCIA_SMI": tendencia_smi, # Renombrado de TENDENCIA_NOTA
            "CIERRES_30_DIAS": hist['Close'].dropna().tail(30).tolist(),
            "SMI_HISTORICO_PARA_GRAFICO": smi_historico_para_grafico, # Renombrado
            "CIERRES_PARA_GRAFICO_TOTAL": cierres_para_grafico_total,
            "PRECIOS_PROYECTADOS": precios_proyectados, # ¡NUEVO! Guardamos la lista aquí
            "OFFSET_DIAS_GRAFICO": OFFSET_DIAS,
            "RESISTENCIA_1": resistencia_1,
            "RESISTENCIA_2": resistencia_2,
            "RESISTENCIA_3": resistencia_3,
            "PRECIO_OBJETIVO": precio_objetivo,
            "FECHAS_HISTORIAL": fechas_historial,
            "FECHAS_PROYECCION": fechas_proyeccion,
            "SMI_PROYECTADO_FUTURO": smi_proyectado_futuro,
            "PROYECCION_FUTURA_DIAS_GRAFICO": PROYECCION_FUTURA_DIAS
        }

        return datos

    except Exception as e:
        print(f"❌ Error al obtener datos de {ticker}: {e}. Saltando a la siguiente empresa...")
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
    # Generación de la recomendación de volumen
    volumen_analisis_text = ""
    if data['VOLUMEN'] != "N/A":
        volumen_actual = data['VOLUMEN']
        try:
            ticker_obj = yf.Ticker(data['TICKER'])
            hist_vol = ticker_obj.history(period="90d")
            if not hist_vol.empty and 'Volume' in hist_vol.columns:
                volumen_promedio_30d = hist_vol['Volume'].tail(30).mean()
                if volumen_promedio_30d > 0:
                    cambio_porcentual_volumen = ((volumen_actual - volumen_promedio_30d) / volumen_promedio_30d) * 100
                    if cambio_porcentual_volumen > 50:
                        volumen_analisis_text = f"El volumen negociado de <strong>{volumen_actual:,.0f} acciones</strong> es notablemente superior al promedio reciente, indicando un fuerte interés del mercado y validando la actual tendencia del SMI ({data['TENDENCIA_SMI']})."
                    elif cambio_porcentual_volumen < -30:
                        volumen_analisis_text = f"El volumen de <strong>{volumen_actual:,.0f} acciones</strong> es inferior a lo habitual, lo que podría sugerir cautela en la actual tendencia. Una confirmación de la señal del SMI ({data['TENDENCIA_SMI']}) requeriría un aumento en la participación del mercado."
                    else:
                        volumen_analisis_text = f"El volumen de <strong>{volumen_actual:,.0f} acciones</strong> se mantiene en línea con el promedio. Es un volumen adecuado, pero no excepcional, para confirmar de manera contundente la señal del SMI ({data['TENDENCIA_SMI']})."
                else:
                    volumen_analisis_text = f"El volumen de <strong>{volumen_actual:,.0f} acciones</strong> es importante para confirmar cualquier movimiento. "
            else:
                volumen_analisis_text = f"El volumen de <strong>{volumen_actual:,.0f} acciones</strong> es importante para confirmar cualquier movimiento. "
        except Exception as e:
            volumen_analisis_text = f"El volumen de <strong>{volumen_actual:,.0f} acciones</strong> es importante para confirmar cualquier movimiento. No fue posible comparar con el volumen promedio: {e}"
    else:
        volumen_analisis_text = "El volumen de negociación no está disponible en este momento."

    titulo_post = f"Análisis Técnico: {data['NOMBRE_EMPRESA']} ({data['TICKER']}) - Recomendación de {data['RECOMENDACION']}"

    # Recuperamos los datos del diccionario 'data' que ahora contiene las listas de precios
    smi_historico_para_grafico = data.get('SMI_HISTORICO_PARA_GRAFICO', [])
    smi_proyectado_futuro = data.get('SMI_PROYECTADO_FUTURO', [])
    precios_reales_para_grafico = data.get('CIERRES_30_DIAS', [])
    precios_proyectados = data.get('PRECIOS_PROYECTADOS', [])

    # Preparamos los datos para los gráficos de SMI
    smi_historico_con_nulos_futuros = smi_historico_para_grafico + [None] * len(smi_proyectado_futuro)
    smi_proyectado_con_nulos_historicos = [None] * len(smi_historico_para_grafico) + smi_proyectado_futuro

    # Preparamos los datos para los gráficos de precios
    precios_historicos_con_nulos_futuros = precios_reales_para_grafico + [None] * len(precios_proyectados)
    precios_proyectados_con_nulos_historicos = [None] * len(precios_reales_para_grafico) + precios_proyectados

    labels_historial = data.get('FECHAS_HISTORIAL', [])
    labels_proyeccion = data.get('FECHAS_PROYECCION', [])
    labels_total = labels_historial + labels_proyeccion

    chart_html = ""
    if smi_historico_para_grafico and cierres_para_grafico_total:
        labels_historial = data.get("FECHAS_HISTORIAL", [])
        labels_proyeccion = data.get("FECHAS_PROYECCION", [])
        labels_total = labels_historial + labels_proyeccion

        precios_reales_grafico = cierres_para_grafico_total[:30]
        data_proyectada = [None] * (len(labels_historial) - 1) + [precios_reales_grafico[-1]] + cierres_para_grafico_total[len(labels_historial):]

        smi_desplazados_para_grafico = smi_historico_para_grafico + data.get('SMI_PROYECTADO_FUTURO', [])


        chart_html += f"""
        <h2>Evolución del Stochastic Momentum Index (SMI) y Precio</h2>
        <p>Para ofrecer una perspectiva visual clara de la evolución del SMI...
        Es importante recordar que el SMI de hoy (D) se alinea con el precio de D+4,
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
                            label: 'SMI Histórico',
                            data: {json.dumps(smi_historico_con_nulos_futuros)},
                            borderColor: 'rgba(54, 162, 235, 1)',
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            yAxisID: 'y',
                            tension: 0.1,
                            fill: false,
                            pointRadius: 0
                        }},
                        {{
                            label: 'SMI Proyectado',
                            data: {json.dumps(smi_proyectado_con_nulos_historicos)},
                            borderColor: 'rgba(54, 162, 235, 1)',
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            yAxisID: 'y',
                            tension: 0.1,
                            fill: false,
                            borderDash: [5, 5],
                            pointRadius: 0
                        }},
                        {{
                            label: 'Precio',
                            data: {json.dumps(precios_historicos_con_nulos_futuros)},
                            borderColor: 'rgba(75, 192, 192, 1)',
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            yAxisID: 'y1',
                            tension: 0.1,
                            fill: false,
                        }},
                        {{
                            label: 'Precio Proyectado',
                            data: {json.dumps(precios_proyectados_con_nulos_historicos)},
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
                                // Zonas de color en el eje Y del SMI (-100 a 100)
                                zonaSobrecompra: {{
                                    type: 'box',
                                    yScaleID: 'y',
                                    yMin: 40,
                                    yMax: 100,
                                    backgroundColor: 'rgba(255, 0, 0, 0.1)', // Rojo claro para sobrecompra
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
                                    backgroundColor: 'rgba(255, 255, 0, 0.1)', // Amarillo claro para neutral
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
                                    backgroundColor: 'rgba(0, 255, 0, 0.1)', // Verde claro para sobreventa
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
                                    pointStyle: {'"triangle"' if 'Compra' in data['RECOMENDACION'] else ('"triangle"' if 'Venta' in data['RECOMENDACION'] else '"circle"')},
                                    rotation: {{0 if 'Compra' in data['RECOMENDACION'] else (180 if 'Venta' in data['RECOMENDACION'] else 0)}},
                                    backgroundColor: {'"rgba(0, 200, 0, 0.8)"' if 'Compra' in data['RECOMENDACION'] else ('"rgba(200, 0, 0, 0.8)"' if 'Venta' in data['RECOMENDACION'] else '"rgba(100, 100, 100, 0.8)"')},
                                    borderColor: 'white',
                                    borderWidth: 2,
                                    display: {('true' if 'Compra' in data['RECOMENDACION'] or 'Venta' in data['RECOMENDACION'] else 'false')},
                                    label: {{
                                        content: '{data["RECOMENDACION"]}',
                                        enabled: true,
                                        position: 'top',
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
                                text: 'Stochastic Momentum Index (SMI)'
                            }},
                            min: -100,
                            max: 100
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

    soportes_unicos = []
    temp_soportes = sorted([data['SOPORTE_1'], data['SOPORTE_2'], data['SOPORTE_3']], reverse=True)
    
    if len(temp_soportes) > 0:
        soportes_unicos.append(temp_soportes[0])
        for i in range(1, len(temp_soportes)):
            if abs(temp_soportes[i] - soportes_unicos[-1]) / soportes_unicos[-1] > 0.005:
                soportes_unicos.append(temp_soportes[i])
    
    if not soportes_unicos:
        soportes_unicos.append(0.0)

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
        <td style="padding: 8px;">SMI Actual</td>
        <td style="padding: 8px;"><strong>{data['SMI']:,}</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Precio Objetivo de Compra</td>
        <td style="padding: 8px;"><strong>{data['PRECIO_OBJETIVO_COMPRA']:,}€</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Tendencia del SMI</td>
        <td style="padding: 8px;"><strong>{data['TENDENCIA_SMI']}</strong></td>
    </tr>
</table>
<br/>
"""

    prompt = f"""
Actúa como un trader profesional con amplia experiencia en análisis técnico y mercados financieros. Genera el análisis completo en **formato HTML**, ideal para publicaciones web. Utiliza etiquetas `<h2>` para los títulos de sección y `<p>` para cada párrafo de texto. Redacta en primera persona, con total confianza en tu criterio.

Destaca los datos importantes como precios, valores del SMI, cifras financieras y el nombre de la empresa utilizando la etiqueta `<strong>`. Asegúrate de que no haya asteriscos u otros símbolos de marcado en el texto final, solo HTML válido. Asegurate que todo este escrito en español independientemente del idioma de donde saques los datos.

Genera un análisis técnico completo de aproximadamente 800 palabras sobre la empresa {data['NOMBRE_EMPRESA']}, utilizando los siguientes datos reales extraídos de Yahoo Finance. Presta especial atención al **valor actual del SMI ({data['SMI']})**.

¡ATENCIÓN URGENTE! Para CADA EMPRESA analizada, debes generar el CÓDIGO HTML Y JAVASCRIPT COMPLETO y Único para TODOS sus gráficos solicitados. Bajo ninguna circunstancia debes omitir ningún script, resumir bloques de código o utilizar frases como 'código JavaScript idéntico al ejemplo anterior'. Cada gráfico, para cada empresa, debe tener su script completamente incrustado, funcional e independiente de otros. Asegúrate de que los datos de cada gráfico corresponden SIEMPRE a la empresa que se está analizando en ese momento

**Datos clave:**
- Precio actual: {data['PRECIO_ACTUAL']}
- Volumen del último día completo: {data['VOLUMEN']}
- Soporte 1: {data['SOPORTE_1']}
- Soporte 2: {data['SOPORTE_2']}
- Soporte 3: {data['SOPORTE_3']}
- Resistencia clave: {data['RESISTENCIA']}
- Recomendación general: {data['RECOMENDACION']}
- SMI actual: {data['SMI']}
- Precio objetivo de compra: {data['PRECIO_OBJETIVO_COMPRA']}€
- Tendencia del SMI: {data['TENDENCIA_SMI']}


Importante: si algún dato no está disponible ("N/A", "No disponibles", "No disponible"), no lo menciones ni digas que falta. No expliques que la recomendación proviene de un indicador o dato específico. La recomendación debe presentarse como una conclusión personal basada en tu experiencia y criterio profesional como analista. Al redactar el análisis, haz referencia al **valor actual del SMI ({data['SMI']})** en al menos dos de los párrafos principales (Recomendación General, Análisis a Corto Plazo o Predicción a Largo Plazo) como un factor clave para tu valoración.

---
<h1>{titulo_post}</h1>


<h2>Análisis Inicial y Recomendación</h2>
<p><strong>{data['NOMBRE_EMPRESA']} ({data['TICKER']})</strong> cotiza actualmente a <strong>{data['PRECIO_ACTUAL']:,}€</strong>. Mi precio objetivo de compra se sitúa en <strong>{data['PRECIO_OBJETIVO_COMPRA']:,}€</strong>. El volumen negociado recientemente, alcanzó las <strong>{data['VOLUMEN']:,} acciones</strong>.</p>

<p>Nuestro análisis del Stochastic Momentum Index (SMI) arroja un valor de <strong>{data['SMI']}</strong>. Esta puntuación, combinada con la **{data['TENDENCIA_SMI']}** tendencia del SMI, la proximidad a soportes y resistencias, y el volumen, nos lleva a una recomendación de <strong>{data['RECOMENDACION']}</strong>. Actualmente, el mercado se encuentra en una situación de <strong>{data['CONDICION_RSI']}</strong>. Esto refleja {'una excelente fortaleza técnica y baja volatilidad esperada a corto plazo, lo que indica un bajo riesgo técnico en relación con el potencial de crecimiento.' if "Compra" in data['RECOMENDACION'] and data['SMI'] < -20 else ''}
{'una fortaleza técnica moderada, con un equilibrio entre potencial y riesgo, sugiriendo una oportunidad que requiere seguimiento.' if "Compra Moderada" in data['RECOMENDACION'] or (data['SMI'] >= -20 and data['SMI'] < 0) else ''}
{'una situación técnica neutral, donde el gráfico no muestra un patrón direccional claro, indicando que es un momento para la observación y no para la acción inmediata.' if "Neutral" in data['RECOMENDACION'] else ''}
{'cierta debilidad técnica, con posibles señales de corrección o continuación bajista, mostrando una pérdida de impulso alcista y un aumento de la presión vendedora.' if "Venta" in data['RECOMENDACION'] or (data['SMI'] > 0 and data['SMI'] <= 20) else ''}
{'una debilidad técnica significativa y una posible sobrecompra en el gráfico, lo que sugiere un alto riesgo de corrección.' if "Venta Fuerte" in data['RECOMENDACION'] or data['SMI'] > 20 else ''} </p>
{chart_html}

{tabla_resumen}

<h2>Estrategia de Inversión y Gestión de Riesgos</h2>
<p>Mi evaluación profesional indica que la tendencia actual del SMI es **{data['TENDENCIA_SMI']}**, lo que, en combinación con el resto de nuestros indicadores, se alinea con una recomendación de <strong>{data['RECOMENDACION']}</strong>.</p>
<p><strong>Motivo de la Recomendación:</strong> {data['MOTIVO_RECOMENDACION']}</p>

<p>{volumen_analisis_text}</p>

<h2>Predicción a Largo Plazo y Conclusión</h2>
<p>Considerando el valor actual del SMI de <strong>{data['SMI']}</strong> y la dirección de su tendencia (<strong>{data['TENDENCIA_SMI']}</strong>), mi pronóstico a largo plazo para <strong>{data['NOMBRE_EMPRESA']}</strong> es {("optimista. La empresa muestra una base sólida para un crecimiento sostenido, respaldada por indicadores técnicos favorables y una gestión financiera prudente. Si los planes de expansión y los acuerdos estratégicos se materializan, podríamos ver una apreciación significativa del valor en el futuro." if data['SMI'] < -20 else "")}
{("cauteloso. Si bien no hay señales inmediatas de alarma, el SMI sugiere que la empresa podría enfrentar desafíos en el corto y mediano plazo. Es crucial monitorear de cerca los riesgos identificados y cualquier cambio en el sentimiento del mercado para ajustar la estrategia." if data['SMI'] >= -20 and data['SMI'] <= 20 else "")}
{("pesimista. La debilidad técnica persistente y los factores de riesgo sugieren que la empresa podría experimentar una presión bajista considerable. Se recomienda extrema cautela y considerar estrategias de protección de capital." if data['SMI'] > 20 else "")}.</p>

<h2>Conclusión General y Descargo de Responsabilidad</h2>
<p>Para cerrar este análisis de <strong>{data['NOMBRE_EMPRESA']}</strong>, considero que las claras señales técnicas que apuntan a {('un rebote desde una zona de sobreventa extrema, configurando una oportunidad atractiva' if data['SMI'] < -20 else 'una posible corrección, lo que exige cautela')}, junto con aspectos fundamentales que requieren mayor claridad, hacen de esta empresa un activo para mantener bajo estricta vigilancia. </p>
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
    ticker_deseado = "ADX.MC"

    tickers_for_today = [ticker_deseado]

    if tickers_for_today:
        print(f"\nAnalizando el ticker solicitado: {ticker_deseado}")
        generar_contenido_con_gemini(tickers_for_today)
    else:
        print(f"No se especificó ningún ticker para analizar.")

if __name__ == '__main__':
    main()
