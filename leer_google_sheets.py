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
    
def calcular_ganancias_simuladas(precios, smis, fechas, capital_inicial=10000):
    compras = []
    ventas = []
    posicion_abierta = False
    precio_compra_actual = 0
    ganancia_total = 0  # Acumula las ganancias de las operaciones CERRADAS

    # Calcular la pendiente del SMI para cada punto
    pendientes_smi = [0] * len(smis)
    for i in range(1, len(smis)):
        pendientes_smi[i] = smis[i] - smis[i-1]

    # Iterar sobre los datos históricos para encontrar señales
    for i in range(2, len(smis)):
        print(f"[{fechas[i]}] SMI[i-1]={smis[i-1]:.2f}, SMI[i]={smis[i]:.2f}, pendiente[i]={pendientes_smi[i]:.2f}, pendiente[i-1]={pendientes_smi[i-1]:.2f}")
        # Señal de compra: la pendiente del SMI cambia de negativa a positiva y no está en sobrecompra
        # Se anticipa un día la compra y se añade la condición de sobrecompra
        if i >= 1 and pendientes_smi[i] > 0 and pendientes_smi[i-1] <= 0:
            if not posicion_abierta:
                if smis[i-1] < 40:
                    posicion_abierta = True
                    precio_compra_actual = precios[i-1]
                    compras.append({'fecha': fechas[i-1], 'precio': precio_compra_actual})
                    print(f"✅ COMPRA: {fechas[i-1]} a {precio_compra_actual:.2f}")
                else:
                    print(f"❌ No compra en {fechas[i-1]}: SMI demasiado alto ({smis[i-1]:.2f})")
            else:
                print(f"❌ No compra en {fechas[i-1]}: Ya hay posición abierta")

        # Señal de venta: la pendiente del SMI cambia de positiva a negativa (anticipando un día)
        elif i >= 1 and pendientes_smi[i] < 0 and pendientes_smi[i-1] >= 0:
            if posicion_abierta:
                posicion_abierta = False
                ventas.append({'fecha': fechas[i-1], 'precio': precios[i-1]})
                num_acciones = capital_inicial / precio_compra_actual
                ganancia_total += (precios[i-1] - precio_compra_actual) * num_acciones
                print(f"✅ VENTA: {fechas[i-1]} a {precios[i-1]:.2f}")
            else:
                print(f"❌ No venta en {fechas[i-1]}: No hay posición abierta")

    # --- Generación de la lista HTML de operaciones completadas (SIEMPRE) ---
    operaciones_html = ""
    # Solo iterar sobre las operaciones que se han completado (pares compra-venta)
    num_operaciones_completadas = min(len(compras), len(ventas))

    for i in range(num_operaciones_completadas):
        compra = compras[i]
        venta = ventas[i]
        num_acciones_op = capital_inicial / compra['precio']  # Se asume capital_inicial para cada operación
        ganancia_operacion = (venta['precio'] - compra['precio']) * num_acciones_op

        estado_ganancia = "Ganancia" if ganancia_operacion >= 0 else "Pérdida"

        operaciones_html += f"<li>Compra en {compra['fecha']} a <strong>{compra['precio']:,.2f}€</strong>, Venta en {venta['fecha']} a <strong>{venta['precio']:,.2f}€</strong> - {estado_ganancia}: <strong>{ganancia_operacion:,.2f}€</strong></li>"

    html_resultados = ""

    if not compras:  # No se realizaron compras en el período
        html_resultados = f"""
        <p>No se encontraron señales de compra o venta significativas en el período analizado para Nuestro logaritmo.</p>
        <p>Esto podría deberse a una baja volatilidad, a que el SMI no generó las señales esperadas, o a que el período de análisis es demasiado corto.</p>
        """
    else:  # Hubo al menos una compra
        if posicion_abierta:  # La última posición sigue abierta
            # Calcular la ganancia/pérdida actual de la posición abierta
            ganancia_actual_posicion_abierta = (precios[-1] - precio_compra_actual) * (capital_inicial / precio_compra_actual)
            # La ganancia total incluye las operaciones cerradas y la ganancia (o pérdida) actual de la posición abierta
            ganancia_simulada_total_incl_abierta = ganancia_total + ganancia_actual_posicion_abierta

            html_resultados = f"""
            <p>Se encontraron señales de compra en el período. La última posición abierta no se ha cerrado todavía.</p>
            <p>Si hubieras invertido {capital_inicial:,.2f}€ en cada operación, tu ganancia simulada total (contando operaciones cerradas y la ganancia/pérdida actual de la posición abierta) sería de <strong>{ganancia_simulada_total_incl_abierta:,.2f}€</strong>.</p>
            """
            # Si hay operaciones completadas (ventas realizadas), las mostramos
            if compras and posicion_abierta: # NUEVA LÍNEA AÑADIDA
                html_resultados += f"""
                <p>La última posición comprada fue en {compras[-1]['fecha']} a <strong>{compras[-1]['precio']:,.2f}€</strong> y todavía no se ha vendido.</p>
                """
            if operaciones_html:
                html_resultados += f"""
                <p>A continuación, se detallan las operaciones completadas en el periodo analizado:</p>
                <ul>{operaciones_html}</ul>
                """
        else:  # Todas las posiciones se cerraron
            html_resultados = f"""
            <p>La fiabilidad de nuestro sistema se confirma en el histórico de operaciones. Nuestro logaritmo ha completado un ciclo de compra y venta en el período. Si hubieras invertido {capital_inicial:,.2f}€ en cada operación, tu ganancia simulada total habría sido de <strong>{ganancia_total:,.2f}€</strong>.</p>
            """
            # Siempre mostramos las operaciones detalladas si hay alguna
            if operaciones_html:
                html_resultados += f"""
                <p>A continuación, se detallan las operaciones realizadas en el periodo analizado:</p>
                <ul>{operaciones_html}</ul>
                """

    return html_resultados, compras, ventas

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

        # Obtener datos históricos para el volumen del día anterior completo
        # Solicitamos un periodo más largo (por ejemplo, 5 días) para tener margen
        # y asegurarnos de encontrar un día de trading completo anterior.
        hist_recent = stock.history(period="5d", interval="1d") 
        
        current_price = round(info["currentPrice"], 2) # Este sigue siendo el precio actual

        current_volume = "N/A" # Inicializamos a N/A
        if not hist_recent.empty:
            # Intentamos obtener el volumen del penúltimo día. 
            # Si el último día es el actual (incompleto), el penúltimo será el anterior completo.
            # Si solo hay un día (por ejemplo, fin de semana y solo trae el último viernes), entonces es ese.
            if len(hist_recent) >= 2:
                current_volume = hist_recent['Volume'].iloc[-2] # Penúltima fila
            else: # Solo hay un día de datos (ejecutándose un lunes temprano y solo trae el viernes anterior)
                current_volume = hist_recent['Volume'].iloc[-1] # Última fila (que sería el día anterior completo)

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


        # Nuevas variables para los gráficos con offset y proyección
        OFFSET_DIAS = 0 # El SMI de hoy (D) se alinea con el precio de D+4
        PROYECCION_FUTURA_DIAS = 5 # Días a proyectar después del último precio real

        # Aseguramos tener suficientes datos para el historial, el offset y la proyección
        smi_history_full = hist_extended['SMI'].dropna() # Ahora el SMI final está en 'SMI'
        cierres_history_full = hist_extended['Close'].dropna()

        # Calcula el volumen promedio de los últimos 30 días usando hist_extended
        volumen_promedio_30d = hist_extended['Volume'].tail(30).mean()


        # Fechas reales de cotización para los últimos 30 días
        fechas_historial = hist_extended['Close'].dropna().tail(30).index.strftime("%d/%m").tolist()
        ultima_fecha_historial = hist_extended['Close'].dropna().tail(1).index[0]
        fechas_proyeccion = [(ultima_fecha_historial + timedelta(days=i)).strftime("%d/%m (fut.)") for i in range(1, PROYECCION_FUTURA_DIAS + 1)]
        
        # SMI para los 30 días del gráfico (serán los que se visualicen)
        # Serán los 30 SMI más recientes disponibles
        smi_historico_para_grafico = []
        if len(smi_history_full) >= 30:
            smi_historico_para_grafico = smi_history_full.tail(30).tolist()
        elif smi_history_full.empty:
            smi_historico_para_grafico = [0.0] * 30 # Default neutral if no data
        else:
            # Fill with first available SMI if less than 30
            first_smi_val = smi_history_full.iloc[0]
            smi_historico_para_grafico = [first_smi_val] * (30 - len(smi_history_full)) + smi_history_full.tolist()


        # Precios para el gráfico: 30 días DESPLAZADOS + PROYECCIÓN
        # Necesitamos los últimos (30 + OFFSET_DIAS) precios reales para tener el rango completo
        precios_reales_para_grafico = []
        if len(cierres_history_full) >= (30 + OFFSET_DIAS):
            # Tomamos los 30 precios que se alinearán con los 30 SMI (considerando el offset)
            precios_reales_para_grafico = cierres_history_full.tail(30).tolist()
        elif len(cierres_history_full) > OFFSET_DIAS: # Si tenemos menos de 30 pero más que el offset
            # Tomamos lo que tengamos después del offset y rellenamos al principio
            temp_prices = cierres_history_full.iloc[OFFSET_DIAS:].tolist()
            first_price_val = temp_prices[0] if temp_prices else current_price
            precios_reales_para_grafico = [first_price_val] * (30 - len(temp_prices)) + temp_prices
        else: # Muy pocos datos históricos
             precios_reales_para_grafico = [current_price] * 30 # Default to current price if no historical data
            
        smi_history_last_30 = hist['SMI'].dropna().tail(30).tolist()
        

        # --- NUEVA Lógica: Proyección lineal sin soportes/resistencias ---
        precios_proyectados = []
        ultimo_precio_conocido = precios_reales_para_grafico[-1] if precios_reales_para_grafico else current_price

        # Determinar la dirección de la tendencia y el movimiento diario constante
        # Usamos la pendiente del SMI para determinar si la tendencia es alcista o bajista
        smi_history_full = hist_extended['SMI'].dropna()
        smi_ultimos_5 = smi_history_full.tail(5).dropna()

        pendiente_smi = 0
        if len(smi_ultimos_5) >= 2:
            x = np.arange(len(smi_ultimos_5))
            y = smi_ultimos_5.values
            pendiente_smi, _ = np.polyfit(x, y, 1)

        # Definir un movimiento diario constante, lo suficientemente grande para no redondearse
        # Usamos 1% como un valor base claro y visible
        movimiento_diario = 0
        if pendiente_smi > 0.1 or smi_actual < -40:  # Si SMI sube o está en sobreventa
            movimiento_diario = 0.01  # +1% de subida diaria
        elif pendiente_smi < -0.1 or smi_actual > 40: # Si SMI baja o está en sobrecompra
            movimiento_diario = -0.01 # -1% de bajada diaria

        for _ in range(PROYECCION_FUTURA_DIAS):
            siguiente_precio = ultimo_precio_conocido * (1 + movimiento_diario)
            siguiente_precio = round(siguiente_precio, 2)
            precios_proyectados.append(siguiente_precio)
            ultimo_precio_conocido = siguiente_precio

        # --- Fin de la NUEVA lógica lineal ---



      
     
        # Unir precios reales y proyectados
        cierres_para_grafico_total = precios_reales_para_grafico + precios_proyectados
        precio_proyectado_dia_5 = cierres_para_grafico_total[-1]  # Último precio proyectado a 5 días

        # Guarda los datos para la simulación
        smi_historico_para_simulacion = [round(s, 2) for s in hist_extended['SMI'].dropna().tail(30).tolist()]
        precios_para_simulacion = precios_reales_para_grafico
        fechas_para_simulacion = hist_extended.tail(30).index.strftime("%d/%m/%Y").tolist() # CORREGIDO: ahora se aplica .tail() al DataFrame
        tendencia_ibexia = "No disponible"
        
        if len(smi_history_last_30) >= 2:
            x = np.arange(len(smi_history_last_30))
            y = np.array(smi_history_last_30)
            if np.std(y) > 0.01:
                slope, intercept = np.polyfit(x, y, 1)
            else:
                slope = 0.0

            if slope > 0.1:
                tendencia_ibexia = "mejorando (alcista)"
                recomendacion = "Comprar"
                motivo_recomendacion = f"Nuestro logaritmo muestra una tendencia alcista, lo que sugiere que el precio podría dirigirse hacia la próxima resistencia en {resistencia_1:.2f}€."
            elif slope < -0.1:
                tendencia_ibexia = "empeorando (bajista)"
                recomendacion = "Vender"
                motivo_recomendacion = f"Nuestro logaritmo muestra una tendencia bajista, lo que indica que el precio podría caer hacia el próximo soporte en {soporte_1:.2f}€."
            else:
                tendencia_ibexia = "cambio de tendencia"
                recomendacion = "Atención máxima"
                motivo_recomendacion = "El precio podría girar a a corto plazo."

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
            "tendencia_ibexia": tendencia_ibexia, # Renombrado de TENDENCIA_NOTA
            "CIERRES_30_DIAS": hist['Close'].dropna().tail(30).tolist(),
            "SMI_HISTORICO_PARA_GRAFICO": smi_historico_para_grafico, # Renombrado
            "CIERRES_PARA_GRAFICO_TOTAL": cierres_para_grafico_total,
            "OFFSET_DIAS_GRAFICO": OFFSET_DIAS,
            "RESISTENCIA_1": resistencia_1,
            "RESISTENCIA_2": resistencia_2,
            "RESISTENCIA_3": resistencia_3,
            "PRECIO_OBJETIVO": precio_objetivo,
            "FECHAS_HISTORIAL": fechas_historial,
            "FECHAS_PROYECCION": fechas_proyeccion,
            "PRECIO_PROYECTADO_5DIAS": precio_proyectado_dia_5,
            'PRECIOS_PARA_SIMULACION': precios_para_simulacion,
            'SMI_PARA_SIMULACION': smi_historico_para_simulacion,
            'FECHAS_PARA_SIMULACION': fechas_para_simulacion,
            "PROYECCION_FUTURA_DIAS_GRAFICO": PROYECCION_FUTURA_DIAS
        }
        # --- NUEVA LÓGICA DE RECOMENDACIÓN BASADA EN PROYECCIÓN DE PRECIO ---
        diferencia_precio_porcentual = ((precio_proyectado_dia_5 - current_price) / current_price) * 100 if current_price != 0 else 0

        recomendacion = "sin dirección clara"
        motivo_analisis = "La proyección de precio a 5 días es muy similar al precio actual, lo que indica un mercado en consolidación. Se recomienda cautela."
        
        if diferencia_precio_porcentual > 3:
            recomendacion = "Comprar (Impulso Fuerte)"
            motivo_analisis = f"El precio proyectado a 5 días de {precio_proyectado_dia_5:,.2f}€ es significativamente superior al precio actual, indicando un fuerte impulso alcista."
        elif diferencia_precio_porcentual > 1:
            recomendacion = "Comprar (Impulso Moderado)"
            motivo_analisis = f"El precio proyectado a 5 días de {precio_proyectado_dia_5:,.2f}€ es superior al precio actual, sugiriendo un impulso alcista moderado."
        elif diferencia_precio_porcentual < -3:
            recomendacion = "Vender (Impulso Fuerte)"
            motivo_analisis = f"El precio proyectado a 5 días de {precio_proyectado_dia_5:,.2f}€ es significativamente inferior al precio actual, lo que indica una fuerte presión bajista."
        elif diferencia_precio_porcentual < -1:
            recomendacion = "Vender (Impulso Moderado)"
            motivo_analisis = f"El precio proyectado a 5 días de {precio_proyectado_dia_5:,.2f}€ es inferior al precio actual, sugiriendo un impulso bajista moderado."
        
        # Sobrescribir las variables recomendacion y motivo_analisis
        datos['RECOMENDACION'] = recomendacion
        datos['motivo_analisis'] = motivo_analisis
        # --- FIN NUEVA LÓGICA DE RECOMENDACIÓN ---
        return datos

    except Exception as e:
        print(f"❌ Error al obtener datos de {ticker}: {e}. Saltando a la siguiente empresa...")
        return None




def construir_prompt_formateado(data):
    # Generación de la recomendación de volumen
    volumen_analisis_text = ""
    # Recuperar los datos de compras y ventas simuladas
    compras_simuladas = data.get('COMPRAS_SIMULADAS', [])
    ventas_simuladas = data.get('VENTAS_SIMULADAS', [])
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
                        volumen_analisis_text = f"El volumen negociado de <strong>{volumen_actual:,.0f} acciones</strong> es notablemente superior al promedio reciente, indicando un fuerte interés del mercado y validando la actual tendencia de Nuestro logaritmo ({data['tendencia_ibexia']})."
                    elif cambio_porcentual_volumen < -30:
                        volumen_analisis_text = f"El volumen de <strong>{volumen_actual:,.0f} acciones</strong> es inferior a lo habitual, lo que podría sugerir cautela en la actual tendencia. Una confirmación de la señal de Nuestro logaritmo ({data['tendencia_ibexia']}) requeriría un aumento en la participación del mercado."
                    else:
                        volumen_analisis_text = f"El volumen de <strong>{volumen_actual:,.0f} acciones</strong> se mantiene en línea con el promedio. Es un volumen adecuado, pero no excepcional, para confirmar de manera contundente la señal de Nuestro logaritmo ({data['tendencia_ibexia']})."
                else:
                    volumen_analisis_text = f"El volumen de <strong>{volumen_actual:,.0f} acciones</strong> es importante para confirmar cualquier movimiento. "
            else:
                volumen_analisis_text = f"El volumen de <strong>{volumen_actual:,.0f} acciones</strong> es importante para confirmar cualquier movimiento. "
        except Exception as e:
            volumen_analisis_text = f"El volumen de <strong>{volumen_actual:,.0f} acciones</strong> es importante para confirmar cualquier movimiento. No fue posible comparar con el volumen promedio: {e}"
    else:
        volumen_analisis_text = "El volumen de negociación no está disponible en este momento."

    titulo_post = f"{data['NOMBRE_EMPRESA']} ({data['TICKER']}) - Precio futuro previsto en 5 días: {data['PRECIO_PROYECTADO_5DIAS']:,.2f}€"

    # Datos para el gráfico principal de SMI y Precios
    smi_historico_para_grafico = data.get('SMI_HISTORICO_PARA_GRAFICO', [])
    cierres_para_grafico_total = data.get('CIERRES_PARA_GRAFICO_TOTAL', [])
    OFFSET_DIAS = data.get('OFFSET_DIAS_GRAFICO', 4)
    PROYECCION_FUTURA_DIAS = data.get('PROYECCION_FUTURA_DIAS_GRAFICO', 5)


    # NUEVA SECCIÓN DE ANÁLISIS DE GANANCIAS SIMULADAS
    # Llamamos a la nueva función para obtener el HTML y las listas de compras/ventas
    ganancias_html, compras_simuladas, ventas_simuladas = calcular_ganancias_simuladas(
        precios=data['PRECIOS_PARA_SIMULACION'],
        smis=data['SMI_PARA_SIMULACION'],
        fechas=data['FECHAS_PARA_SIMULACION']
    )

    # Añadimos las listas de compras y ventas al diccionario de datos
    data['COMPRAS_SIMULADAS'] = compras_simuladas
    data['VENTAS_SIMULADAS'] = ventas_simuladas
    
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
        <td style="padding: 8px;">Precio Objetivo de Compra</td>
        <td style="padding: 8px;"><strong>{data['PRECIO_OBJETIVO_COMPRA']:,}€</strong></td>
    </tr>
</table>
<br/>
"""
    
    # Nuevo HTML del gráfico (incluyendo el análisis detallado)
    chart_html = ""
    if smi_historico_para_grafico and cierres_para_grafico_total:
        labels_historial = data.get("FECHAS_HISTORIAL", [])
        labels_proyeccion = data.get("FECHAS_PROYECCION", [])
        labels_total = labels_historial + labels_proyeccion

        precios_reales_grafico = cierres_para_grafico_total[:30]
        data_proyectada = [None] * (len(labels_historial) - 1) + [precios_reales_grafico[-1]] + cierres_para_grafico_total[len(labels_historial):]

        smi_desplazados_para_grafico = smi_historico_para_grafico
        if len(smi_desplazados_para_grafico) < len(labels_total):
            smi_desplazados_para_grafico.extend([None] * (len(labels_total) - len(smi_desplazados_para_grafico)))
        
        # 2. Generación del análisis dinámico del gráfico
        analisis_grafico_html = "<h2>Análisis Detallado del Gráfico</h2>"
        precios = data['PRECIOS_PARA_SIMULACION']
        smis = data['SMI_PARA_SIMULACION']
        fechas = data['FECHAS_PARA_SIMULACION']
        
        analisis_grafico_html += f"<p>A continuación, analizaremos los movimientos clave de nuestro logaritmo y cómo se reflejaron en el precio de la acción:</p>"

        def get_trend(smi_val):
            if smi_val > 40:
                return "sobrecompra"
            elif smi_val < -40:
                return "sobreventa"
            elif smi_val > 0.1:
                return "alcista"
            elif smi_val < -0.1:
                return "bajista"
            else:
                return "consolidación"

        pendientes_smi = [0] * len(smis)
        for i in range(1, len(smis)):
            pendientes_smi[i] = smis[i] - smis[i-1]

        i = 1
        while i < len(smis):
            tendencia_actual_smi = get_trend(pendientes_smi[i])
            start_index = i - 1
            
            while i < len(smis) and get_trend(pendientes_smi[i]) == tendencia_actual_smi:
                i += 1
            
            end_index = i - 1
            
            # Descripción narrativa del tramo
            if tendencia_actual_smi == "alcista":
                analisis_grafico_html += f"<p>Desde el <strong>{fechas[start_index]}</strong>, nuestro logaritmo comenzó a girar y mostró una clara tendencia <strong>alcista</strong>. Este impulso llevó al precio hasta <strong>{precios[end_index]:,.2f}€</strong>.</p>"
            elif tendencia_actual_smi == "bajista":
                analisis_grafico_html += f"<p>A partir del <strong>{fechas[start_index]}</strong>, nuestro logaritmo giró a la baja. Durante esta tendencia <strong>bajista</strong>, el precio de la acción descendió hasta <strong>{precios[end_index]:,.2f}€</strong>.</p>"
            elif tendencia_actual_smi == "consolidación":
                analisis_grafico_html += f"<p>El período entre el <strong>{fechas[start_index]}</strong> y el <strong>{fechas[end_index]}</strong> fue de <strong>consolidación</strong>. Nuestro logaritmo se mantuvo plano y el precio se movió lateralmente, finalizando en <strong>{precios[end_index]:,.2f}€</strong>.</p>"
            
            # Chequeo de compra o venta en el cambio de tramo
            compra_en_giro = next((c for c in compras_simuladas if c['fecha'] == fechas[end_index]), None)
            if compra_en_giro:
                analisis_grafico_html += f"<p>✅ ¡Se detectó una señal de compra! Nuestro logaritmo mostró un giro y se compró en <strong>{compra_en_giro['precio']:,.2f}€</strong>.</p>"
            
            venta_en_giro = next((v for v in ventas_simuladas if v['fecha'] == fechas[end_index]), None)
            if venta_en_giro:
                analisis_grafico_html += f"<p>❌ ¡Se detectó una señal de venta! Se vendió en el giro a <strong>{venta_en_giro['precio']:,.2f}€</strong>.</p>"
    
        # Conclusión basada en la última tendencia
        ultima_tendencia = get_trend(pendientes_smi[-1])
        if ultima_tendencia == "alcista":
            analisis_grafico_html += f"<p>Actualmente, nuestro logaritmo muestra una tendencia <strong>alcista</strong>. Nos mantendremos en posición y atentos a los próximos movimientos para futuras ventas.</p>"
        elif ultima_tendencia == "bajista":
            analisis_grafico_html += f"<p>En estos momentos, nuestro logaritmo tiene una pendiente <strong>bajista</strong>. Esto no es momento de comprar, por lo que esperaremos una señal de giro más adelante.</p>"
        elif ultima_tendencia == "consolidación":
            analisis_grafico_html += f"<p>Nuestro logaritmo se encuentra en una fase de <strong>consolidación</strong>, moviéndose de forma lateral. Nos mantendremos atentos para entrar o salir del mercado cuando se detecte un giro claro.</p>"
        elif ultima_tendencia == "sobrecompra":
            analisis_grafico_html += f"<p>Nuestro logaritmo ha entrado en una zona de <strong>sobrecompra</strong>. Esto indica que la tendencia alcista podría estar agotándose y podríamos ver una señal de venta o un giro en cualquier momento.</p>"
        elif ultima_tendencia == "sobreventa":
            analisis_grafico_html += f"<p>Nuestro logaritmo se encuentra en una zona de <strong>sobreventa</strong>. Esto indica que la tendencia bajista está llegando a su fin y podríamos ver un giro y una señal de compra en breve.</p>"

        chart_html = f"""
        {analisis_grafico_html}
        <div style="width: 100%; max-width: 800px; margin: auto;">
            <canvas id="smiPrecioChart" style="height: 600px;"></canvas>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@1.4.0"></script>
        <script>
            // Configuración del gráfico
            var ctx = document.getElementById('smiPrecioChart').getContext('2d');
            var smiPrecioChart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {labels_total},
                    datasets: [
                        {{
                            label: 'Nuestro Logaritmo',
                            data: {smi_desplazados_para_grafico},
                            borderColor: 'rgb(255, 99, 132)',
                            backgroundColor: 'rgba(255, 99, 132, 0.5)',
                            yAxisID: 'y1',
                            pointRadius: 0,
                            borderWidth: 2
                        }},
                        {{
                            label: 'Precio Real',
                            data: {precios_reales_grafico},
                            borderColor: 'rgb(54, 162, 235)',
                            backgroundColor: 'rgba(54, 162, 235, 0.5)',
                            yAxisID: 'y',
                            pointRadius: 0,
                            borderWidth: 2
                        }},
                        {{
                            label: 'Precio Proyectado',
                            data: {data_proyectada},
                            borderColor: 'rgb(75, 192, 192)',
                            borderDash: [5, 5],
                            backgroundColor: 'rgba(75, 192, 192, 0.5)',
                            yAxisID: 'y',
                            pointRadius: 0,
                            borderWidth: 2
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
                    scales: {{
                        y: {{
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {{
                                display: true,
                                text: 'Precio (EUR)'
                            }}
                        }},
                        y1: {{
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {{
                                display: true,
                                text: 'Nuestro Logaritmo'
                            }},
                            grid: {{
                                drawOnChartArea: false,
                            }},
                            min: -100,
                            max: 100,
                            ticks: {{
                                stepSize: 20
                            }}
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            display: true
                        }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    let label = context.dataset.label || '';
                                    if (label) {{
                                        label += ': ';
                                    }}
                                    if (context.parsed.y !== null) {{
                                        label += context.parsed.y.toFixed(2);
                                    }}
                                    return label;
                                }}
                            }}
                        }},
                        annotation: {{
                            annotations: {{
                                compra: {{
                                    type: 'line',
                                    mode: 'horizontal',
                                    scaleID: 'y1',
                                    value: 40,
                                    borderColor: 'rgba(0, 128, 0, 0.5)',
                                    borderWidth: 2,
                                    label: {{
                                        content: 'Sobrecompra (+40)',
                                        enabled: true,
                                        position: 'start',
                                        backgroundColor: 'rgba(0, 128, 0, 0.5)'
                                    }}
                                }},
                                venta: {{
                                    type: 'line',
                                    mode: 'horizontal',
                                    scaleID: 'y1',
                                    value: -40,
                                    borderColor: 'rgba(255, 0, 0, 0.5)',
                                    borderWidth: 2,
                                    label: {{
                                        content: 'Sobreventa (-40)',
                                        enabled: true,
                                        position: 'start',
                                        backgroundColor: 'rgba(255, 0, 0, 0.5)'
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            }});
        </script>
        """
    else:
        chart_html = "<p>No hay suficientes datos para generar el gráfico.</p>"
    
    
    prompt = f"""
Actúa como un trader profesional con amplia experiencia en análisis técnico y mercados financieros. Genera el análisis completo en **formato HTML**, ideal para publicaciones web. Utiliza etiquetas `<h2>` para los títulos de sección y `<p>` para cada párrafo de texto. Redacta en primera persona, con total confianza en tu criterio.

Destaca los datos importantes como precios, cifras financieras y el nombre de la empresa utilizando la etiqueta `<strong>`. Asegúrate de que no haya asteriscos u otros símbolos de marcado en el texto final, solo HTML válido. Asegurate que todo este escrito en español independientemente del idioma de donde saques los datos.

Genera un análisis técnico completo de aproximadamente 800 palabras sobre la empresa {data['NOMBRE_EMPRESA']}, utilizando los siguientes datos reales extraídos de Yahoo Finance. Presta especial atención (pero no lo menciones) al **valor actual del SMI ({data['SMI']})**.

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
- Tendencia del SMI: {data['tendencia_ibexia']}


Importante: si algún dato no está disponible ("N/A", "No disponibles", "No disponible"), no lo menciones ni digas que falta. No expliques que la recomendación proviene de un indicador o dato específico. La recomendación debe presentarse como una conclusión personal basada en tu experiencia y criterio profesional como analista.

---
<h1>{titulo_post}</h1>

<h2>Análisis Inicial y Recomendación</h2>
<p>La cotización actual de <strong>{data['NOMBRE_EMPRESA']} ({data['TICKER']})</strong> se encuentra en <strong>{data['PRECIO_ACTUAL']:,}€</strong>. Nuestra recomendación es <strong>{data['RECOMENDACION']}</strong>. Según nuestras proyecciones, el precio podría situarse en <strong>{data['PRECIO_PROYECTADO_5DIAS']:,}€</strong> en los próximos 5 días. El volumen de negociación reciente fue de <strong>{data['VOLUMEN']:,} acciones</strong>. {data['motivo_analisis']}.</p>

<h2>La Clave: El Logaritmo como tu "Guía de Compra"</h2>
<p>Nuestro sistema se basa en un <strong>logaritmo</strong> que funciona como una brújula que te dice si es un buen momento para comprar o no. La clave está en cómo se mueve:</p>
<ul>
    <li>
        <strong>Si el logaritmo está en sobreventa (muy abajo):</strong> La acción podría estar "demasiado barata". Es probable que el logaritmo gire hacia arriba, lo que sería una <strong>señal de compra</strong>.
    </li>
    <li>
        <strong>Si el logaritmo está en sobrecompra (muy arriba):</strong> La acción podría estar "demasiado cara". El logaritmo podría girar a la baja, lo que sería una <strong>señal para no comprar</strong>.
    </li>
</ul>
<p>Más allá de la sobrecompra o sobreventa, la señal de compra más clara es cuando el logaritmo <strong>gira hacia arriba</strong>. Si ves que sube, es un buen momento para comprar (siempre y cuando no esté en una zona extrema de sobrecompra). Si gira a la baja, es mejor esperar.</p>

{chart_html}

<h2>Historial de Operaciones</h2>
{ganancias_html}

{tabla_resumen}
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
    try:
        all_tickers = leer_google_sheets()[1:]
    except Exception as e:
        print(f"❌ Error al leer Google Sheets: {e}. Asegúrate de que las variables de entorno están configuradas correctamente y el archivo JSON de credenciales es válido.")
        return
    
    if not all_tickers:
        print("No hay tickers para procesar.")
        return

    day_of_week = datetime.today().weekday()
    
    num_tickers_per_day = 10  
    total_tickers_in_sheet = len(all_tickers)
    
    start_index = (day_of_week * num_tickers_per_day) % total_tickers_in_sheet
    
    end_index = start_index + num_tickers_per_day
    
    tickers_for_today = []
    if end_index <= total_tickers_in_sheet:
        tickers_for_today = all_tickers[start_index:end_index]
    else:
        tickers_for_today = all_tickers[start_index:] + all_tickers[:end_index - total_tickers_in_sheet]

    if tickers_for_today:
        print(f"Procesando tickers para el día {datetime.today().strftime('%A')}: {tickers_for_today}")
        generar_contenido_con_gemini(tickers_for_today)
    else:
        print(f"No hay tickers disponibles para el día {datetime.today().strftime('%A')} en el rango calculado. "
              f"start_index: {start_index}, end_index: {end_index}, total_tickers: {total_tickers_in_sheet}")


if __name__ == '__main__':
    main()
