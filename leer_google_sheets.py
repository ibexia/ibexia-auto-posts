import os
import json
import smtplib
import yfinance as yf
import google.generativeai as genai
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
import re
import random

# NUEVA FUNCIÓN AÑADIDA PARA GARANTIZAR LA SERIALIZACIÓN A JSON/NULL
def safe_json_dump(data_list):
    """
    Serializa una lista de Python a una cadena JSON, asegurando que los valores None
    se conviertan a la palabra clave 'null' de JavaScript.
    """
    # json.dumps convierte None a 'null' y los floats a formato JavaScript (con punto decimal)
    # Se asegura de que la lista solo contenga valores o None, para que json.dumps funcione.
    # CORRECCIÓN DE SYNTAX ERROR EN ESTA LÍNEA (LÍNEA 25)
    return json.dumps([val if val is not None else None for val in data_list])


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
            return f"{num / 1_000_000_000:,.3f}B".replace(",", "X").replace(".", ",").replace("X", ".")
        elif abs(num) >= 1_000_000:
            return f"{num / 1_000_000:,.3f}M".replace(",", "X").replace(".", ",").replace("X", ".")
        elif abs(num) >= 1_000:
            return f"{num / 1_000:,.3f}K".replace(",", "X").replace(".", ",").replace("X", ".")
        else:
            return f"{num:,.3f}".replace(",", "X").replace(".", ",").replace("X", ".")
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

# NUEVA FUNCIÓN: Obtener SMI Semanal
def obtener_smi_semanal(ticker):
    try:
        stock = yf.Ticker(ticker)
        # 1 año de datos semanales para asegurar suficientes velas
        hist_weekly = stock.history(period="1y", interval="1wk")
        if hist_weekly.empty:
            print(f"⚠️ Advertencia: No hay datos semanales para {ticker}.")
            return 0.0

        hist_weekly = calculate_smi_tv(hist_weekly)
        smi_weekly_series = hist_weekly['SMI'].dropna()

        if not smi_weekly_series.empty:
            return round(smi_weekly_series.iloc[-1], 3)
        else:
            print(f"⚠️ Advertencia: No hay datos de SMI semanal válidos para {ticker}.")
            return 0.0
    except Exception as e:
        print(f"❌ Error al obtener SMI semanal para {ticker}: {e}")
        return 0.0
    
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
        print(f"[{fechas[i]}] SMI[i-1]={smis[i-1]:.3f}, SMI[i]={smis[i]:.3f}, pendiente[i]={pendientes_smi[i]:.3f}, pendiente[i-1]={pendientes_smi[i-1]:.3f}")
        # Señal de compra: la pendiente del SMI cambia de negativa a positiva y no está en sobrecompra
        # Se anticipa un día la compra y se añade la condición de sobrecompra
        if i >= 1 and pendientes_smi[i] > 0 and pendientes_smi[i-1] <= 0:
            if not posicion_abierta:
                if smis[i-1] < 40:
                    posicion_abierta = True
                    precio_compra_actual = precios[i-1]
                    compras.append({'fecha': fechas[i-1], 'precio': precio_compra_actual})
                    print(f"✅ COMPRA: {fechas[i-1]} a {precio_compra_actual:.3f}")
                else:
                    print(f"❌ No compra en {fechas[i-1]}: SMI demasiado alto ({smis[i-1]:.3f})")
            else:
                print(f"❌ No compra en {fechas[i-1]}: Ya hay posición abierta")

        # Señal de venta: la pendiente del SMI cambia de positiva a negativa (anticipando un día)
        elif i >= 1 and pendientes_smi[i] < 0 and pendientes_smi[i-1] >= 0:
            if posicion_abierta:
                posicion_abierta = False
                ventas.append({'fecha': fechas[i-1], 'precio': precios[i-1]})
                num_acciones = capital_inicial / precio_compra_actual
                ganancia_total += (precios[i-1] - precio_compra_actual) * num_acciones
                print(f"✅ VENTA: {fechas[i-1]} a {precios[i-1]:.3f}")
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

        operaciones_html += f"<li>Compra en {compra['fecha']} a <strong>{formatear_numero(compra['precio'])}€</strong>, Venta en {venta['fecha']} a <strong>{formatear_numero(venta['precio'])}€</strong> - {estado_ganancia}: <strong>{formatear_numero(ganancia_operacion)}€</strong></li>"

    html_resultados = ""

    if not compras:  # No se realizaron compras en el período
        html_resultados = f"""
        <p>No se encontraron señales de compra o venta significativas en el período analizado para Nuestro Algoritmo.</p>
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
            <p>Si hubieras invertido 10000€ en cada operación, tu ganancia simulada total (contando operaciones cerradas y la ganancia/pérdida actual de la posición abierta) sería de <strong>{formatear_numero(ganancia_simulada_total_incl_abierta)}€</strong>.</p>
            """
            # Si hay operaciones completadas (ventas realizadas), las mostramos
            if compras and posicion_abierta: # NUEVA LÍNEA AÑADIDA
                html_resultados += f"""
                <p>La última posición comprada fue en {compras[-1]['fecha']} a <strong>{formatear_numero(compras[-1]['precio'])}€</strong> y todavía no se ha vendido.</p>
                """
            if operaciones_html:
                html_resultados += f"""
                <p>A continuación, se detallan las operaciones completadas en el periodo analizado:</p>
                <ul>{operaciones_html}</ul>
                """
        else:  # Todas las posiciones se cerraron
            html_resultados = f"""
            <p>La fiabilidad de nuestro sistema se confirma en el histórico de operaciones. Nuestro Algoritmo ha completado un ciclo de compra y venta en el período. Si hubieras invertido {formatear_numero(capital_inicial)}€ en cada operación, tu ganancia simulada total habría sido de <strong>{formatear_numero(ganancia_total)}€</strong>.</p>
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
        
        # Ampliar periodo para el SMI, soportes/resistencias y simulación
        hist_extended = stock.history(period="90d", interval="1d")
        hist_extended = calculate_smi_tv(hist_extended)

        # Obtener datos históricos para el volumen del día anterior completo
        hist_recent = stock.history(period="5d", interval="1d") 
        
        current_price = round(info["currentPrice"], 3) # Este sigue siendo el precio actual

        current_volume = "N/A" # Inicializamos a N/A
        if not hist_recent.empty:
            if len(hist_recent) >= 2:
                current_volume = hist_recent['Volume'].iloc[-2] # Penúltima fila
            else: # Solo hay un día de datos (ejecutándose un lunes temprano y solo trae el viernes anterior)
                current_volume = hist_recent['Volume'].iloc[-1] # Última fila

        # Get last valid SMI signal
        smi_actual_series = hist_extended['SMI'].dropna() # Usamos el historial extendido para asegurar datos

        if not smi_actual_series.empty:
            smi_actual = round(smi_actual_series.iloc[-1], 3)
        else:
            print(f"⚠️ Advertencia: No hay datos de SMI válidos para {ticker}. Asignando SMI neutral.")
            smi_actual = 0  # Un valor por defecto para smi_actual
        
        # NUEVA ADICIÓN: Obtener SMI Semanal
        smi_semanal = obtener_smi_semanal(ticker)


        # Calcular soportes y resistencia
        # Asegurarse de tener al menos 30 días para un cálculo significativo
        hist_for_sr = hist_extended.tail(30) # Usar los últimos 30 días del historial extendido
        if len(hist_for_sr) < 1: # Si no hay datos, usar el precio actual
             highs_lows = np.array([current_price])
        else:
             highs_lows = hist_for_sr[['High', 'Low', 'Close']].values.flatten()
        

        # Calculamos soportes y resistencias como listas ordenadas
        # Soportes: de menor a mayor
        soportes_raw = np.unique(highs_lows)
        soportes = np.sort(soportes_raw).tolist()

        # Resistencias: de mayor a menor
        resistencias_raw = np.unique(highs_lows)
        resistencias = np.sort(resistencias_raw)[::-1].tolist() # Orden inverso para tener las más altas primero


        # Definir los 3 soportes
        if len(soportes) >= 3:
            soporte_1 = round(soportes[0], 3)
            soporte_2 = round(soportes[1], 3)
            soporte_3 = round(soportes[2], 3)
        elif len(soportes) == 2:
            soporte_1 = round(soportes[0], 3)
            soporte_2 = round(soportes[1], 3)
            soporte_3 = soporte_2 # Usar el mismo si no hay 3 distintos
        elif len(soportes) == 1:
            soporte_1 = round(soportes[0], 3)
            soporte_2 = soporte_1
            soporte_3 = soporte_1
        else:
            soporte_1, soporte_2, soporte_3 = round(current_price * 0.95, 3), round(current_price * 0.9, 3), round(current_price * 0.85, 3) # Default si no hay datos

        # Definir las 3 resistencias (similar a soportes)
        if len(resistencias) >= 3:
            resistencia_1 = round(resistencias[0], 3)
            resistencia_2 = round(resistencias[1], 3)
            resistencia_3 = round(resistencias[2], 3)
        elif len(resistencias) == 2:
            resistencia_1 = round(resistencias[0], 3)
            resistencia_2 = round(resistencias[1], 3)
            resistencia_3 = resistencia_2
        elif len(resistencias) == 1:
            resistencia_1 = round(resistencias[0], 3)
            resistencia_2 = resistencia_1
            resistencia_3 = resistencia_1
        else:
            resistencia_1, resistencia_2, resistencia_3 = round(current_price * 1.05, 3), round(current_price * 1.1, 3), round(current_price * 1.15, 3) # Default si no hay datos

        # --- LÓGICA MEJORADA PARA EL PRECIO OBJETIVO ---
        # Asegúrate de tener historial completo para calcular SMI reciente
        smi_history_full = hist_extended['SMI'].dropna()

        # Calcular pendiente de los últimos 5 días del SMI
        smi_ultimos_5 = smi_history_full.tail(5).dropna()
        pendiente_smi = 0
        if len(smi_ultimos_5) >= 2:
            x = np.arange(len(smi_ultimos_5))
            y = smi_ultimos_5.values
            pendiente_smi, _ = np.polyfit(x, y, 1)

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

        precio_objetivo = round(precio_objetivo, 3)
        # --- FIN NUEVA LÓGICA ---
        # --- FIN DE LA LÓGICA MEJORADA PARA EL PRECIO OBJETIVO ---

        # Precio objetivo de compra (ejemplo simple, puedes refinarlo)
        # Este 'precio_objetivo_compra' es diferente al 'precio_objetivo' general
        precio_objetivo_compra = round(current_price * 0.98, 3) # Un 2% por debajo del precio actual como ejemplo

        

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

        # --- MANEJO ROBUSTO DE LOS 30 DÍAS DE DATOS PARA EL GRÁFICO (APEXCHARTS) ---
        
        # 1. Datos OHLC y SMI para el gráfico (últimos 30 días con datos)
        # Usamos solo las filas donde tenemos todos los datos necesarios (OHLC y SMI)
        ohlc_smi_df_hist = hist_extended[['Open', 'High', 'Low', 'Close', 'SMI']].dropna().tail(30)
        
        ohlc_para_grafico = [] # Formato: [{x: timestamp_ms, y: [O, H, L, C]}, ...]
        smi_historico_para_grafico_formato_linea = [] # Formato: [[timestamp_ms, SMI_value], ...]
        precios_linea_para_grafico = [] # Formato: [[timestamp_ms, Close_value], ...]
        fechas_etiquetas_grafico = []

        if ohlc_smi_df_hist.empty:
            print(f"❌ Error: No hay suficientes datos OHLC/SMI para generar el gráfico.")
            # Continuamos con arrays vacíos
            ultimo_precio_conocido = current_price
            ultimo_timestamp_conocido_ms = int(datetime.today().timestamp() * 1000)
            
        else:
            for index, row in ohlc_smi_df_hist.iterrows():
                # Convertir timestamp a milisegundos Unix (requerido por ApexCharts)
                timestamp_ms = int(index.timestamp() * 1000)
                
                # Datos de la vela (OHLC)
                ohlc_para_grafico.append({
                    'x': timestamp_ms,
                    'y': [round(row['Open'], 3), round(row['High'], 3), round(row['Low'], 3), round(row['Close'], 3)]
                })
                
                # Datos de la línea de precio (Close)
                precios_linea_para_grafico.append([timestamp_ms, round(row['Close'], 3)])
                
                # Datos del SMI
                smi_historico_para_grafico_formato_linea.append([timestamp_ms, round(row['SMI'], 3)])
                
                # Etiquetas de fecha para la simulación
                fechas_etiquetas_grafico.append(index.strftime("%d/%m"))
            
            ultimo_precio_conocido = precios_linea_para_grafico[-1][1]
            ultimo_timestamp_conocido_ms = precios_linea_para_grafico[-1][0]

        
        # 2. Datos de Simulación de Ganancias (los últimos 30 días CON SMI válido)
        # Usamos los datos limpios de ohlc_smi_df_hist
        precios_para_simulacion = ohlc_smi_df_hist['Close'].apply(lambda x: round(x, 3)).tolist()
        smi_historico_para_simulacion = ohlc_smi_df_hist['SMI'].apply(lambda x: round(x, 3)).tolist()
        fechas_para_simulacion = ohlc_smi_df_hist.index.strftime("%d/%m/%Y").tolist() 

        # 3. Lógica de Proyección Lineal
        # (La lógica de pendiente/movimiento diario es la misma)
        # ... (Determinación de movimiento_diario - no modificado) ...
        smi_history_full_for_slope = hist_extended['SMI'].dropna()
        smi_ultimos_5_for_slope = smi_history_full_for_slope.tail(5).dropna()

        pendiente_smi = 0
        if len(smi_ultimos_5_for_slope) >= 2:
            x = np.arange(len(smi_ultimos_5_for_slope))
            y = smi_ultimos_5_for_slope.values
            pendiente_smi, _ = np.polyfit(x, y, 1)

        movimiento_diario = 0.0
        if smi_actual > 40:
            movimiento_diario = -0.01 
        elif smi_actual < -40:
            movimiento_diario = 0.01
        elif -40 <= smi_actual <= 40:
            if pendiente_smi > 0.1:
                movimiento_diario = 0.005 
            elif pendiente_smi < -0.1:
                movimiento_diario = -0.005
            else:
                movimiento_diario = 0.0
        # ... (Fin determinación de movimiento_diario) ...


        precios_proyectados_linea = []
        fechas_proyeccion = []
        # El último índice conocido (que debería ser un día de trading)
        ultima_fecha_historial = ohlc_smi_df_hist.index[-1].to_pydatetime() if not ohlc_smi_df_hist.empty else datetime.today()
        current_date_for_projection = ultima_fecha_historial + timedelta(days=1)

        for _ in range(PROYECCION_FUTURA_DIAS):
            # Encontrar el siguiente día de la semana (saltar Sáb/Dom)
            while current_date_for_projection.weekday() >= 5: # 5 es Sábado, 6 es Domingo
                 current_date_for_projection += timedelta(days=1)
            
            siguiente_precio = ultimo_precio_conocido * (1 + movimiento_diario)
            siguiente_precio = round(siguiente_precio, 3)
            
            # Formato de línea [Timestamp (ms), Value]
            timestamp_ms = int(current_date_for_projection.timestamp() * 1000)
            precios_proyectados_linea.append([timestamp_ms, siguiente_precio])
            
            # Actualizar
            ultimo_precio_conocido = siguiente_precio
            current_date_for_projection += timedelta(days=1) # Avanzar un día
            
            # Etiqueta para el gráfico
            fechas_proyeccion.append(current_date_for_projection.strftime("%d/%m (fut.)"))


        # Unir precios reales y proyectados para la línea de tendencia
        cierres_para_grafico_total_linea = precios_linea_para_grafico + precios_proyectados_linea
        precio_proyectado_dia_5 = cierres_para_grafico_total_linea[-1][1] if cierres_para_grafico_total_linea else current_price # Último precio proyectado a 5 días

        # Lógica de tendencia para la nota
        tendencia_ibexia = "No disponible"
        slope = 0.0
        if len(smi_historico_para_simulacion) >= 2:
            x = np.arange(len(smi_historico_para_simulacion))
            y = np.array(smi_historico_para_simulacion)
            if np.std(y) > 0.01:
                slope, intercept = np.polyfit(x, y, 1)
            else:
                slope = 0.0

            if slope > 0.1:
                tendencia_ibexia = "mejorando (alcista)"
            elif slope < -0.1:
                tendencia_ibexia = "empeorando (bajista)"
            else:
                tendencia_ibexia = "cambio de tendencia"
        
        # --- JSON Serialization (NEW) ---
        # Asegurarse de que se usa json.dumps en los arrays de datos para ApexCharts
        ohlc_data_json = json.dumps(ohlc_para_grafico)
        smi_data_json = json.dumps(smi_historico_para_grafico_formato_linea)
        proyeccion_data_json = json.dumps(cierres_para_grafico_total_linea)


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
            "SMI_SEMANAL": smi_semanal, # NUEVA ADICIÓN
            "PRECIO_OBJETIVO_COMPRA": precio_objetivo_compra,
            "tendencia_ibexia": tendencia_ibexia, # Renombrado de TENDENCIA_NOTA
            "OHLC_PARA_GRAFICO_JSON": ohlc_data_json, # NUEVO
            "SMI_HISTORICO_PARA_GRAFICO_JSON": smi_data_json, # NUEVO
            "CIERRES_PARA_GRAFICO_TOTAL_LINEA_JSON": proyeccion_data_json, # NUEVO
            "RESISTENCIA_1": resistencia_1,
            "RESISTENCIA_2": resistencia_2,
            "RESISTENCIA_3": resistencia_3,
            "PRECIO_OBJETIVO": precio_objetivo,
            "FECHAS_HISTORIAL": fechas_etiquetas_grafico, # Etiquetas de los 30 días de historia
            "FECHAS_PROYECCION": fechas_proyeccion, # Etiquetas de los 5 días futuros
            "PRECIO_PROYECTADO_5DIAS": precio_proyectado_dia_5,
            'PRECIOS_PARA_SIMULACION': precios_para_simulacion,
            'SMI_PARA_SIMULACION': smi_historico_para_simulacion,
            'FECHAS_PARA_SIMULACION': fechas_para_simulacion,
            "PROYECCION_FUTURA_DIAS_GRAFICO": PROYECCION_FUTURA_DIAS
        }
        
        # --- NUEVA LÓGICA DE RECOMENDACIÓN BASADA EN PROYECCIÓN DE PRECIO Y RIESGO SEMANAL ---
        diferencia_precio_porcentual = ((precio_proyectado_dia_5 - current_price) / current_price) * 100 if current_price != 0 else 0

        recomendacion = "sin dirección clara"
        motivo_analisis = "La proyección de precio a 5 días es muy similar al precio actual, lo que indica un mercado en consolidación. Se recomienda cautela."
        
        if diferencia_precio_porcentual > 3:
            recomendacion = "Comprar (Impulso Fuerte)"
            motivo_analisis = f"El precio proyectado a 5 días de {formatear_numero(precio_proyectado_dia_5)}€ es significativamente superior al precio actual, indicando un fuerte impulso alcista."
        elif diferencia_precio_porcentual > 1:
            recomendacion = "Comprar (Impulso Moderado)"
            motivo_analisis = f"El precio proyectado a 5 días de {formatear_numero(precio_proyectado_dia_5)}€ es superior al precio actual, sugiriendo un impulso alcista moderado."
        elif diferencia_precio_porcentual < -3:
            recomendacion = "Vender (Impulso Fuerte)"
            motivo_analisis = f"El precio proyectado a 5 días de {formatear_numero(precio_proyectado_dia_5)}€ es significativamente inferior al precio actual, lo que indica una fuerte presión bajista."
        elif diferencia_precio_porcentual < -1:
            recomendacion = "Vender (Impulso Moderado)"
            motivo_analisis = f"El precio proyectado a 5 días de {formatear_numero(precio_proyectado_dia_5)}€ es inferior al precio actual, sugiriendo un impulso bajista moderado."
        
        # Lógica de RIESGO: Si la recomendación es de compra y SMI semanal está en sobrecompra
        if "Comprar" in recomendacion and smi_semanal > 40:
            recomendacion = recomendacion.replace("Comprar", "Compra (ALTO RIESGO)")
            motivo_analisis += f" **ADVERTENCIA DE RIESGO:** A pesar del impulso alcista diario/proyectado, el Algoritmo Semanal (SMI Semanal en {smi_semanal:.3f}) se encuentra en zona de sobrecompra (> 40), lo que aumenta el riesgo de una corrección a corto plazo. Se recomienda extrema cautela."
        
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
                        volumen_analisis_text = f"El volumen negociado de <strong>{volumen_actual:,.0f} acciones</strong> es notablemente superior al promedio reciente, indicando un fuerte interés del mercado y validando la actual tendencia de Nuestro Algoritmo ({data['tendencia_ibexia']})."
                    elif cambio_porcentual_volumen < -30:
                        volumen_analisis_text = f"El volumen de <strong>{volumen_actual:,.0f} acciones</strong> es inferior a lo habitual, lo que podría sugerir cautela en la actual tendencia. Una confirmación de la señal de Nuestro Algoritmo ({data['tendencia_ibexia']}) requeriría un aumento en la participación del mercado."
                    else:
                        volumen_analisis_text = f"El volumen de <strong>{volumen_actual:,.0f} acciones</strong> se mantiene en línea con el promedio. Es un volumen adecuado, pero no excepcional, para confirmar de manera contundente la señal de Nuestro Algoritmo ({data['tendencia_ibexia']})."
                else:
                    volumen_analisis_text = f"El volumen de <strong>{volumen_actual:,.0f} acciones</strong> es importante para confirmar cualquier movimiento. "
            else:
                volumen_analisis_text = f"El volumen de <strong>{volumen_actual:,.0f} acciones</strong> es importante para confirmar cualquier movimiento. "
        except Exception as e:
            volumen_analisis_text = f"El volumen de <strong>{volumen_actual:,.0f} acciones</strong> es importante para confirmar cualquier movimiento. No fue posible comparar con el volumen promedio: {e}"
    else:
        volumen_analisis_text = "El volumen de negociación no está disponible en este momento."

    # NUEVO FORMATO: "Analisis actualizado el FECHA de NOMBRE DE LA EMPRESA."
    fecha_actual_str = datetime.today().strftime('%d/%m/%Y')
    titulo_post = f"Análisis actualizado el {fecha_actual_str} de {data['NOMBRE_EMPRESA']}. "

    # --- DATOS PARA APEXCHARTS ---
    ohlc_data_json = data.get('OHLC_PARA_GRAFICO_JSON', '[]')
    smi_data_json = data.get('SMI_HISTORICO_PARA_GRAFICO_JSON', '[]')
    proyeccion_data_json = data.get('CIERRES_PARA_GRAFICO_TOTAL_LINEA_JSON', '[]')


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
            # Usar una tolerancia para considerar que son diferentes
            if abs(temp_soportes[i] - soportes_unicos[-1]) / (soportes_unicos[-1] or 1) > 0.005:
                soportes_unicos.append(temp_soportes[i])
    
    if not soportes_unicos:
        soportes_unicos.append(0.0)

    soportes_texto = ""
    if len(soportes_unicos) == 1:
        soportes_texto = f"un soporte clave en <strong>{formatear_numero(soportes_unicos[0])}€</strong>."
    elif len(soportes_unicos) == 2:
        soportes_texto = f"dos soportes importantes en <strong>{formatear_numero(soportes_unicos[0])}€</strong> y <strong>{formatear_numero(soportes_unicos[1])}€</strong>."
    elif len(soportes_unicos) >= 3:
        soportes_texto = (f"tres soportes relevantes: el primero en <strong>{formatear_numero(soportes_unicos[0])}€</strong>, "
                          f"el segundo en <strong>{formatear_numero(soportes_unicos[1])}€</strong>, y el tercero en <strong>{formatear_numero(soportes_unicos[2])}€</strong>.")
    else:
        soportes_texto = "no presenta soportes claros en el análisis reciente, requiriendo un seguimiento cauteloso."

    # Bloque de código a insertar en construir_prompt_formateado
    # Va después del 'Historial de Operaciones' y antes del 'Gráfico'
    anuncio_html = """
    <div style="background-color: #DB6927; color: #FFFFFF; padding: 15px; margin: 20px 0; text-align: center; border-radius: 8px; border: 1px solid #cceeff;">
        <p style="font-size: 1.1em; margin: 0; font-weight: bold;">
            Este analísis detallado lo hacemos 1 vez por semana para cada empresa, si no quieres esperar en la pagina principal consulta tu empresa en el buscador, el análisis lo actualizamos tres veces al día. <a href="https://ibexia.es/" style="color: #007bff; font-weight: bold; text-decoration: underline;">**ENTRA.**</a>
        </p>
    </div>
    """
    
    # NUEVA ADICIÓN: Alerta de riesgo si es "Compra (ALTO RIESGO)"
    alerta_riesgo_html = ""
    if "ALTO RIESGO" in data['RECOMENDACION']:
         alerta_riesgo_html = f"""
        <div style="background-color: #fce4e4; color: #c62828; padding: 15px; margin: 20px 0; text-align: center; border-radius: 8px; border: 1px solid #c62828; font-weight: bold;">
            ⚠️ {data['RECOMENDACION']} - {data['motivo_analisis']}
        </div>
        """
    

    # --- INICIO DE LA CONSTRUCCIÓN DEL PROMPT HTML ---
    html_prompt = f"""
    <div style="background-color: #f4f4f4; padding: 20px; border-radius: 10px; font-family: Arial, sans-serif;">
        
        <h1 style="color: #333; border-bottom: 2px solid #DB6927; padding-bottom: 10px; margin-bottom: 20px; text-align: center;">
            {titulo_post}
        </h1>
        
        {alerta_riesgo_html}

        <div style="display: flex; flex-wrap: wrap; justify-content: space-around; background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="margin: 10px; padding: 10px; border-left: 3px solid #007bff;">
                <p style="margin: 0; font-size: 1.2em; font-weight: bold; color: #333;">Precio Actual:</p>
                <p style="margin: 0; font-size: 1.5em; color: #28a745;">{formatear_numero(data['PRECIO_ACTUAL'])}€</p>
            </div>
            <div style="margin: 10px; padding: 10px; border-left: 3px solid #DB6927;">
                <p style="margin: 0; font-size: 1.2em; font-weight: bold; color: #333;">Recomendación IBEXIA:</p>
                <p style="margin: 0; font-size: 1.5em; color: {'#c62828' if 'Vender' in data['RECOMENDACION'] else '#28a745' if 'Comprar' in data['RECOMENDACION'] else '#007bff'};">{data['RECOMENDACION']}</p>
            </div>
            <div style="margin: 10px; padding: 10px; border-left: 3px solid #ffc107;">
                <p style="margin: 0; font-size: 1.2em; font-weight: bold; color: #333;">Proyección 5 Días:</p>
                <p style="margin: 0; font-size: 1.5em; color: #17a2b8;">{formatear_numero(data['PRECIO_PROYECTADO_5DIAS'])}€</p>
            </div>
        </div>

        <h2 style="color: #333; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px;">Análisis de Nuestro Algoritmo (SMI)</h2>
        
        <p style="line-height: 1.6;">
            Nuestro algoritmo de inversión ha determinado que el valor de **{data['NOMBRE_EMPRESA']} ({data['TICKER']})** está en una fase de <strong>{data['tendencia_ibexia']}</strong>.
            Actualmente, el indicador **SMI** (Stochastic Momentum Index) se sitúa en <strong>{data['SMI']:.3f}</strong>, y el SMI Semanal en <strong>{data['SMI_SEMANAL']:.3f}</strong>.
            Esto nos lleva a la siguiente conclusión:
        </p>

        <p style="background-color: #e9ecef; padding: 15px; border-left: 4px solid #DB6927; font-style: italic;">
            **Motivo del Análisis:** {data['motivo_analisis']}
        </p>
        
        {anuncio_html}

        <h2 style="color: #333; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px;">Resistencias, Soportes y Volumen</h2>
        
        <p style="line-height: 1.6;">
            La resistencia más cercana se encuentra en <strong>{formatear_numero(data['RESISTENCIA_1'])}€</strong>. Superar este nivel podría confirmar la tendencia alcista. Por otro lado, la cotización de la acción presenta {soportes_texto}
        </p>
        <p style="line-height: 1.6;">
            {volumen_analisis_text}
        </p>

        <h2 style="color: #333; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px;">Historial de Operaciones Simuladas por Nuestro Algoritmo</h2>
        <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            {ganancias_html}
        </div>

        <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
        <div style="margin-top: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 8px;">
            <h3 style="text-align: center; color: #333; margin-bottom: 15px;">Gráfico de Velas, SMI y Proyección (Últimos 30 Días + 5 Días de Proyección)</h3>
            <div id="apexchart-candlestick" style="width: 100%; height: 500px;"></div>
        </div>

        <script>
            // --- DATOS DEL GRÁFICO ---
            const ohlcData = {ohlc_data_json};
            const smiData = {smi_data_json};
            const proyeccionData = {proyeccion_data_json};
            
            // Función para formatear el timestamp Unix a fecha legible
            function formatDate(timestamp) {{
                const date = new Date(timestamp);
                const day = date.getDate().toString().padStart(2, '0');
                const month = (date.getMonth() + 1).toString().padStart(2, '0');
                return `${{day}}/{{month}}`;
            }}

            // Colores base
            const colorAlcista = '#00e396'; // Verde para velas y tendencias alcistas
            const colorBajista = '#ff4560'; // Rojo para velas y tendencias bajistas
            const colorNeutro = '#008ffb'; // Azul para el SMI

            // --- CONFIGURACIÓN DE APEXCHARTS ---
            var options = {{
                series: [
                    {{
                        name: 'Velas (OHLC)',
                        type: 'candlestick',
                        data: ohlcData,
                    }},
                    {{
                        name: 'SMI (Algoritmo)',
                        type: 'line',
                        data: smiData,
                    }},
                    {{
                        name: 'Precio y Proyección 5D',
                        type: 'line',
                        data: proyeccionData,
                    }}
                ],
                chart: {{
                    height: 500,
                    type: 'line',
                    toolbar: {{
                        show: true,
                        tools: {{
                            download: true,
                            selection: false,
                            zoom: true,
                            zoomin: true,
                            zoomout: true,
                            pan: true,
                            reset: true | '<img src="/static/icons/reset.png" width="20">'
                        }},
                    }}
                }},
                title: {{
                    text: 'Gráfico de Velas y Tendencia',
                    align: 'left',
                    style: {{
                        fontSize: '16px'
                    }}
                }},
                xaxis: {{
                    type: 'datetime',
                    labels: {{
                        formatter: function(val) {{
                            return formatDate(val); // Mostrar DD/MM
                        }},
                        style: {{
                             fontSize: '12px'
                        }}
                    }},
                    tooltip: {{
                        enabled: true,
                        formatter: function(val) {{
                            return formatDate(val);
                        }}
                    }}
                }},
                yaxis: [
                    {{
                        seriesName: 'Velas (OHLC)',
                        axisTicks: {{
                            show: true
                        }},
                        axisBorder: {{
                            show: true,
                            color: colorAlcista
                        }},
                        labels: {{
                            style: {{
                                colors: colorAlcista,
                            }},
                            formatter: function(val) {{
                                return val.toFixed(3) + '€';
                            }}
                        }},
                        title: {{
                            text: "Precio (€)",
                            style: {{
                                color: colorAlcista
                            }}
                        }},
                        tooltip: {{
                            enabled: true
                        }},
                        min: Math.min(...ohlcData.map(d => d.y[2])) * 0.98,
                        max: Math.max(...ohlcData.map(d => d.y[1])) * 1.02,
                    }},
                    {{
                        seriesName: 'SMI (Algoritmo)',
                        opposite: true,
                        axisTicks: {{
                            show: true
                        }},
                        axisBorder: {{
                            show: true,
                            color: colorNeutro
                        }},
                        labels: {{
                            style: {{
                                colors: colorNeutro,
                            }},
                            formatter: function(val) {{
                                return val.toFixed(2);
                            }}
                        }},
                        title: {{
                            text: "SMI",
                            style: {{
                                color: colorNeutro
                            }}
                        }},
                        min: -100,
                        max: 100
                    }}
                ],
                tooltip: {{
                    x: {{
                        formatter: function(val) {{
                            return formatDate(val);
                        }}
                    }},
                    y: {{
                        formatter: function (val, opts) {{
                            if (opts.seriesIndex === 0) {{
                                // Candlestick tooltip (OHLC)
                                 const dataPoint = opts.w.config.series[0].data[opts.dataPointIndex].y;
                                 return `A: ${{(dataPoint[0]).toFixed(3)}}€<br>M: ${{(dataPoint[1]).toFixed(3)}}€<br>m: ${{(dataPoint[2]).toFixed(3)}}€<br>C: ${{(dataPoint[3]).toFixed(3)}}€`;
                            }} else if (opts.seriesIndex === 1) {{
                                // SMI
                                return `${{val.toFixed(3)}}`;
                            }} else if (opts.seriesIndex === 2) {{
                                // Proyección
                                return `${{val.toFixed(3)}}€`;
                            }}
                            return val;
                        }}
                    }}
                }},
                stroke: {{
                    width: [1, 2, 2] // Grosor de las series (Velas, SMI, Proyección)
                }},
                colors: [colorAlcista, colorNeutro, colorBajista], // Velas (se ignora), SMI (Azul), Proyección (Rojo)
                plotOptions: {{
                    candlestick: {{
                        colors: {{
                            up: colorAlcista, // Verde
                            down: colorBajista  // Rojo
                        }}
                    }}
                }},
                markers: {{
                    size: [0, 0, 4] // Sin marcadores para Velas y SMI, 4px para Proyección
                }},
                legend: {{
                     tooltipHoverFormatter: function(val, opts) {{
                        return val + ' - ' + opts.w.globals.series[opts.seriesIndex][opts.dataPointIndex] + ''
                    }}
                }},
                grid: {{
                    row: {{
                        colors: ['#f3f3f3', 'transparent'], // Fondo
                        opacity: 0.5
                    }},
                }}
            }};

            // Inicialización de la gráfica: Nos aseguramos de que haya datos
            if (ohlcData.length > 0) {{
                var chart = new ApexCharts(document.querySelector("#apexchart-candlestick"), options);
                chart.render();
            }} else {{
                document.querySelector("#apexchart-candlestick").innerHTML = '<p style="text-align: center; color: #c62828;">No hay suficientes datos históricos (OHLC/SMI) para generar el gráfico de velas.</p>';
            }}
        </script>
        <p style="text-align: center; font-size: 0.8em; color: #6c757d; margin-top: 30px;">
            Este análisis es generado automáticamente por el Algoritmo de IBEXIA.
        </p>
    </div>
    """
    return html_prompt


def enviar_email(html_prompt, ticker, email_list, subject_prefix="Análisis de"):
    try:
        sender_email = os.getenv('SENDER_EMAIL')
        sender_password = os.getenv('SENDER_PASSWORD')
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        
        if not sender_email or not sender_password:
             print("❌ Error: Variables de entorno SENDER_EMAIL o SENDER_PASSWORD no configuradas.")
             return

        msg = MIMEMultipart("alternative")
        msg['Subject'] = f"{subject_prefix} {ticker} - IBEXIA Algoritmo"
        msg['From'] = sender_email
        
        # Unir todos los destinatarios para el campo To (aunque se envía individualmente)
        msg['To'] = ", ".join(email_list)
        
        part = MIMEText(html_prompt, "html")
        msg.attach(part)
        
        # Conectar y enviar el correo
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            
            # Enviar a cada destinatario individualmente (BCC implícito)
            for recipient in email_list:
                try:
                    server.sendmail(sender_email, recipient, msg.as_string())
                    print(f"✅ Email enviado con éxito a {recipient} para {ticker}.")
                except Exception as e:
                    print(f"❌ Error al enviar email a {recipient}: {e}")

    except Exception as e:
        print(f"❌ Error general en la función enviar_email: {e}")
        

def generar_contenido_con_gemini(tickers_for_today):
    try:
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("La variable de entorno GEMINI_API_KEY no está configurada.")
        
        genai.configure(api_key=gemini_api_key)
        client = genai.Client()
        
    except Exception as e:
        print(f"❌ Error al configurar Gemini: {e}")
        return

    # Usar emails de prueba si la lista de la hoja de cálculo está vacía o si solo hay un elemento (el encabezado)
    email_list_raw = leer_google_sheets()
    if len(email_list_raw) <= 1:
        email_list = ["test@example.com", "otro@test.com"] # Lista de prueba si no hay emails reales
        print("⚠️ Advertencia: Usando lista de emails de prueba.")
    else:
        # Asumiendo que el primer elemento es un encabezado y los siguientes son los emails
        email_list = email_list_raw[1:]
        print(f"Usando lista de emails leída: {email_list}")

    for ticker in tickers_for_today:
        print(f"\n--- Procesando Ticker: {ticker} ---")
        
        try:
            data = obtener_datos_yfinance(ticker)
            if data is None:
                continue

            # Construir el prompt para la simulación de ganancias (antes del prompt a Gemini)
            # Esto pobla el diccionario 'data' con el HTML de ganancias y las operaciones
            ganancias_html, compras_simuladas, ventas_simuladas = calcular_ganancias_simuladas(
                precios=data['PRECIOS_PARA_SIMULACION'],
                smis=data['SMI_PARA_SIMULACION'],
                fechas=data['FECHAS_PARA_SIMULACION']
            )
            # Actualizar el diccionario data con los resultados de la simulación
            data['COMPRAS_SIMULADAS'] = compras_simuladas
            data['VENTAS_SIMULADAS'] = ventas_simuladas

            # Generar el HTML final con el gráfico ApexCharts integrado
            html_prompt_completo = construir_prompt_formateado(data)

            # Enviar el email con el HTML generado
            enviar_email(html_prompt_completo, ticker, email_list, subject_prefix=f"Análisis Técnico {ticker}")

        except Exception as e:
            print(f"❌ Error general al procesar {ticker}: {e}")
            
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
    
    num_tickers_per_day = 12  
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
        print(f"No hay tickers para procesar hoy.")

if __name__ == "__main__":
    main()
