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
        
        current_price = info.get("currentPrice")
        if current_price is None:
             raise ValueError("No se pudo obtener el precio actual del ticker de 'info'.")
        current_price = round(current_price, 3) # Este sigue siendo el precio actual

        # Ampliar periodo para el SMI, soportes/resistencias y simulación
        hist_extended = stock.history(period="90d", interval="1d")
        
        # --- FIX CRÍTICO: Comprobar si hay datos históricos ---
        if hist_extended.empty:
             raise ValueError("No se encontraron datos históricos de OHLC para el ticker en el período especificado (yfinance devolvió un DataFrame vacío).")
        # --- FIN FIX CRÍTICO ---

        hist_extended = calculate_smi_tv(hist_extended)

        # Obtener datos históricos para el volumen del día anterior completo
        hist_recent = stock.history(period="5d", interval="1d") 
        
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
        
        # OBTENCIÓN DE DATOS ADICIONALES PARA EL GRÁFICO CANDLESTICK (Open, High, Low)
        ohlc_history_full = hist_extended[['Open', 'High', 'Low', 'Close']].dropna()


        # Calcula el volumen promedio de los últimos 30 días usando hist_extended
        volumen_promedio_30d = hist_extended['Volume'].tail(30).mean()


        # Fechas reales de cotización para los últimos 30 días
        fechas_historial = ohlc_history_full.tail(30).index.strftime("%Y-%m-%d").tolist() # Formato ISO para ApexCharts
        ultima_fecha_historial = ohlc_history_full.index[-1] if not ohlc_history_full.empty else datetime.today()
        # Las fechas de proyección deben ser en un formato que ApexCharts pueda manejar, aunque es solo para relleno
        fechas_proyeccion = [(ultima_fecha_historial + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, PROYECCION_FUTURA_DIAS + 1)]
        
        # --- MANEJO ROBUSTO DE LOS 30 DÍAS DE DATOS PARA EL GRÁFICO ---
        # SMI para los 30 días del gráfico
        smi_historico_para_grafico = []
        if len(smi_history_full) >= 30:
            smi_historico_para_grafico = smi_history_full.tail(30).tolist()
        else:
            # Rellenar con el primer valor SMI disponible o 0.0 si no hay ninguno
            first_smi_val = smi_history_full.iloc[0] if not smi_history_full.empty else 0.0
            smi_historico_para_grafico = [first_smi_val] * (30 - len(smi_history_full)) + smi_history_full.tolist()

        # Datos OHLC para el gráfico (Candlestick)
        ohlc_reales_para_grafico = []
        if len(ohlc_history_full) >= 30:
            ohlc_history_for_chart = ohlc_history_full.tail(30)
            for idx, row in ohlc_history_for_chart.iterrows():
                # Formato de datos para ApexCharts: [timestamp, [Open, High, Low, Close]]
                # Usaremos la fecha en formato ISO y luego la convertiremos a timestamp en JS
                date_str = idx.strftime("%Y-%m-%d")
                ohlc_reales_para_grafico.append({
                    "x": date_str,
                    "y": [round(row['Open'], 3), round(row['High'], 3), round(row['Low'], 3), round(row['Close'], 3)]
                })
        else:
            # Si no hay 30 días, rellenar con el primer día conocido o el precio actual
            num_fill = 30 - len(ohlc_history_full)
            if not ohlc_history_full.empty:
                 first_row = ohlc_history_full.iloc[0]
                 default_ohlc = [round(first_row['Open'], 3), round(first_row['High'], 3), round(first_row['Low'], 3), round(first_row['Close'], 3)]
            else:
                 default_ohlc = [current_price] * 4 # Si no hay ningún dato

            # Rellenar fechas de forma inversa para que el rellenado vaya al inicio
            fecha_temp = ohlc_history_full.index[0] if not ohlc_history_full.empty else datetime.today()
            fechas_relleno = [(fecha_temp - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(num_fill, 0, -1)]
            
            # Crear datos de relleno
            for date_str in fechas_relleno:
                 ohlc_reales_para_grafico.append({"x": date_str, "y": default_ohlc})
            
            # Agregar los datos reales disponibles
            for idx, row in ohlc_history_full.iterrows():
                date_str = idx.strftime("%Y-%m-%d")
                ohlc_reales_para_grafico.append({
                    "x": date_str,
                    "y": [round(row['Open'], 3), round(row['High'], 3), round(row['Low'], 3), round(row['Close'], 3)]
                })
            
            fechas_historial = fechas_relleno + [idx.strftime("%Y-%m-%d") for idx in ohlc_history_full.index]


        # Precios de Cierre para la Simulación (usados también para calcular la proyección)
        precios_reales_para_simulacion = [data['y'][3] for data in ohlc_reales_para_grafico] # Usar los 30 cierres finales

        # Asegurarse de que las etiquetas de fecha coincidan con los 30 días de datos
        if len(fechas_historial) != 30:
             print(f"❌ Error crítico de longitud de fechas históricas. Fechas: {len(fechas_historial)}")
             fechas_historial = [] # Abortar el gráfico


        # --- NUEVA Lógica: Proyección lineal SIN soportes/resistencias (solo SMI) ---
        precios_proyectados = []
        ultimo_precio_conocido = precios_reales_para_simulacion[-1] if precios_reales_para_simulacion else current_price

        # Determinar la dirección de la tendencia y el movimiento diario constante
        smi_history_full_for_slope = hist_extended['SMI'].dropna()
        smi_ultimos_5_for_slope = smi_history_full_for_slope.tail(5).dropna()

        pendiente_smi = 0
        if len(smi_ultimos_5_for_slope) >= 2:
            x = np.arange(len(smi_ultimos_5_for_slope))
            y = smi_ultimos_5_for_slope.values
            pendiente_smi, _ = np.polyfit(x, y, 1)

        # Definir un movimiento diario constante (usaremos +/- 1% o +/- 0.5%)
        movimiento_diario = 0.0

        # Prioridad 1: Sobrecompra / Sobreventa Extrema (Fuerza de Reversión)
        if smi_actual > 40:
            # En sobrecompra: proyectamos caída (reversión)
            movimiento_diario = -0.01 
        elif smi_actual < -40:
            # En sobreventa: proyectamos subida (reversión)
            movimiento_diario = 0.01
        
        # Prioridad 2: Tendencia en Zona Media (SMI entre -40 y 40)
        # Se evalúa SÓLO si no se cumplió ninguna de las condiciones de extremos anteriores.
        elif -40 <= smi_actual <= 40:
            if pendiente_smi > 0.1:
                # Subiendo en zona media: proyectamos subida
                movimiento_diario = 0.005 # Subida moderada
            elif pendiente_smi < -0.1:
                # Bajando en zona media: proyectamos caída
                movimiento_diario = -0.005 # Caída moderada
            else:
                # Aplanado en zona media: proyectamos lateral
                movimiento_diario = 0.0

        for _ in range(PROYECCION_FUTURA_DIAS):
            siguiente_precio = ultimo_precio_conocido * (1 + movimiento_diario)
            siguiente_precio = round(siguiente_precio, 3)
            precios_proyectados.append(siguiente_precio)
            ultimo_precio_conocido = siguiente_precio

        # --- Fin de la NUEVA lógica lineal ---

        # Unir precios reales y proyectados (Solo para el cálculo de la simulación de ganancias)
        cierres_para_simulacion_total = precios_reales_para_simulacion + precios_proyectados
        precio_proyectado_dia_5 = cierres_para_simulacion_total[-1] if cierres_para_simulacion_total else current_price # Último precio proyectado a 5 días

        # SMI para simulación (solo los 30 días de historial)
        smi_historico_para_simulacion = [round(s, 3) for s in smi_history_full.tail(30).tolist()]
        # Fechas para simulación (formato DD/MM/AAAA para la simulación)
        fechas_para_simulacion = hist_extended.tail(30).index.strftime("%d/%m/%Y").tolist() 
        
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


        # --- NUEVOS DATOS PARA EL GRÁFICO APEXCHARTS ---
        # 1. Datos OHLC (Candlestick)
        # Los datos ya están en ohlc_reales_para_grafico
        
        # 2. SMI (Línea) y Fechas (Fechas históricas + proyección)
        smi_para_grafico_completo = smi_historico_para_grafico + [None] * PROYECCION_FUTURA_DIAS
        fechas_grafico_completo = fechas_historial + fechas_proyeccion

        # 3. Datos de Proyección de Precio (Línea)
        # Se necesita crear la serie de precios para la línea de proyección
        # [null] * 30 + [precios_proyectados]
        # Nota: ApexCharts manejará los 'null' sin conectar la línea
        precios_proyeccion_linea = [None] * len(fechas_historial) + precios_proyectados
        # Si no hay datos históricos (es un caso borde que debería estar cubierto por el rellenado)
        if not precios_reales_para_simulacion and precios_proyectados:
            # Si no hay historial, pero sí proyección (sólo debería ocurrir con el rellenado)
            # Creamos un punto de conexión en el precio actual
             punto_conexion = [current_price]
             precios_proyeccion_linea = [None] * (len(fechas_historial) - 1) + punto_conexion + precios_proyectados
        
        # 4. Datos de Cierre para el gráfico de línea (para mostrar el cierre real)
        # [Cierres reales] + [null] * 5
        cierres_reales_linea = precios_reales_para_simulacion + [None] * PROYECCION_FUTURA_DIAS


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
            # DATOS PARA EL GRÁFICO APEXCHARTS
            "OHLC_REALE_PARA_GRAFICO": ohlc_reales_para_grafico, # Nuevo: Datos Candlestick
            "SMI_PARA_GRAFICO_COMPLETO": smi_para_grafico_completo, # Nuevo: Datos SMI Linea
            "PRECIOS_PROYECCION_LINEA": precios_proyeccion_linea, # Nuevo: Datos Proyección Linea
            "CIERRES_REALES_LINEA": cierres_reales_linea, # Nuevo: Cierres Reales Linea (para overlay)
            "FECHAS_GRAFICO_COMPLETO": fechas_grafico_completo, # Nuevo: Etiquetas de Eje X
            # DATOS ANTIGUOS QUE AUN SE USAN EN SIMULACIÓN/PROYECCIÓN
            "CIERRES_30_DIAS": precios_reales_para_simulacion, # Cierres para la simulación
            "SMI_HISTORICO_PARA_GRAFICO": smi_historico_para_simulacion, # SMI para la simulación
            "CIERRES_PARA_GRAFICO_TOTAL": cierres_para_simulacion_total, # Usado para el cálculo del precio proyectado día 5
            "OFFSET_DIAS_GRAFICO": OFFSET_DIAS,
            "RESISTENCIA_1": resistencia_1,
            "RESISTENCIA_2": resistencia_2,
            "RESISTENCIA_3": resistencia_3,
            "PRECIO_OBJETIVO": precio_objetivo,
            "FECHAS_HISTORIAL": fechas_historial,
            "FECHAS_PROYECCION": fechas_proyeccion,
            "PRECIO_PROYECTADO_5DIAS": precio_proyectado_dia_5,
            'PRECIOS_PARA_SIMULACION': precios_reales_para_simulacion,
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

    # Datos para el gráfico principal de SMI y Precios
    smi_historico_para_grafico = data.get('SMI_HISTORICO_PARA_GRAFICO', [])
    cierres_para_grafico_total = data.get('CIERRES_PARA_GRAFICO_TOTAL', [])
    OFFSET_DIAS = data.get('OFFSET_DIAS_GRAFICO', 0) # Corregido a 0 para el nuevo manejo
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
        <div style="background-color: #fce4e4; color: #c62828; padding: 15px; margin: 20px 0; text-align: center; border-radius: 8px; border: 2px solid #e57373;">
            <p style="font-size: 1.2em; margin: 0; font-weight: bold;">
                ⚠️ ALERTA DE ALTO RIESGO (SMI Semanal en Sobrecompra):
            </p>
            <p style="margin: 5px 0 0 0;">
                El SMI semanal de <strong>{data['NOMBRE_EMPRESA']}</strong> está en <strong>{formatear_numero(data['SMI_SEMANAL'])}</strong>. A pesar de la señal de compra diaria, la sobrecompra en el marco temporal semanal sugiere que la subida podría ser débil o que una corrección está cerca. <br/> **Se recomienda actuar con extrema cautela y considerar la posibilidad de una caída a pesar de la proyección alcista.**
            </p>
        </div>
        """


    # Nuevo HTML del gráfico (incluyendo el análisis detallado)
    analisis_grafico_html = ""
    chart_html = ""

    # REVISIÓN CRÍTICA DE DATOS ANTES DE GENERAR EL GRÁFICO
    ohlc_data = data.get("OHLC_REALE_PARA_GRAFICO", [])
    smi_data = data.get("SMI_PARA_GRAFICO_COMPLETO", [])
    proj_data = data.get("PRECIOS_PROYECCION_LINEA", [])
    cierres_reales_linea = data.get("CIERRES_REALES_LINEA", [])
    fechas_completo = data.get("FECHAS_GRAFICO_COMPLETO", [])
    num_labels_hist = len(data.get("FECHAS_HISTORIAL", []))
    num_labels_total = len(fechas_completo)

    if not ohlc_data or not smi_data or num_labels_total == 0:
        chart_html = "<p>No hay suficientes datos válidos para generar el gráfico.</p>"
    else:
        # --- PREPARACIÓN DE DATOS PARA APEXCHARTS ---
        # 1. Datos Candlestick (OHLC) - Ya están en ohlc_data
        # Debemos serializar los datos OHLC para JS (formato: [{x: 'date', y: [O, H, L, C]}, ...])
        ohlc_json = json.dumps(ohlc_data)
        
        # 2. Datos SMI (Línea)
        # Formato: [{x: 'date', y: SMI}, ...] - SMI puede ser None (null en JSON) en la proyección
        smi_series = []
        for i, date_str in enumerate(fechas_completo):
             smi_series.append({"x": date_str, "y": smi_data[i] if i < len(smi_data) else None})
        smi_json = json.dumps(smi_series)
        
        # 3. Datos de Proyección de Precio (Línea)
        # Formato: [{x: 'date', y: Precio}, ...] - Los precios históricos deben ser None/null
        proj_series = []
        for i, date_str in enumerate(fechas_completo):
             proj_series.append({"x": date_str, "y": proj_data[i] if i < len(proj_data) else None})
        proj_json = json.dumps(proj_series)

        # 4. Datos de Cierre Reales (Línea)
        # Formato: [{x: 'date', y: Precio}, ...] - Los precios proyectados deben ser None/null
        cierres_reales_series = []
        for i, date_str in enumerate(fechas_completo):
             # Solo incluir el cierre real para el histórico, y None para la proyección
             cierres_reales_series.append({"x": date_str, "y": cierres_reales_linea[i] if i < num_labels_hist and i < len(cierres_reales_linea) else None})
        cierres_reales_json = json.dumps(cierres_reales_series)
        
        # ---- FIN DE LA PREPARACIÓN ----

        # Reemplazo para la sección de análisis detallado del gráfico
        analisis_grafico_html = f"""
        <h2 style="color: #333333; background-color: #e9e9e9; padding: 10px; border-radius: 5px; text-align: center;">Análisis Detallado del Gráfico</h2>
        <div style="background-color: #fafafa; padding: 15px; border-radius: 8px; border: 1px solid #dddddd;">
            <table style="width: 100%; border-collapse: collapse; color: #333333; font-family: Arial, sans-serif;">
                <thead>
                    <tr style="background-color: #dcdcdc; border-bottom: 2px solid #aaaaaa;">
                        <th style="padding: 12px; text-align: left; font-size: 14px; font-weight: bold;">Período</th>
                        <th style="padding: 12px; text-align: left; font-size: 14px; font-weight: bold;">Movimiento del Algoritmo</th>
                        <th style="padding: 12px; text-align: left; font-size: 14px; font-weight: bold;">Evolución del Precio</th>
                        <th style="padding: 12px; text-align: left; font-size: 14px; font-weight: bold;">Decisión / Estado</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        precios = data['PRECIOS_PARA_SIMULACION']
        smis = data['SMI_PARA_SIMULACION']
        fechas = data['FECHAS_PARA_SIMULACION']
        
        def get_trend(smi_val, prev_smi_val):
            # Analiza la pendiente
            if smi_val - prev_smi_val > 0.1:
                return "alcista"
            elif smi_val - prev_smi_val < -0.1:
                return "bajista"
            else:
                return "consolidación"

        def get_event_action(start_date, end_date):
            compra = next((c for c in data.get('COMPRAS_SIMULADAS', []) if c['fecha'] >= start_date and c['fecha'] <= end_date), None)
            venta = next((v for v in data.get('VENTAS_SIMULADAS', []) if v['fecha'] >= start_date and v['fecha'] <= end_date), None)
            
            if compra:
                return f"<strong>✅ Compra</strong> en {formatear_numero(compra['precio'])}€"
            elif venta:
                return f"<strong>❌ Venta</strong> en {formatear_numero(venta['precio'])}€"
            return "Sin operación"
            
        i = 1
        while i < len(smis):
            start_index = i - 1
            tendencia_actual = get_trend(smis[i], smis[i-1])
            
            while i < len(smis) and get_trend(smis[i], smis[i-1]) == tendencia_actual:
                i += 1
            
            end_index = i - 1
            
            fecha_inicio = fechas[start_index]
            fecha_fin = fechas[end_index]
            precio_inicio = formatear_numero(precios[start_index])
            precio_final = formatear_numero(precios[end_index])
            
            movimiento_algoritmo = ""
            evolucion_precio = f"De <strong>{precio_inicio}€</strong> a <strong>{precio_final}€</strong>"
            decision_inversion = get_event_action(fecha_inicio, fecha_fin)

            if tendencia_actual == "alcista":
                movimiento_algoritmo = "Tendencia alcista"
                evolucion_precio = f"<span style='color: #4CAF50;'>Subida</span> de <strong>{precio_inicio}€</strong> a <strong>{precio_final}€</strong>"
            elif tendencia_actual == "bajista":
                movimiento_algoritmo = "Tendencia bajista"
                evolucion_precio = f"<span style='color: #F44336;'>Bajada</span> de <strong>{precio_inicio}€</strong> a <strong>{precio_final}€</strong>"
            elif tendencia_actual == "consolidación":
                movimiento_algoritmo = "Fase de consolidación"
                evolucion_precio = f"<span style='color: #FFC107;'>Lateral</span> de <strong>{precio_inicio}€</strong> a <strong>{precio_final}€</strong>"

            analisis_grafico_html += f"""
                    <tr style="border-bottom: 1px solid #333333;">
                        <td style="padding: 12px; vertical-align: top; font-size: 12px;">{fecha_inicio} a {fecha_fin}</td>
                        <td style="padding: 12px; vertical-align: top; font-size: 12px;">{movimiento_algoritmo}</td>
                        <td style="padding: 12px; vertical-align: top; font-size: 12px;">{evolucion_precio}</td>
                        <td style="padding: 12px; vertical-align: top; font-size: 12px;">{decision_inversion}</td>
                    </tr>
            """
        
        # Última fila para el estado actual
        ultima_tendencia = "sin datos" 
        if len(smis) > 1:
             ultima_tendencia_smi = get_trend(smis[-1], smis[-2])
             if ultima_tendencia_smi == "alcista":
                ultima_tendencia = "alcista"
             elif ultima_tendencia_smi == "bajista":
                ultima_tendencia = "bajista"
             elif ultima_tendencia_smi == "consolidación":
                 ultima_tendencia = "consolidación"

        estado_actual = ""
        if ultima_tendencia == "alcista":
            estado_actual = "Actualmente, el Algoritmo muestra una **tendencia alcista**."
        elif ultima_tendencia == "bajista":
            estado_actual = "En estos momentos, el Algoritmo tiene una **tendencia bajista**."
        elif ultima_tendencia == "consolidación":
            estado_actual = "El Algoritmo se encuentra en una fase de **consolidación**, moviéndose de forma lateral."

        analisis_grafico_html += f"""
                </tbody>
            </table>
        </div>
        <p style="text-align: center; color: #aaaaaa; margin-top: 15px;">{estado_actual}</p>
        """

        # El gráfico en sí, que debe ir antes que el análisis
        # --- APEXCHARTS: CANDLESTICK CON LÍNEA SMI Y PROYECCIÓN ---
        chart_html = f"""
        <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
        <div id="chartCandlestick" style="width: 100%; max-width: 800px; margin: auto; background-color: #1a1a2e; padding: 20px; border-radius: 10px;"></div>

        <script>
        // Los datos se pasan como JSON de Python
        var ohlcData = {ohlc_json};
        var smiData = {smi_json};
        var projData = {proj_json};
        var cierresRealesData = {cierres_reales_json};

        // Convertir las fechas ISO a milisegundos para ApexCharts (necesario para el eje X de tipo datetime)
        ohlcData = ohlcData.map(d => ({{
            x: new Date(d.x).getTime(),
            y: d.y
        }}));
        smiData = smiData.map(d => ({{
            x: new Date(d.x).getTime(),
            y: d.y
        }}));
        projData = projData.map(d => ({{
            x: new Date(d.x).getTime(),
            y: d.y
        }}));
        cierresRealesData = cierresRealesData.map(d => ({{
            x: new Date(d.x).getTime(),
            y: d.y
        }}));
        
        // El SMI se graficará en un segundo gráfico, apilado (stacked)
        // La proyección se graficará como una línea sobre el Candlestick.

        // --- Gráfico Principal (Candlestick y Proyección) ---
        var optionsCandlestick = {{
            series: [
                {{
                    name: 'Precio Real',
                    type: 'candlestick',
                    data: ohlcData
                }},
                {{
                    name: 'Cierre Real',
                    type: 'line',
                    data: cierresRealesData,
                    color: '#2979ff'
                }},
                {{
                    name: 'Precio Proyectado',
                    type: 'line',
                    data: projData,
                    color: '#ffc107',
                    stroke: {{
                        dashArray: 5
                    }},
                    marker: {{
                        size: 0
                    }}
                }}
            ],
            chart: {{
                id: 'mainChart',
                height: 350,
                type: 'line',
                toolbar: {{
                    autoSelected: 'pan',
                    show: true
                }},
                animations: {{
                    enabled: false
                }},
                background: '#1a1a2e',
                foreColor: '#e0e0e0',
                stacked: false
            }},
            // AÑADE ESTO: stroke.width controla el grosor de la línea del borde de la vela.
            stroke: {{
                width: 0.5 // Puedes usar 1 o 0.5 para una línea muy fina.
            }},
            title: {{
                text: 'Gráfico Candlestick y Proyección',
                align: 'left',
                style: {{
                    color: '#e0e0e0'
                }}
            }},
            xaxis: {{
                type: 'datetime',
                tooltip: {{
                    enabled: true
                }},
                labels: {{
                    formatter: function(val) {{
                        return new Date(val).toLocaleDateString('es-ES', {{day: '2-digit', month: 'short'}});
                    }}
                }},
                axisBorder: {{
                    color: '#4a4a5e'
                }},
                axisTicks: {{
                    color: '#4a4a5e'
                }}
            }},
            yaxis: {{
                title: {{
                    text: 'Precio (EUR)',
                    style: {{
                        color: '#e0e0e0'
                    }}
                }},
                labels: {{
                    formatter: function(val) {{
                        return val.toFixed(2) + '€';
                    }}
                }},
                opposite: true
            }},
            plotOptions: {{
                candlestick: {{
                    colors: {{
                        up: '#00bfa5', // Verde para vela alcista (Open < Close)
                        down: '#ef5350' // Rojo para vela bajista (Open > Close)
                    }},
                    wick: {{
                        useFillColor: true
                    }}
                }}
            }},
            tooltip: {{
                theme: 'dark',
                x: {{
                    format: 'dd MMM yyyy'
                }}
            }},
            grid: {{
                borderColor: '#4a4a5e'
            }}
        }};

        // --- Gráfico Secundario (SMI) ---
        var optionsSMI = {{
            series: [
                {{
                    name: 'Nuestro Algoritmo (SMI)',
                    type: 'line',
                    data: smiData,
                    color: '#00bfa5'
                }}
            ],
            chart: {{
                id: 'smiChart',
                height: 150,
                type: 'line',
                toolbar: {{
                    autoSelected: 'pan',
                    show: false
                }},
                animations: {{
                    enabled: false
                }},
                background: '#1a1a2e',
                foreColor: '#e0e0e0',
                stacked: false
            }},
            stroke: {{
                width: [2]
            }},
            xaxis: {{
                type: 'datetime',
                labels: {{
                    show: true,
                    formatter: function(val) {{
                        return new Date(val).toLocaleDateString('es-ES', {{day: '2-digit', month: 'short'}});
                    }}
                }}
                }},
                tooltip: {{
                    enabled: false
                }},
                axisBorder: {{
                    color: '#4a4a5e'
                }},
                axisTicks: {{
                    color: '#4a4a5e'
                }}
            }},
            yaxis: {{
                min: -100,
                max: 100,
                tickAmount: 8,
                title: {{
                    text: 'Algoritmo',
                    style: {{
                        color: '#e0e0e0'
                    }}
                }},
                labels: {{
                    formatter: function(val) {{
                        return val.toFixed(1);
                    }}
                }},
                opposite: false
            }},
            grid: {{
                borderColor: '#4a4a5e'
            }},
            annotations: {{
                yaxis: [
                    {{
                        y: 40,
                        borderColor: '#d32f2f',
                        label: {{
                            borderColor: '#d32f2f',
                            style: {{
                                color: '#fff',
                                background: '#d32f2f'
                            }},
                            text: 'Sobrecompra (+40)'
                        }}
                    }},
                    {{
                        y: -40,
                        borderColor: '#388e3c',
                        label: {{
                            borderColor: '#388e3c',
                            style: {{
                                color: '#fff',
                                background: '#388e3c'
                            }},
                            text: 'Sobreventa (-40)'
                        }}
                    }}
                ]
            }},
            tooltip: {{
                theme: 'dark',
                shared: true,
                x: {{
                    formatter: function(val) {{
                        return new Date(val).toLocaleDateString('es-ES', {{day: '2-digit', month: '2-digit', year: 'numeric'}});
                    }}
                }}
            }}
        }};

        var chartCandlestick = new ApexCharts(document.querySelector("#chartCandlestick"), optionsCandlestick);
        chartCandlestick.render();
        
        // Crear un contenedor para el gráfico SMI debajo del Candlestick
        var smiContainer = document.createElement('div');
        smiContainer.id = 'chartSMI';
        smiContainer.style.width = '100%';
        smiContainer.style.maxWidth = '800px';
        smiContainer.style.margin = 'auto';
        document.getElementById('chartCandlestick').parentNode.insertBefore(smiContainer, document.getElementById('chartCandlestick').nextSibling);

        var chartSMI = new ApexCharts(document.querySelector("#chartSMI"), optionsSMI);
        chartSMI.render();
        </script>
        """
    
    # MODIFICACIÓN: Incluir SMI Semanal en la tabla de resumen
    tabla_resumen = f"""
<h2>Resumen de Puntos Clave</h2>
<table border="1" style="width:100%; border-collapse: collapse;">
    <tr>
        <th style="padding: 8px; text-align: left; background-color: #f3f3f2;">Métrica</th>
        <th style="padding: 8px; text-align: left; background-color: #f3f3f2;">Valor</th>
    </tr>
    <tr>
        <td style="padding: 8px;">Precio Actual</td>
        <td style="padding: 8px;"><strong>{formatear_numero(data['PRECIO_ACTUAL'])}€</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Volumen</td>
        <td style="padding: 8px;"><strong>{data['VOLUMEN']:,} acciones</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Soporte Clave</td>
        <td style="padding: 8px;"><strong>{formatear_numero(soportes_unicos[0])}€</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Resistencia Clave</td>
        <td style="padding: 8px;"><strong>{formatear_numero(data['RESISTENCIA'])}€</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Precio Objetivo de Compra</td>
        <td style="padding: 8px;"><strong>{formatear_numero(data['PRECIO_OBJETIVO_COMPRA'])}€</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">SMI Semanal</td>
        <td style="padding: 8px;"><strong>{formatear_numero(data['SMI_SEMANAL'])}</strong></td>
    </tr>
</table>
<br/>
"""

    
    prompt = f"""
Actúa como un generador de contenido estricto. Tu única tarea es completar las secciones HTML solicitadas a continuación, utilizando EXACTAMENTE el formato proporcionado. NO agregues ni elimines secciones, párrafos, ni introduzcas opiniones o análisis personales más allá del texto que ya se encuentra definido en las plantillas. Genera el análisis completo en **formato HTML**, ideal para publicaciones web. Utiliza etiquetas `<h2>` para los títulos de sección y `<p>` para cada párrafo de texto. Redacta en primera persona, con total confianza en tu criterio.

Destaca los datos importantes como precios, cifras financieras y el nombre de la empresa utilizando la etiqueta `<strong>`. Asegúrate de que no haya asteriscos u otros símbolos de marcado en el texto final, solo HTML válido. Asegurate que todo este escrito en español independientemente del idioma de donde saques los datos.

Genera un análisis técnico completo sobre la empresa {data['NOMBRE_EMPRESA']}, utilizando los siguientes datos reales extraídos de Yahoo Finance. Presta especial atención (pero no lo menciones) al **valor actual del SMI ({data['SMI']})** y al **SMI Semanal ({data['SMI_SEMANAL']})**.

¡ATENCIÓN URGENTE! Para CADA EMPRESA analizada, debes generar el CÓDIGO HTML Y JAVASCRIPT COMPLETO y Único para TODOS sus gráficos solicitados. Bajo ninguna circunstancia debes omitir ningún script, resumir bloques de código o utilizar frases como 'código JavaScript idéntico al ejemplo anterior'. Cada gráfico, para cada empresa, debe tener su script completamente incrustado, funcional e independiente de otros. Asegúrate de que los datos de cada gráfico corresponden SIEMPRE a la empresa que se está analizando en ese momento

**Datos clave:**
- Precio actual: {formatear_numero(data['PRECIO_ACTUAL'])}
- Volumen del último día completo: {data['VOLUMEN']}
- Soporte 1: {formatear_numero(data['SOPORTE_1'])}
- Soporte 2: {formatear_numero(data['SOPORTE_2'])}
- Soporte 3: {formatear_numero(data['SOPORTE_3'])}
- Resistencia clave: {formatear_numero(data['RESISTENCIA'])}
- Recomendación general: {data['RECOMENDACION']}
- SMI actual: {data['SMI']}
- SMI semanal: {data['SMI_SEMANAL']}
- Precio objetivo de compra: {formatear_numero(data['PRECIO_OBJETIVO_COMPRA'])}€
- Tendencia del SMI: {data['tendencia_ibexia']}


Importante: si algún dato no está disponible ("N/A", "No disponibles", "No disponible"), no lo menciones ni digas que falta. No expliques que la recomendación proviene de un indicador o dato específico.

---
<h1>{titulo_post}</h1>
<p style="font-size: 0.8em; color: #FF9800; font-weight: bold; text-align: center;">
    El análisis redactado a continuación se actualiza una vez por semana. La ficha superior SI se actualiza varias veces al día donde puedes ver nuestra posición en tiempo real y análisis resumido.
</p>

<h2>Análisis Inicial</h2>
<p>La cotización actual de <strong>{data['NOMBRE_EMPRESA']} ({data['TICKER']})</strong> se encuentra en <strong>{formatear_numero(data['PRECIO_ACTUAL'])}€</strong>. El volumen de negociación reciente fue de <strong>{data['VOLUMEN']:,} acciones</strong>. Recuerda que este análisis es solo para fines informativos y no debe ser considerado como asesoramiento financiero. Se recomienda encarecidamente que realices tu propia investigación y consultes a un profesional antes de tomar cualquier decisión de inversión.</p>

<h2>Historial de Operaciones</h2>
{ganancias_html}

{alerta_riesgo_html}

{anuncio_html}

{chart_html}
{analisis_grafico_html}

<h2>La Clave: El Algoritmo como tu "Guía de Compra"</h2>
<p>Nuestro sistema se basa en un <strong>Algoritmo</strong> que funciona como una brújula que te dice si es un buen momento para comprar o no. La clave está en cómo se mueve:</p>
<ul>
    <li>
        <strong>Si el Algoritmo está en sobreventa (muy abajo):</strong> La acción podría estar "demasiado barata". Es probable que el Algoritmo gire hacia arriba, lo que sería una <strong>señal de compra</strong>.
    </li>
    <li>
        <strong>Si el Algoritmo está en sobrecompra (muy arriba):</strong> La acción podría estar "demasiado cara". El Algoritmo podría girar a la baja, lo que sería una <strong>señal para no comprar</strong>.
    </li>
</ul>
<p>Más allá de la sobrecompra o sobreventa, la señal de compra más clara es cuando el Algoritmo <strong>gira hacia arriba</strong>. Si ves que sube, es un buen momento para comprar (siempre y cuando no esté en una zona extrema de sobrecompra). Si gira a la baja, es mejor esperar.</p>

{tabla_resumen}

**FIN DEL ANÁLISIS. NO AÑADAS NINGÚN TEXTO O SECCIÓN ADICIONAL DESPUÉS DEL RESUMEN DE PUNTOS CLAVE.**
"""
    return prompt, titulo_post

def enviar_email(texto_generado, asunto_email, nombre_archivo):
    """
    Envía un correo electrónico a través de Brevo (Sendinblue) con un archivo HTML adjunto,
    utilizando la configuración SMTP hardcodeada.
    """
    # 1. CONFIGURACIÓN HARDCODEADA DE BREVO Y DESTINATARIO
    servidor_smtp = 'smtp-relay.brevo.com'
    puerto_smtp = 587
    remitente_header = "IBEXIA.es <info@ibexia.es>" # Usado en el campo 'From'
    remitente_login = "9853a2001@smtp-brevo.com"    # Usuario SMTP para login
    password = "PRHTU5GN1ygZ9XVC"                   # Contraseña SMTP para login
    destinatario = "XUMKOX@GMAIL.COM"               # ¡DESTINATARIO HARCODEADO!
    
    # Extraer la dirección de correo visible (info@ibexia.es) del header completo
    match_remitente_email = re.search(r'<(.*?)>', remitente_header)
    # Esta dirección se usará como remitente en la transacción SMTP
    remitente_visible_email = match_remitente_email.group(1) if match_remitente_email else remitente_login
    
    ruta_archivo = f"{nombre_archivo}.html"
    
    # 2. Guardar el contenido generado en un archivo local temporal
    try:
        with open(ruta_archivo, "w", encoding="utf-8") as f:
            f.write(texto_generado)
    except Exception as e:
        print(f"❌ Error al escribir el archivo {ruta_archivo}: {e}")
        return

    # 3. Construcción del mensaje MIME
    msg = MIMEMultipart()
    msg['From'] = remitente_header # Ej: "IBEXIA.es <info@ibexia.es>"
    msg['To'] = destinatario
    msg['Subject'] = asunto_email
    
    # Cuerpo del email
    msg.attach(MIMEText("Adjunto el análisis en formato HTML.", 'plain'))

    # Adjuntar el archivo HTML
    try:
        with open(ruta_archivo, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        
        # Codificación y cabeceras para el adjunto
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {ruta_archivo}",
        )
        msg.attach(part)
    except Exception as e:
        print(f"❌ Error al adjuntar el archivo {ruta_archivo}: {e}")
        # Asegurarse de que el archivo temporal se borre incluso si falla el adjunto
        try:
            os.remove(ruta_archivo)
        except OSError:
            pass
        return
        
    # 4. Conexión al servidor Brevo SMTP
    try:
        print(f"🌐 Intentando conectar a Brevo SMTP: {servidor_smtp}:{puerto_smtp}")
        servidor = smtplib.SMTP(servidor_smtp, puerto_smtp)
        servidor.starttls() 
        
        print(f"🔑 Intentando iniciar sesión con el usuario: {remitente_login}")
        # Usa el login y la clave de Brevo para la autenticación
        servidor.login(remitente_login, password)
        
        print(f"✉️ Enviando correo a: {destinatario} desde: {remitente_visible_email}")
        # Usa el email visible como el remitente de la transacción
        servidor.sendmail(remitente_visible_email, destinatario, msg.as_string())
        
        servidor.quit()
        print("✅ Correo enviado exitosamente a Brevo.")

    except smtplib.SMTPAuthenticationError:
        print(f"❌ ERROR de Autenticación SMTP. Verifica el login y la clave SMTP de Brevo: {remitente_login}")
    except Exception as e:
        print(f"❌ Ocurrió un error al enviar el correo vía Brevo: {e}")
    finally:
        # 5. Limpieza (Borrar el archivo temporal)
        try:
            os.remove(ruta_archivo)
        except OSError as e:
            print(f"⚠️ Error al intentar borrar el archivo temporal {ruta_archivo}: {e}")


def generar_contenido_con_gemini(tickers):
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise Exception("No se encontró la variable de entorno GEMINI_API_KEY")

    model = genai.GenerativeModel(model_name="gemini-2.5-flash")  

    for ticker in tickers:
        print(f"\n📊 Procesando ticker: {ticker}")
        data = obtener_datos_yfinance(ticker)
        if not data:
            print(f"⏩ Saltando {ticker} debido a un error al obtener datos.")
            continue
        
        # ACCESO A LAS VARIABLES DESDE EL DICCIONARIO 'data'
        cierres_para_grafico_total = data.get('CIERRES_PARA_GRAFICO_TOTAL', [])
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

                    print(f"❌ Cuota de Gemini excedida al generar contenido. Reintentando en {delay_with_jitter:.3f} segundos... (Intento {retries + 1}/{max_retries})")
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
        print(f"No hay tickers disponibles para el día {datetime.today().strftime('%A')} en el rango calculado. "
              f"start_index: {start_index}, end_index: {end_index}, total_tickers: {total_tickers_in_sheet}")


if __name__ == '__main__':
    main()
