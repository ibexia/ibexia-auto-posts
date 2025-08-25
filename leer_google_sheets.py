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

def generar_explicacion_tramos(smis, fechas):
    """
    Genera una narrativa día por día del comportamiento del logaritmo (SMI).
    Identifica tramos de subida, bajada, aplanamiento, giros, sobrecompra y sobreventa.
    """
    narrativa = []
    if not smis or not fechas:
        return "<p>No hay datos suficientes para generar la explicación del gráfico.</p>"

    estado_anterior = None
    inicio_tramo = fechas[0]

    for i in range(1, len(smis)):
        if pd.isna(smis[i]) or pd.isna(smis[i-1]):
            continue

        diferencia = smis[i] - smis[i-1]
        fecha_actual = fechas[i]

        # Determinar estado actual
        if diferencia > 0.5:
            estado_actual = "subida"
        elif diferencia < -0.5:
            estado_actual = "bajada"
        else:
            estado_actual = "aplanamiento"

        # Detectar sobrecompra/sobreventa
        extra = ""
        if smis[i] > 40:
            extra = " entrando en <strong>sobrecompra</strong>"
        elif smis[i] < -40:
            extra = " acercándose a <strong>sobreventa</strong>"

        # Detectar giros
        if estado_anterior and estado_actual != estado_anterior:
            narrativa.append(
                f"<li>Del {inicio_tramo} al {fechas[i-1]} el logaritmo estuvo en {estado_anterior}. "
                f"El {fecha_actual} se produjo un <strong>giro</strong> hacia {estado_actual}{extra}.</li>"
            )
            inicio_tramo = fecha_actual
        elif i == len(smis) - 1:
            narrativa.append(
                f"<li>Del {inicio_tramo} al {fecha_actual} el logaritmo mostró una fase de {estado_actual}{extra}.</li>"
            )

        estado_anterior = estado_actual

    return "<h2>Evolución del Logaritmo</h2><ul>" + "".join(narrativa) + "</ul>"


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
        fechas_proyeccion = [(ultima_fecha_historial + timedelta(days=i)).strftime("%d/%m (fut.)") for i in range(1, PROYECCION_FUTURA_DIAS + 1)] # SMI para los 30 días del gráfico (serán los que se visualicen)
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
        elif len(cierres_history_full) > OFFSET_DIAS:
            # Si tenemos menos de 30 pero más que el offset
            # Tomamos lo que tengamos después del offset y rellenamos al principio
            temp_prices = cierres_history_full.iloc[OFFSET_DIAS:].tolist()
            first_price_val = temp_prices[0] if temp_prices else current_price
            precios_reales_para_grafico = [first_price_val] * (30 - len(temp_prices)) + temp_prices
        else:
            # Muy pocos datos históricos
            precios_reales_para_grafico = [current_price] * 30 # Default to current price if no historical data

        smi_history_last_30 = hist['SMI'].dropna().tail(30).tolist()

        # --- Lógica FINAL sin Precio Objetivo: Movimiento lineal constante ---
        precios_proyectados = []
        ultimo_precio_conocido = precios_reales_para_grafico[-1] if precios_reales_para_grafico else current_price
        # Determinar la dirección de la tendencia y el movimiento diario constante
        # Usamos la pendiente del SMI para determinar si la tendencia es alcista o bajista
        smi_history_full = hist_extended['SMI'].dropna()
        smi_ultimos_5 = smi_history_full.tail(5).dropna()
        if len(smi_ultimos_5) >= 2:
            x = np.arange(len(smi_ultimos_5))
            y = smi_ultimos_5.values
            pendiente_smi, _ = np.polyfit(x, y, 1)
        else:
            pendiente_smi = 0

        # Movimiento lineal constante: subida si SMI es positivo, bajada si es negativo
        if pendiente_smi > 0.1:
            movimiento_diario = (resistencia_1 - ultimo_precio_conocido) / PROYECCION_FUTURA_DIAS
        elif pendiente_smi < -0.1:
            movimiento_diario = (soporte_1 - ultimo_precio_conocido) / PROYECCION_FUTURA_DIAS
        else:
            movimiento_diario = 0

        for i in range(1, PROYECCION_FUTURA_DIAS + 1):
            nuevo_precio = ultimo_precio_conocido + (movimiento_diario * i)
            precios_proyectados.append(round(nuevo_precio, 2))

        # --- Fin Lógica Final ---

        # Combinar precios históricos con los precios proyectados
        # Aseguramos que solo tomamos los 30 precios históricos más recientes
        precios_para_grafico = precios_reales_para_grafico + precios_proyectados


        # Fechas del gráfico: 30 días de historial + 5 días de proyección
        fechas_grafico = fechas_historial + fechas_proyeccion

        # SMI proyectado: movimiento lineal simple basado en la pendiente actual
        smis_proyectados = []
        ultimo_smi_conocido = smi_historico_para_grafico[-1] if smi_historico_para_grafico else smi_actual
        for i in range(1, PROYECCION_FUTURA_DIAS + 1):
            nuevo_smi = ultimo_smi_conocido + (pendiente_smi * i * 0.1)
            smis_proyectados.append(round(nuevo_smi, 2))

        smis_para_grafico = smi_historico_para_grafico + smis_proyectados

        return {
            "current_price": current_price,
            "current_volume": formatear_numero(current_volume),
            "smi_actual": smi_actual,
            "soporte_1": soporte_1,
            "soporte_2": soporte_2,
            "soporte_3": soporte_3,
            "resistencia_1": resistencia_1,
            "resistencia_2": resistencia_2,
            "resistencia_3": resistencia_3,
            "precio_objetivo": precio_objetivo,
            "precio_objetivo_compra": precio_objetivo_compra,
            "recomendacion": recomendacion,
            "condicion_rsi": condicion_rsi,
            "volumen_promedio_30d": formatear_numero(volumen_promedio_30d),
            "fechas_grafico": fechas_grafico,
            "smis_para_grafico": smis_para_grafico,
            "precios_para_grafico": precios_para_grafico
        }

    except Exception as e:
        print(f"Error al obtener datos de Yahoo Finance para {ticker}: {e}")
        return None

def generar_contenido_con_gemini(tickers):
    if not tickers:
        print("No hay tickers para generar contenido.")
        return

    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        print("❌ Error: GEMINI_API_KEY no encontrada en las variables de entorno.")
        return

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    # Diccionario para almacenar los resultados del análisis
    resultados_analisis = {}

    for ticker in tickers:
        print(f"🔄 Procesando {ticker}...")
        datos = obtener_datos_yfinance(ticker)
        if datos:
            html_operaciones, compras, ventas = calcular_ganancias_simuladas(
                datos['precios_para_grafico'],
                datos['smis_para_grafico'],
                datos['fechas_grafico']
            )

            explicacion_tramos_html = generar_explicacion_tramos(
                datos['smis_para_grafico'],
                datos['fechas_grafico']
            )

            # Lógica de recomendación y condición SMI
            recomendacion = "Revisar"
            condicion_smi = "Neutral"

            if datos['smi_actual'] < -40:
                recomendacion = "Fuerte Compra"
                condicion_smi = "Sobreventa"
            elif datos['smi_actual'] >= -40 and datos['smi_actual'] < -20:
                recomendacion = "Compra"
                condicion_smi = "Acercándose a Sobreventa"
            elif datos['smi_actual'] > 40:
                recomendacion = "Venta"
                condicion_smi = "Sobrecompra"
            elif datos['smi_actual'] <= 40 and datos['smi_actual'] > 20:
                recomendacion = "Venta parcial"
                condicion_smi = "Acercándose a Sobrecompra"
            
            # Formatear el precio objetivo de compra
            precio_objetivo_compra_formateado = f"<strong>{datos['precio_objetivo_compra']:,.2f}€</strong>"
            if datos['precio_objetivo_compra'] == datos['current_price'] * 0.98:
                precio_objetivo_compra_formateado = f"~{precio_objetivo_compra_formateado}"

            # Construir el prompt con toda la información
            prompt = f"""
            Eres un analista de mercado experto, especializado en analizar el mercado de valores y los indicadores técnicos. Tu objetivo es generar un análisis exhaustivo y comprensible para el usuario sobre el ticker {ticker}. El análisis debe ser formal, muy completo y convincente, como si fuera de un analista de un gran banco.

            Aquí tienes los datos técnicos y las conclusiones del algoritmo:
            - Precio Actual: {datos['current_price']:,.2f}€
            - Volumen del día anterior: {datos['current_volume']}
            - SMI (Stochastic Momentum Index) Actual: {datos['smi_actual']:,.2f}
            - Soporte 1: {datos['soporte_1']:,.2f}€
            - Soporte 2: {datos['soporte_2']:,.2f}€
            - Soporte 3: {datos['soporte_3']:,.2f}€
            - Resistencia 1: {datos['resistencia_1']:,.2f}€
            - Resistencia 2: {datos['resistencia_2']:,.2f}€
            - Resistencia 3: {datos['resistencia_3']:,.2f}€
            - Precio Objetivo: {datos['precio_objetivo']:,.2f}€
            - Precio Objetivo de Compra: {datos['precio_objetivo_compra']:,.2f}€
            - Volumen Promedio (30 días): {datos['volumen_promedio_30d']}
            - Recomendación de nuestro logaritmo: {recomendacion}
            - Condición SMI: {condicion_smi}

            El análisis debe seguir esta estructura:

            ### Análisis Fundamental
            Crea una breve sección de análisis fundamental, basándote en un conocimiento general del ticker. Esta sección debe ser concisa y no usar los datos técnicos proporcionados.

            ### Análisis Técnico y de Volatilidad
            Utiliza los datos proporcionados para hacer un análisis técnico profundo y detallado. Describe la situación actual de los precios y el SMI, mencionando las zonas de soporte y resistencia. Habla sobre la fiabilidad de nuestro algoritmo para este activo, si el SMI está en sobrecompra, sobreventa o neutralidad. No es necesario mencionar todos los soportes y resistencias, solo los más relevantes o los que el algoritmo se está acercando.

            ### Proyección de Precios (Largo Plazo)
            Proporciona una proyección de precios a largo plazo basada en los datos proporcionados, especialmente el Precio Objetivo. Usa un tono de confianza, pero también prudente, mencionando que las proyecciones son un cálculo y no una garantía.

            ### Conclusiones y Recomendaciones
            Resume el análisis y presenta la recomendación final del logaritmo (compra, venta, etc.), explicando por qué se llegó a esa conclusión. Utiliza un tono profesional y persuasivo.

            **Ejemplo de formato:**
            * **Análisis Fundamental:** [Texto]
            * **Análisis Técnico y de Volatilidad:** [Texto]
            * **Proyección de Precios:** [Texto]
            * **Conclusiones y Recomendaciones:** [Texto]
            
            Además, genera una breve conclusión final de 2-3 frases, separada de la estructura anterior, que resuma todo el análisis en un tono muy motivador y convincente.
            """

            # Intentar generar el contenido 3 veces
            intentos = 0
            contenido_generado = ""
            while intentos < 3:
                try:
                    response = model.generate_content(prompt)
                    contenido_generado = response.text
                    break
                except Exception as e:
                    print(f"Error en el intento {intentos + 1} de generar contenido para {ticker}: {e}")
                    intentos += 1
                    time.sleep(10) # Espera antes de reintentar

            if contenido_generado:
                # Almacenar el resultado en el diccionario
                resultados_analisis[ticker] = {
                    "analisis": contenido_generado,
                    "html_operaciones": html_operaciones,
                    "explicacion_tramos_html": explicacion_tramos_html
                }
            else:
                print(f"❌ Error: No se pudo generar el contenido para {ticker} después de {intentos} reintentos.")
        
        print(f"⏳ Esperando 180 segundos antes de procesar el siguiente ticker...")
        time.sleep(180)


    # Enviar correo electrónico
    enviar_email_con_resultados(resultados_analisis)

def enviar_email_con_resultados(resultados):
    sender_email = os.getenv('SENDER_EMAIL')
    sender_password = os.getenv('SENDER_PASSWORD')
    receiver_email = os.getenv('RECEIVER_EMAIL')

    if not all([sender_email, sender_password, receiver_email]):
        print("❌ Error: Las variables de entorno para el correo electrónico no están configuradas correctamente.")
        return
    
    # Asunto del correo
    date_str = datetime.today().strftime('%d-%m-%Y')
    subject = f"Análisis de Mercado del {date_str}"

    # Construir el cuerpo HTML del correo
    html_body = f"""
    <html>
    <head>
    <style>
    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; color: #333; }}
    .container {{ max-width: 800px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
    .header {{ text-align: center; border-bottom: 2px solid #ddd; padding-bottom: 10px; margin-bottom: 20px; }}
    .header h1 {{ margin: 0; color: #0056b3; }}
    .ticker-section {{ border: 1px solid #eee; border-radius: 6px; padding: 15px; margin-bottom: 25px; }}
    .ticker-section h2 {{ color: #007bff; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
    .analysis p, .analysis ul, .analysis li {{ line-height: 1.6; margin: 0; padding: 0; }}
    .analysis ul {{ margin-top: 10px; }}
    .analysis li {{ margin-bottom: 5px; }}
    .simulacion h3 {{ color: #28a745; margin-top: 15px; }}
    .simulacion ul {{ list-style-type: none; padding: 0; }}
    .simulacion li {{ background: #f8f9fa; margin-bottom: 8px; padding: 10px; border-radius: 4px; border-left: 3px solid #007bff; }}
    .footer {{ text-align: center; color: #999; font-size: 0.8em; margin-top: 20px; }}
    .chart-placeholder {{ text-align: center; margin: 20px 0; }}
    </style>
    </head>
    <body>
    <div class="container">
        <div class="header">
            <h1>Análisis de Mercado Diario</h1>
            <p><strong>Fecha:</strong> {date_str}</p>
        </div>
    """

    for ticker, data in resultados.items():
        html_body += f"""
        <div class="ticker-section">
            <h2>{ticker}</h2>
            <div class="analysis">
                {data['analisis']}
            </div>
            <div class="simulacion">
                <h3>Simulación Histórica del Algoritmo</h3>
                {data['html_operaciones']}
            </div>
            <div class="evolucion">
                {data['explicacion_tramos_html']}
            </div>
        </div>
        """

    html_body += f"""
        <div class="footer">
            <p>Este informe fue generado automáticamente por un algoritmo de análisis de mercado. Las proyecciones no son una garantía de resultados futuros.</p>
        </div>
    </div>
    </body>
    </html>
    """

    try:
        # Configurar el mensaje
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = sender_email
        message["To"] = receiver_email

        # Añadir el cuerpo HTML
        part = MIMEText(html_body, "html")
        message.attach(part)

        # Conectar con el servidor SMTP y enviar el correo
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
            print("✅ Correo enviado exitosamente.")

    except Exception as e:
        print(f"❌ Error al enviar el correo: {e}")


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
        print(f"No hay tickers asignados para hoy, el índice calculado es {start_index} y el total de tickers es {total_tickers_in_sheet}.")


if __name__ == "__main__":
    main()
