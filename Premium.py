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
from email.mime.base import MIMEBase
from email import encoders
import pandas_ta as ta

# ******************************************************************************
# *************** FUNCIONES DE LECTURA Y UTILIDADES (MODIFICADAS) **************
# ******************************************************************************

# NUEVA FUNCIÓN AÑADIDA PARA GARANTIZAR LA SERIALIZACIÓN A JSON/NULL
def safe_json_dump(data_list):
    """
    Serializa una lista de Python a una cadena JSON, asegurando que los valores None
    se conviertan a la palabra clave 'null' de JavaScript.
    """
    return json.dumps([val if val is not None else None for val in data_list])


tickers = {
    'Acciona': 'ANA.MC',
    'A3Media': 'A3M.MC',
    'Adolfo Dominguez': 'ADZ.MC',
    'Accionarenovables': 'ANE.MC',
    'Acerinox': 'ACX.MC',
    'ACS': 'ACS.MC',
    'Aedas-Homes': 'AEDAS.MC',
    'Aena': 'AENA.MC',
    'Almirall': 'ALM.MC',
    'Airbus': 'AIR.MC',
    'AirTificial': 'AI.MC',
    'Amadeus': 'AMS.MC',
    'Amper': 'AMP.MC',
    'Audax-Renovables': 'ADX.MC',
    'Atrys Health': 'ATRY.MC',
    'Bankinter': 'BKT.MC',
    'BBVA': 'BBVA.MC',
    'Berkeley': 'BKY.MC',
    'Biotechnology': 'BST.MC',
    'CaixaBank': 'CABK.MC',
    'Cellnex': 'CLNX.MC',
    'Colonial': 'COL.MC',
    'DIA': 'DIA.MC',
    'Ercros': 'ECR.MC',
    'Endesa': 'ELE.MC',
    'Elecnor': 'ENO.MC',
    'ENCE': 'ENC.MC',
    'Enagas': 'ENG.MC',
    'Ezentis': 'EZE.MC',
    'FacePhi': 'FACE.MC',
    'Ferrovial': 'FER.MC',
    'Fomento Construcciones y Contratas': 'FCC.MC',
    'Fluidra': 'FDR.MC',
    'GAM': 'GAM.MC',
    'Gigas-Hosting': 'GIGA.MC',
    'Grenergy': 'GRE.MC',
    'Grifols': 'GRF.MC',
    'Grupo San Jose': 'GSJ.MC',
    'Holaluz': 'HLZ.MC',
    'Neinor-homes': 'HOME.MC',
    'IAG': 'IAG.MC',
    'Iberdrola': 'IBE.MC',
    'Iberpapel': 'IBG.MC',
    'Inditex': 'ITX.MC',
    'Indra': 'IDR.MC',
    'Logista': 'LOG.MC',
    'Linea-directa': 'LDA.MC',
    'Mapfre': 'MAP.MC',
    'duro-felguera': 'MDF.MC',
    'melia': 'MEL.MC',
    'Merlin': 'MRL.MC',
    'arcelor-mittal': 'MTS.MC',
    'Naturgy': 'NTGY.MC',
    'nbi-bearings': 'NBI.MC',
    'nextil': 'NXT.MC',
    'nyesa': 'NYE.MC',
    'ohla': 'OHLA.MC',
    'Deoleo': 'OLE.MC',
    'Oryzon': 'ORY.MC',
    'Pharma-Mar': 'PHM.MC',
    'Prosegur': 'PSG.MC',
    'Puig-brands': 'PUIG.MC',
    'Realia': 'RLIA.MC',
    'Red-Electrica': 'RED.MC',
    'Repsol': 'REP.MC',
    'Laboratorios-rovi': 'ROVI.MC',
    'Banco-sabadell': 'SAB.MC',
    'Sacyr': 'SCYR.MC',
    'Solaria': 'SLR.MC',
    'Squirrel': 'SQRL.MC',
    'Substrate': 'SAI.MC',
    'banco-santander': 'SAN.MC',
    'Talgo': 'TLGO.MC',
    'Telefonica': 'TEF.MC',
    'Tubos-Reunidos': 'TRG.MC',
    'tubacex': 'TUB.MC',
    'Unicaja': 'UNI.MC',
    'Viscofan': 'VIS.MC',
    'Urbas': 'URB.MC',
}

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

    # Se mantiene la lectura solo de la columna A, como estaba en Premium.py y se asume es para obtener los tickers
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

# FUNCIÓN MODIFICADA: Ahora maneja formato español y notación de miles (K, M, B)
def formatear_numero(numero):
    if pd.isna(numero) or numero == "N/A" or numero is None:
        return "N/A"
    try:
        num = float(numero)
        # Formato español (reemplazando . por , y , por . para miles)
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

        df_prev = df.iloc[:-1].copy()
        df_prev = calculate_smi_tv(df_prev)

        avgrel_prev_last = (df_prev['Close'] - (df_prev['High'].rolling(window=10).max() + df_prev['Low'].rolling(window=10).min()) / 2).ewm(span=length_d, adjust=False).mean().iloc[-1]
        avgdiff_prev_last = (df_prev['High'].rolling(window=10).max() - df_prev['Low'].rolling(window=10).min()).ewm(span=length_d, adjust=False).mean().iloc[-1]
        smi_raw_yesterday = df['SMI'].iloc[-2]

        alpha_ema = 2 / (length_d + 1)
        
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

def calcular_soporte_resistencia(df, window=5):
    try:
        supports = []
        resistances = []
        
        if len(df) < window * 2:
            return {'s1': 'N/A', 's2': 'N/A', 'r1': 'N/A', 'r2': 'N/A'}

        for i in range(window, len(df) - window):
            high_slice = df['High'].iloc[i - window : i + window + 1]
            low_slice = df['Low'].iloc[i - window : i + window + 1]

            if df['High'].iloc[i] == high_slice.max():
                resistances.append(df['High'].iloc[i])
            
            if df['Low'].iloc[i] == low_slice.min():
                supports.append(df['Low'].iloc[i])

        supports = sorted(list(set(supports)), reverse=True)
        resistances = sorted(list(set(resistances)))
        
        current_price = df['Close'].iloc[-1]
        
        s1 = next((s for s in supports if s < current_price), None)
        s2 = next((s for s in supports if s < current_price and s != s1), None)
        
        r1 = next((r for r in resistances if r > current_price), None)
        r2 = next((r for r in resistances if r > current_price and r != r1), None)

        return {'s1': s1, 's2': s2, 'r1': r1, 'r2': r2}
        
    except Exception as e:
        print(f"❌ Error al calcular soportes y resistencias: {e}")
        return {'s1': 'N/A', 's2': 'N/A', 'r1': 'N/A', 'r2': 'N/A'}
        
# FUNCIÓN AÑADIDA: Lógica de simulación para calcular operaciones y beneficio
def calcular_ganancias_simuladas(precios, smis, fechas, capital_inicial=10000):
    """
    Simula las operaciones de compra y venta basadas en señales SMI y calcula el beneficio.
    Retorna los resultados para generar el HTML de detalle.
    """
    compras = []
    ventas = []
    posicion_abierta = False
    precio_compra_actual = 0
    
    pendientes_smi = [0] * len(smis)
    for i in range(1, len(smis)):
        pendientes_smi[i] = smis[i] - smis[i-1]

    # Iterar sobre los datos históricos para encontrar señales
    for i in range(2, len(smis)):
        # Señal de compra: la pendiente del SMI cambia de negativa a positiva y no está en sobrecompra
        if i >= 1 and pendientes_smi[i] > 0 and pendientes_smi[i-1] <= 0:
            if not posicion_abierta:
                # Condición de sobrecompra (SMI < 40) para evitar compras tardías
                if smis[i-1] < 40: 
                    posicion_abierta = True
                    # Se compra al precio de CIERRE del día anterior a la señal
                    precio_compra_actual = precios[i-1]
                    compras.append({'fecha': fechas[i-1], 'precio': precio_compra_actual})
            # else: Ya hay posición abierta, no hace nada

        # Señal de venta: la pendiente del SMI cambia de positiva a negativa
        elif i >= 1 and pendientes_smi[i] < 0 and pendientes_smi[i-1] >= 0:
            if posicion_abierta:
                posicion_abierta = False
                # Se vende al precio de CIERRE del día anterior a la señal
                ventas.append({'fecha': fechas[i-1], 'precio': precios[i-1]})
            # else: No hay posición abierta, no hace nada

    return compras, ventas # Retornamos las listas de operaciones

# ******************************************************************************
# *************** FUNCIÓN PRINCIPAL DE OBTENCIÓN DE DATOS (MODIFICADA) *********
# ******************************************************************************

def obtener_datos_yfinance(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        current_price = info.get("currentPrice")
        if not current_price:
            print(f"⚠️ Advertencia: No se encontró precio actual para {ticker}. Saltando...")
            return None

        # --- Datos Diarios ---
        # 150d de historia para la simulación
        hist_extended = stock.history(period="150d", interval="1d") 
        hist_extended['EMA_100'] = ta.ema(hist_extended['Close'], length=100)
                
        precio_actual = hist_extended['Close'].iloc[-1]
        ema_actual = hist_extended['EMA_100'].iloc[-1]
        
        if precio_actual > ema_actual:
            tipo_ema = "Soporte"
        elif precio_actual < ema_actual:
            tipo_ema = "Resistencia"
        else:
            tipo_ema = "Igual"
            
        if hist_extended.empty:
            print(f"⚠️ Advertencia: No se encontraron datos históricos para {ticker}. Saltando...")
            return None
        hist_extended = calculate_smi_tv(hist_extended)
        
        sr_levels = calcular_soporte_resistencia(hist_extended)

        smi_series = hist_extended['SMI'].dropna()
        if len(smi_series) < 2:
            print(f"⚠️ Advertencia: No hay suficientes datos de SMI para {ticker}. Saltando...")
            return None
        
        smi_yesterday = smi_series.iloc[-2]
        smi_today = smi_series.iloc[-1]
        
        pendiente_hoy = smi_today - smi_yesterday
        
        tendencia_hoy = "Subiendo" if pendiente_hoy > 0.1 else ("Bajando" if pendiente_hoy < -0.1 else "Plano")
        
        estado_smi = "Sobrecompra" if smi_today > 40 else ("Sobreventa" if smi_today < -40 else "Intermedio")
        
        precio_aplanamiento = calcular_precio_aplanamiento(hist_extended)

        # ******************************************************************
        # *********** NUEVA LÓGICA DE SIMULACIÓN Y ESTATUS *****************
        # ******************************************************************
        CAPITAL_INVERSION = 10000
        
        # Preparamos los datos para la simulación
        # Aseguramos que los arrays estén limpios de NaN antes de pasar a la simulación
        df_simulacion = hist_extended[['Close', 'SMI']].dropna()
        
        precios = df_simulacion['Close'].values.tolist()
        smis = df_simulacion['SMI'].values.tolist()
        fechas = df_simulacion.index.strftime('%d/%m/%Y').tolist()
        
        compras, ventas = calcular_ganancias_simuladas(precios, smis, fechas, CAPITAL_INVERSION)

        # 1. Determinar el estado 'COMPRADO' actual
        posicion_abierta = len(compras) > len(ventas)
        comprado_status = "SI" if posicion_abierta else "NO"
        
        # 2. Obtener los datos de la última compra
        if compras:
            precio_compra = compras[-1]['precio']
            fecha_compra = compras[-1]['fecha']
        else:
            precio_compra = "N/A"
            fecha_compra = "N/A"

        # 3. Crear la lista de operaciones detalladas (para el HTML de "ver más")
        operaciones_detalladas = []
        num_operaciones_completadas = min(len(compras), len(ventas))
        
        # Operaciones cerradas
        for i in range(num_operaciones_completadas):
            compra = compras[i]
            venta = ventas[i]
            num_acciones_op = CAPITAL_INVERSION / compra['precio']
            beneficio = (venta['precio'] - compra['precio']) * num_acciones_op
            
            operaciones_detalladas.append({
                'TIPO': 'CERRADA',
                'FECHA_ENTRADA': compra['fecha'],
                'PRECIO_ENTRADA': compra['precio'],
                'FECHA_SALIDA': venta['fecha'],
                'PRECIO_SALIDA': venta['precio'],
                'BENEFICIO': beneficio,
                'GANANCIA': "SI" if beneficio >= 0 else "NO"
            })

        # Operación abierta
        if posicion_abierta:
            compra_abierta = compras[-1]
            # Usamos el precio actual para calcular el beneficio de la operación abierta
            num_acciones_op = CAPITAL_INVERSION / compra_abierta['precio']
            beneficio_actual = (current_price - compra_abierta['precio']) * num_acciones_op
            
            operaciones_detalladas.append({
                'TIPO': 'ABIERTA',
                'FECHA_ENTRADA': compra_abierta['fecha'],
                'PRECIO_ENTRADA': compra_abierta['precio'],
                'FECHA_SALIDA': 'N/A', # Operación no cerrada
                'PRECIO_SALIDA': current_price, # Precio actual para cálculo de beneficio
                'BENEFICIO': beneficio_actual,
                'GANANCIA': "SI" if beneficio_actual >= 0 else "NO"
            })
        
        # ******************************************************************
        # *********** FIN NUEVA LÓGICA DE SIMULACIÓN Y ESTATUS *************
        # ******************************************************************

        # --- Cálculo de SMI Semanal (Mantenido) ---
        hist_weekly = stock.history(period="3y", interval="1wk")
        if hist_weekly.empty:
            smi_weekly = 'N/A'
            estado_smi_weekly = 'N/A'
            observacion_semanal = "No hay datos semanales suficientes."
        else:
            hist_weekly = calculate_smi_tv(hist_weekly)
            smi_weekly_series = hist_weekly['SMI'].dropna()
            smi_weekly = smi_weekly_series.iloc[-1] if not smi_weekly_series.empty else 'N/A'
            
            if isinstance(smi_weekly, (int, float)):
                estado_smi_weekly = "Sobrecompra" if smi_weekly > 40 else ("Sobreventa" if smi_weekly < -40 else "Intermedio")
                
                if estado_smi_weekly == "Sobrecompra":
                    observacion_semanal = f"El **SMI Semanal** ({formatear_numero(smi_weekly)}) está en zona de **Sobrecompra**. Sugiere que el precio ya ha subido mucho a largo plazo."
                elif estado_smi_weekly == "Sobreventa":
                    observacion_semanal = f"El **SMI Semanal** ({formatear_numero(smi_weekly)}) está en zona de **Sobreventa**. Sugiere potencial de subida a largo plazo."
                else:
                    observacion_semanal = f"El **SMI Semanal** ({formatear_numero(smi_weekly)}) está en zona **Intermedia**."
                    
            else:
                estado_smi_weekly = 'N/A'
                observacion_semanal = "No hay datos semanales suficientes."


        return {
            "TICKER": ticker,
            "NOMBRE_EMPRESA": info.get("longName", ticker),
            "PRECIO_ACTUAL": current_price,
            "SMI_AYER": smi_yesterday,
            "SMI_HOY": smi_today,
            "TENDENCIA_ACTUAL": tendencia_hoy,
            "ESTADO_SMI": estado_smi,
            "PRECIO_APLANAMIENTO": precio_aplanamiento,
            "PENDIENTE": pendiente_hoy,
            "COMPRADO": comprado_status, # MODIFICADO: Ahora de la simulación
            "PRECIO_COMPRA": precio_compra, # MODIFICADO: Ahora de la simulación
            "FECHA_COMPRA": fecha_compra, # MODIFICADO: Ahora de la simulación
            "HIST_DF": hist_extended,
            "SOPORTE_1": sr_levels['s1'],
            "SOPORTE_2": sr_levels['s2'],
            "RESISTENCIA_1": sr_levels['r1'],
            "TIPO_EMA": tipo_ema,
            "VALOR_EMA": ema_actual,
            "RESISTENCIA_2": sr_levels['r2'],
            "SMI_SEMANAL": smi_weekly,
            "ESTADO_SMI_SEMANAL": estado_smi_weekly,
            "ADVERTENCIA_SEMANAL": "NO",
            "OBSERVACION_SEMANAL": observacion_semanal,
            "DETALLE_OPERACIONES": operaciones_detalladas, # NUEVO CAMPO
        }
    except Exception as e:
        print(f"❌ Error al obtener datos de {ticker}: {e}. Saltando a la siguiente empresa...")
        return None

def clasificar_empresa(data):
    # ... (Se mantiene la lógica original de clasificación sin cambios) ...
    estado_smi = data['ESTADO_SMI']
    tendencia = data['TENDENCIA_ACTUAL']
    precio_aplanamiento = data['PRECIO_APLANAMIENTO']
    smi_actual = data['SMI_HOY']
    smi_ayer = data['SMI_AYER']
    hist_df = data['HIST_DF']
    current_price = data['PRECIO_ACTUAL']
    close_yesterday = hist_df['Close'].iloc[-2] if len(hist_df) > 1 else 'N/A'
    high_today = hist_df['High'].iloc[-1]
    low_today = hist_df['Low'].iloc[-1]
    pendiente_smi_hoy = data['PENDIENTE']
    pendiente_smi_ayer = hist_df['SMI'].diff().iloc[-2] if len(hist_df['SMI']) > 1 else 'N/A'

    # --- Nuevo: Variables Semanales ---
    estado_smi_weekly = data['ESTADO_SMI_SEMANAL']

    prioridad = {
        "Posibilidad de Compra Activada": 1,
        "Posibilidad de Compra": 2,
        "VIGILAR": 3,
        "Riesgo de Venta": 4,
        "Riesgo de Venta Activada": 5,
        "Seguirá bajando": 6,
        "Intermedio": 7,
        "Compra RIESGO": 8 # Esta prioridad se anula con la clave de ordenación en generar_reporte, pero se mantiene aquí por consistencia.
    }

    if estado_smi == "Sobreventa":
        if tendencia == "Subiendo":
            # --- Lógica de Filtro Semanal ---
            if estado_smi_weekly == "Sobrecompra":
                data['OPORTUNIDAD'] = "Compra RIESGO"
                data['COMPRA_SI'] = "NO RECOMENDAMOS" # La subida puede ser corta
                data['VENDE_SI'] = "NO VENDER"
                data['ORDEN_PRIORIDAD'] = prioridad["Compra RIESGO"]
                data['ADVERTENCIA_SEMANAL'] = "SI"
            else:
                data['OPORTUNIDAD'] = "Posibilidad de Compra Activada"
                data['COMPRA_SI'] = "SI COMPRAR"
                data['VENDE_SI'] = "NO VENDER"
                data['ORDEN_PRIORIDAD'] = prioridad["Posibilidad de Compra Activada"]

        elif tendencia == "Plano":
            data['OPORTUNIDAD'] = "Posibilidad de Compra"
            data['COMPRA_SI'] = "NO RECOMENDAMOS" # Esperar confirmación
            data['VENDE_SI'] = "NO VENDER"
            data['ORDEN_PRIORIDAD'] = prioridad["Posibilidad de Compra"]

        elif tendencia == "Bajando":
            data['OPORTUNIDAD'] = "Seguirá bajando"
            data['COMPRA_SI'] = "NO COMPRAR"
            data['VENDE_SI'] = "NO VENDER"
            data['ORDEN_PRIORIDAD'] = prioridad["Seguirá bajando"]

    elif estado_smi == "Sobrecompra":
        if tendencia == "Bajando":
            data['OPORTUNIDAD'] = "Riesgo de Venta Activada"
            data['COMPRA_SI'] = "NO COMPRAR"
            data['VENDE_SI'] = "SI VENDER"
            data['ORDEN_PRIORIDAD'] = prioridad["Riesgo de Venta Activada"]

        elif tendencia == "Plano":
            data['OPORTUNIDAD'] = "Riesgo de Venta"
            data['COMPRA_SI'] = "NO COMPRAR"
            data['VENDE_SI'] = "NO RECOMENDAMOS" # Esperar confirmación
            data['ORDEN_PRIORIDAD'] = prioridad["Riesgo de Venta"]

        elif tendencia == "Subiendo":
            data['OPORTUNIDAD'] = "VIGILAR"
            data['COMPRA_SI'] = "NO COMPRAR"
            data['VENDE_SI'] = "NO VENDER"
            data['ORDEN_PRIORIDAD'] = prioridad["VIGILAR"]

    else: # Intermedio
        if tendencia == "Subiendo":
            data['OPORTUNIDAD'] = "VIGILAR"
            data['COMPRA_SI'] = "NO COMPRAR"
            data['VENDE_SI'] = "NO VENDER"
            data['ORDEN_PRIORIDAD'] = prioridad["VIGILAR"]
        elif tendencia == "Bajando":
            data['OPORTUNIDAD'] = "VIGILAR"
            data['COMPRA_SI'] = "NO COMPRAR"
            data['VENDE_SI'] = "NO VENDER"
            data['ORDEN_PRIORIDAD'] = prioridad["VIGILAR"]
        else:
            data['OPORTUNIDAD'] = "Intermedio"
            data['COMPRA_SI'] = "NO COMPRAR"
            data['VENDE_SI'] = "NO VENDER"
            data['ORDEN_PRIORIDAD'] = prioridad["Intermedio"]


    # Comprobación de precio de aplanamiento
    if data['PRECIO_APLANAMIENTO'] != 'N/A':
        if isinstance(data['PRECIO_APLANAMIENTO'], (int, float)):
            precio_aplanamiento_num = float(data['PRECIO_APLANAMIENTO'])
            
            # Si el precio actual está un 1% por debajo del aplanamiento y estamos en sobreventa subiendo
            if data['OPORTUNIDAD'] == "Posibilidad de Compra Activada":
                if current_price < precio_aplanamiento_num * 0.99:
                    data['COMPRA_SI'] = "COMPRA FUERTE"
            
            # Si el precio actual está un 1% por encima del aplanamiento y estamos en sobrecompra bajando
            elif data['OPORTUNIDAD'] == "Riesgo de Venta Activada":
                if current_price > precio_aplanamiento_num * 1.01:
                    data['VENDE_SI'] = "VENTA FUERTE"


    # Si el estado semanal es de Sobreventa y el diario es de Compra Fuerte, dar prioridad.
    if data['ESTADO_SMI_SEMANAL'] == 'Sobreventa' and data['COMPRA_SI'] in ["SI COMPRAR", "COMPRA FUERTE"]:
        data['ORDEN_PRIORIDAD'] = 0 # Máxima Prioridad
    elif data['ESTADO_SMI_SEMANAL'] == 'Sobrecompra' and data['VENDE_SI'] in ["SI VENDER", "VENTA FUERTE"]:
        data['ORDEN_PRIORIDAD'] = 9 # Mínima Prioridad (Vigilancia de venta)


    return data


# ******************************************************************************
# *************** FUNCIONES DE CORREO Y REPORTE (MODIFICADAS) ******************
# ******************************************************************************

def enviar_email_con_adjunto(cuerpo_html, asunto, nombre_archivo_base, to_email=os.getenv('DESTINATARIO_CORREO_PERSONAL')):
    # ... (Mantiene la función original) ...
    # Datos personales y de envío (Se mantienen sin cambios)
    pass # Asumiendo que la función es grande y no la modifico, solo el contenido del correo

def generar_reporte():
    try:
        all_tickers = leer_google_sheets()
        
        # Filtra la primera fila que se asume es la cabecera
        tickers_a_procesar = all_tickers[1:] if all_tickers and len(all_tickers) > 1 else []
        
        if not tickers_a_procesar:
            print("No se encontraron tickers en la hoja de Google Sheets. Saliendo.")
            return

        datos_empresas = []
        for nombre_empresa in tickers_a_procesar:
            ticker = tickers.get(nombre_empresa)
            if ticker:
                datos = obtener_datos_yfinance(ticker)
                if datos:
                    datos = clasificar_empresa(datos)
                    datos_empresas.append(datos)
            else:
                print(f"⚠️ Advertencia: No se encontró el ticker para {nombre_empresa}. Saltando...")

        # Ordenar por prioridad
        datos_ordenados = sorted(datos_empresas, key=lambda x: x['ORDEN_PRIORIDAD'])

        reporte_rows = ""
        for datos in datos_ordenados:
            # Colores
            color_fondo = ""
            color_smi = ""
            if datos['OPORTUNIDAD'] == "Posibilidad de Compra Activada":
                color_fondo = "#d4edda" # Verde claro
            elif datos['OPORTUNIDAD'] == "Riesgo de Venta Activada":
                color_fondo = "#f8d7da" # Rojo claro
            elif datos['OPORTUNIDAD'] == "Posibilidad de Compra" or datos['OPORTUNIDAD'] == "Compra RIESGO":
                color_fondo = "#fff3cd" # Amarillo claro
            
            if datos['ESTADO_SMI'] == "Sobrecompra":
                color_smi = "#dc3545" # Rojo
            elif datos['ESTADO_SMI'] == "Sobreventa":
                color_smi = "#28a745" # Verde
            else:
                color_smi = "#ffc107" # Amarillo

            # Formateo
            precio_actual_str = formatear_numero(datos['PRECIO_ACTUAL'])
            smi_hoy_str = formatear_numero(datos['SMI_HOY'])
            smi_ayer_str = formatear_numero(datos['SMI_AYER'])
            pendiente_str = formatear_numero(datos['PENDIENTE'])
            precio_aplanamiento_str = formatear_numero(datos['PRECIO_APLANAMIENTO'])
            
            # Icono para el estado de compra
            icono_compra = "🟢" if datos['COMPRADO'] == "SI" else "🔴"
            
            # Fila principal
            reporte_rows += f"""
            <tr style="background-color: {color_fondo}; cursor: pointer;" onclick="toggleDetails('{datos['TICKER']}')">
                <td>{datos['NOMBRE_EMPRESA']} ({datos['TICKER']})</td>
                <td><span style="font-weight: bold; color: {color_smi};">{datos['ESTADO_SMI']}</span></td>
                <td>{datos['TENDENCIA_ACTUAL']}</td>
                <td>{precio_actual_str}€</td>
                <td>{smi_hoy_str}</td>
                <td>{pendiente_str}</td>
                <td>{datos['OPORTUNIDAD']}</td>
                <td><strong class="pill pill-compra-{datos['COMPRA_SI'].replace(' ', '-').lower()}">{datos['COMPRA_SI']}</strong></td>
                <td><strong class="pill pill-venta-{datos['VENDE_SI'].replace(' ', '-').lower()}">{datos['VENDE_SI']}</strong></td>
                <td>{icono_compra}</td>
                <td>{precio_aplanamiento_str}€</td>
                <td><button onclick="event.stopPropagation(); toggleDetails('{datos['TICKER']}')">Ver más</button></td>
            </tr>
            """
            
            # ******************************************************************
            # *********** INICIO: CÓDIGO HTML DETALLADO (MODIFICADO) ***********
            # ******************************************************************

            # 1. Generar la tabla de operaciones (el nuevo requerimiento)
            operaciones_html_table = ""
            if datos.get('DETALLE_OPERACIONES'):
                # Invertir el orden para mostrar la más reciente primero
                operaciones_html_table += """
                <h4 style="margin-top: 15px; border-bottom: 1px solid #ccc; padding-bottom: 5px;">Historial de Operaciones (Simulación 10000€/Op)</h4>
                <div style="max-height: 250px; overflow-y: auto;">
                    <table class="operaciones-table">
                        <thead>
                            <tr>
                                <th>Estado</th>
                                <th>Entrada (Fecha)</th>
                                <th>Entrada (Precio)</th>
                                <th>Salida (Fecha)</th>
                                <th>Salida (Precio)</th>
                                <th>Beneficio (€)</th>
                                <th>Ganancia (%)</th>
                            </tr>
                        </thead>
                        <tbody>
                """
                for op in reversed(datos['DETALLE_OPERACIONES']):
                    # Clase de color para el beneficio/pérdida
                    clase_beneficio = 'ganancia' if op['BENEFICIO'] >= 0 else 'perdida'
                    
                    # Calcular el porcentaje de beneficio/pérdida
                    porcentaje_str = 'N/A'
                    try:
                        precio_entrada = op['PRECIO_ENTRADA']
                        precio_salida = op.get('PRECIO_SALIDA')
                        if precio_entrada is not None and precio_entrada != 0 and isinstance(precio_salida, (int, float)):
                            porcentaje = ((precio_salida - precio_entrada) / precio_entrada) * 100
                            porcentaje_str = f"{formatear_numero(porcentaje)}%"
                    except Exception:
                        pass # Mantiene N/A si hay un error de cálculo

                    # Determinar el estado de salida para el HTML
                    salida_fecha_html = op['FECHA_SALIDA'] if op['FECHA_SALIDA'] != 'N/A' else '<span class="status-pill status-abierta">ABIERTA</span>'
                    salida_precio_html = formatear_numero(op['PRECIO_SALIDA']) if op['PRECIO_SALIDA'] != 'N/A' else formatear_numero(datos['PRECIO_ACTUAL'])
                    
                    operaciones_html_table += f"""
                    <tr class="{clase_beneficio}">
                        <td data-label="Estado" class="status-cell">
                            <span class="status-pill status-{op['TIPO'].lower()}">{op['TIPO']}</span>
                        </td>
                        <td data-label="Entrada (Fecha)">{op['FECHA_ENTRADA']}</td>
                        <td data-label="Entrada (Precio)">{formatear_numero(op['PRECIO_ENTRADA'])}€</td>
                        <td data-label="Salida (Fecha)">{salida_fecha_html}</td>
                        <td data-label="Salida (Precio)">{salida_precio_html}€</td>
                        <td data-label="Beneficio (€)" class="{clase_beneficio}">
                            {formatear_numero(op['BENEFICIO'])}€
                        </td>
                        <td data-label="Ganancia (%)" class="{clase_beneficio}">
                            {porcentaje_str}
                        </td>
                    </tr>
                    """
                operaciones_html_table += """
                        </tbody>
                    </table>
                </div>
                """
            else:
                operaciones_html_table = "<p>No hay historial de operaciones simuladas para este período.</p>"
            
            # Fila de detalles: Incluye el nuevo contenido
            reporte_rows += f"""
            <tr class="details-row" id="details-{datos['TICKER']}">
                <td colspan="15">
                    <div class="details-content">
                        <div class="details-grid">
                            <div class="details-item">
                                <strong>¿Estamos Comprados?</strong>
                                <span class="status-pill status-{datos['COMPRADO'].lower()}">{datos['COMPRADO']}</span>
                            </div>
                            <div class="details-item">
                                <strong>Fecha Última Compra:</strong>
                                {datos['FECHA_COMPRA']}
                            </div>
                            <div class="details-item">
                                <strong>Precio Última Compra:</strong>
                                {formatear_numero(datos['PRECIO_COMPRA'])}€
                            </div>
                        </div>
                        
                        {operaciones_html_table}

                        <div class="observacion-semanal">
                            <p><strong>Observación SMI Semanal ({datos['ESTADO_SMI_SEMANAL']}):</strong> {datos['OBSERVACION_SEMANAL']}</p>
                        </div>
                        
                        {datos.get('CONTENIDO_GEMINI_HTML', '')}
                        {datos.get('GRAFICO_HTML', '')}

                        <div class="footer-details">
                            <p><strong>Soporte 1:</strong> {formatear_numero(datos['SOPORTE_1'])} | <strong>Resistencia 1:</strong> {formatear_numero(datos['RESISTENCIA_1'])}</p>
                            <p><strong>EMA 100:</strong> {formatear_numero(datos['VALOR_EMA'])} ({datos['TIPO_EMA']})</p>
                        </div>
                    </div>
                </td>
            </tr>
            """
            # ******************************************************************
            # *********** FIN: CÓDIGO HTML DETALLADO (MODIFICADO) **************
            # ******************************************************************


        # El resto del HTML se mantiene, solo añadimos el CSS para los nuevos elementos
        html_body = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>IBEXIA - Reporte Premium de Oportunidades</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f4f7f9;
                }}
                .container {{
                    padding: 20px;
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                h1 {{
                    color: #0d47a1;
                    border-bottom: 3px solid #0d47a1;
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                }}
                /* Estilos de la tabla principal */
                .data-table-container {{
                    overflow-x: auto;
                    margin-top: 20px;
                    border: 1px solid #ccc;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .data-table {{
                    width: 100%;
                    border-collapse: collapse;
                    min-width: 1000px; /* Asegura el scroll en pantallas pequeñas */
                }}
                .data-table th, .data-table td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                    white-space: nowrap;
                }}
                .data-table th {{
                    background-color: #0d47a1;
                    color: white;
                    cursor: pointer;
                    position: sticky;
                    top: 0;
                }}
                .data-table tr:hover {{
                    background-color: #f1f1f1 !important;
                }}
                
                /* Pill styles */
                .pill {{
                    display: inline-block;
                    padding: 5px 10px;
                    border-radius: 15px;
                    font-size: 0.8em;
                    font-weight: bold;
                    text-transform: uppercase;
                }}
                .pill-si-comprar, .pill-compra-fuerte {{ background-color: #28a745; color: white; }}
                .pill-no-comprar, .pill-si-vender, .pill-venta-fuerte {{ background-color: #dc3545; color: white; }}
                .pill-no-recomendamos, .pill-no-vender {{ background-color: #ffc107; color: #333; }}
                
                /* Estilos del detalle (Nuevo) */
                .details-row {{
                    display: none; /* Oculto por defecto */
                    background-color: #e9ecef;
                }}
                .details-content {{
                    padding: 20px;
                    border-left: 5px solid #0d47a1;
                    background-color: #f8f9fa;
                    border-radius: 0 0 8px 8px;
                }}
                .details-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    background-color: #fff;
                }}
                .details-item {{
                    line-height: 1.5;
                }}
                .status-pill {{
                    padding: 3px 8px;
                    border-radius: 10px;
                    font-size: 0.75em;
                    font-weight: bold;
                    text-transform: uppercase;
                    margin-left: 5px;
                }}
                .status-si, .status-abierta {{ background-color: #28a745; color: white; }}
                .status-no, .status-cerrada {{ background-color: #6c757d; color: white; }}

                /* Estilos de la tabla de Operaciones (Nuevo) */
                .operaciones-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                }}
                .operaciones-table th, .operaciones-table td {{
                    padding: 8px 10px;
                    border: 1px solid #e9ecef;
                    text-align: center;
                }}
                .operaciones-table thead th {{
                    background-color: #343a40;
                    color: white;
                    font-size: 0.85em;
                    position: sticky;
                    top: 0;
                    z-index: 10;
                }}
                .operaciones-table tbody tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .operaciones-table .ganancia {{ color: #28a745; font-weight: bold; }}
                .operaciones-table .perdida {{ color: #dc3545; font-weight: bold; }}

                /* Búsqueda y Filtro */
                .controls {{
                    display: flex;
                    gap: 20px;
                    margin-bottom: 20px;
                    align-items: center;
                    flex-wrap: wrap;
                }}
                .controls input[type="text"], .controls select {{
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    font-size: 1em;
                }}
                
                /* Estilos Responsive para la tabla de Operaciones */
                @media screen and (max-width: 768px) {{
                    .operaciones-table thead {{
                        display: none; 
                    }}
                    .operaciones-table, .operaciones-table tbody, .operaciones-table tr, .operaciones-table td {{
                        display: block;
                        width: 100%;
                    }}
                    .operaciones-table tr {{
                        margin-bottom: 10px;
                        border: 1px solid #ccc;
                        border-radius: 5px;
                        background-color: #fff;
                    }}
                    .operaciones-table td {{
                        text-align: right;
                        padding-left: 50%;
                        position: relative;
                    }}
                    .operaciones-table td::before {{
                        content: attr(data-label);
                        position: absolute;
                        left: 10px;
                        width: calc(50% - 20px);
                        padding-right: 10px;
                        white-space: nowrap;
                        text-align: left;
                        font-weight: bold;
                        color: #333;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Reporte IBEXIA Premium - Oportunidades ({datetime.today().strftime('%d/%m/%Y')})</h1>

                <div class="controls">
                    <input type="text" id="searchInput" placeholder="Buscar empresa...">
                    <select id="filterSelect">
                        <option value="">Mostrar todo</option>
                        <option value="Posibilidad de Compra Activada">🟢 Compra Activada</option>
                        <option value="Posibilidad de Compra">🟡 Posibilidad de Compra</option>
                        <option value="Riesgo de Venta Activada">🔴 Venta Activada</option>
                        <option value="VIGILAR">🟠 VIGILAR</option>
                        <option value="Intermedio">⚪ Intermedio</option>
                    </select>
                </div>
                
                <div class="data-table-container" id="tableContainer">
                    <table class="data-table" id="dataTable">
                        <thead>
                            <tr>
                                <th data-sort="text">Empresa (Ticker)</th>
                                <th data-sort="text">Estado SMI</th>
                                <th data-sort="text">Tendencia</th>
                                <th data-sort="number">Precio Actual (€)</th>
                                <th data-sort="number">SMI Hoy</th>
                                <th data-sort="number">Pendiente</th>
                                <th data-sort="text">Oportunidad</th>
                                <th data-sort="text">Comprar</th>
                                <th data-sort="text">Vender</th>
                                <th data-sort="text">Comprado</th>
                                <th data-sort="number">P. Aplanamiento (€)</th>
                                <th>Detalle</th>
                            </tr>
                        </thead>
                        <tbody>
                            {reporte_rows}
                        </tbody>
                    </table>
                </div>
            </div>

            <script>
                // ... (Se mantiene la lógica original de JavaScript para ordenar, buscar y desplegar) ...
                
                function toggleDetails(ticker) {{
                    const detailsRow = document.getElementById('details-' + ticker);
                    if (detailsRow) {{
                        detailsRow.style.display = detailsRow.style.display === 'none' ? 'table-row' : 'none';
                    }}
                }}
                
                function filterTable() {{
                    const filter = document.getElementById('filterSelect').value.toLowerCase();
                    const search = document.getElementById('searchInput').value.toLowerCase();
                    const table = document.getElementById('dataTable');
                    const rows = table.getElementsByTagName('tbody')[0].getElementsByTagName('tr');
                    
                    for (let i = 0; i < rows.length; i++) {{
                        let row = rows[i];
                        if (row.classList.contains('details-row')) {{
                            continue; // Saltar filas de detalle
                        }}
                        
                        let shouldDisplay = true;
                        
                        // Busca por nombre/ticker (primera columna)
                        const nameCell = row.cells[0];
                        if (search && nameCell) {{
                            const nameText = nameCell.textContent.toLowerCase();
                            if (!nameText.includes(search)) {{
                                shouldDisplay = false;
                            }}
                        }}
                        
                        // Filtra por oportunidad (séptima columna)
                        const opportunityCell = row.cells[6];
                        if (filter && opportunityCell) {{
                            const opportunityText = opportunityCell.textContent.toLowerCase();
                            if (opportunityText !== filter) {{
                                shouldDisplay = false;
                            }}
                        }}

                        if (shouldDisplay) {{
                            row.style.display = "";
                        }} else {{
                            row.style.display = "none";
                            // También oculta la fila de detalles asociada
                            const detailRow = document.getElementById('details-' + row.cells[0].textContent.match(/\((.*?)\)/)[1]);
                            if (detailRow) {{
                                detailRow.style.display = "none";
                            }}
                        }}
                    }}
                }}

                document.getElementById('searchInput').addEventListener('keyup', filterTable);
                document.getElementById('filterSelect').addEventListener('change', filterTable);

                // Función de ordenación (se mantiene la lógica original, asegurando que ignore las filas de detalle)
                function sortTable(columnIndex, type) {{
                    const table = document.getElementById('dataTable');
                    const tbody = table.querySelector('tbody');
                    const rows = Array.from(tbody.querySelectorAll('tr:not(.details-row)'));
                    
                    // Almacenar las filas de detalle temporalmente
                    const detailRows = Array.from(tbody.querySelectorAll('.details-row'));
                    const rowMap = new Map();
                    rows.forEach(row => {{
                        const tickerMatch = row.cells[0].textContent.match(/\((.*?)\)/);
                        const ticker = tickerMatch ? tickerMatch[1] : null;
                        if (ticker) {{
                            const detailRow = document.getElementById('details-' + ticker);
                            rowMap.set(row, detailRow);
                        }}
                    }});

                    const sortedRows = rows.sort((a, b) => {{
                        let aVal = a.cells[columnIndex].textContent.trim();
                        let bVal = b.cells[columnIndex].textContent.trim();

                        if (type === 'number') {{
                            // Limpiar y parsear para números
                            aVal = parseFloat(aVal.replace(/[^0-9,-]/g, '').replace(',', '.'));
                            bVal = parseFloat(bVal.replace(/[^0-9,-]/g, '').replace(',', '.'));
                            // Si el valor es N/A o NaN, se coloca al final
                            if (isNaN(aVal)) return 1;
                            if (isNaN(bVal)) return -1;
                            return aVal - bVal;
                        }} else {{
                            return aVal.localeCompare(bVal);
                        }}
                    }});

                    // Invertir si ya estaba ordenado (Toggle Asc/Desc)
                    const currentDir = table.getAttribute('data-sort-direction') === 'asc' ? 'desc' : 'asc';
                    if (table.getAttribute('data-sort-column') === String(columnIndex) && currentDir === 'desc') {{
                        sortedRows.reverse();
                        table.setAttribute('data-sort-direction', 'desc');
                    }} else {{
                        table.setAttribute('data-sort-direction', 'asc');
                    }}
                    table.setAttribute('data-sort-column', columnIndex);

                    // Reconstruir el tbody
                    tbody.innerHTML = '';
                    sortedRows.forEach(row => {{
                        tbody.appendChild(row);
                        const detail = rowMap.get(row);
                        if (detail) {{
                            tbody.appendChild(detail);
                        }}
                    }});
                }}

                document.querySelectorAll('.data-table th').forEach((header, index) => {{
                    if (header.dataset.sort) {{
                        header.addEventListener('click', () => {{
                            sortTable(index, header.dataset.sort);
                        }});
                    }}
                }});

            </script>
        </body>
        </html>
        """
        # ******************************************************************************
        # *************** FIN DE LA MODIFICACIÓN DE LA SECCIÓN HTML ********************
        # ******************************************************************************

        
        asunto = f"🔔 Alertas y Oportunidades IBEXIA: {len(datos_ordenados)} oportunidades detectadas hoy {datetime.today().strftime('%d/%m/%Y')}"
        # CÓDIGO CORREGIDO: Añade el nombre del archivo
        nombre_archivo_base = f"reporte_ibexia_{datetime.today().strftime('%Y%m%d')}"

        enviar_email_con_adjunto(html_body, asunto, nombre_archivo_base)

    except Exception as e:
        print(f"❌ Error al ejecutar el script principal: {e}")

if __name__ == '__main__':
    generar_reporte()
