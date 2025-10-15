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

def formatear_numero(numero):
    if pd.isna(numero) or numero is None:
        return "N/A"
    try:
        num = float(numero)
        return f"{num:,.3f}"
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
        
def calcular_beneficio_perdida(precio_compra, precio_actual, inversion=10000):
    # MODIFICACIÓN PARA SER MÁS ROBUSTO Y DEVOLVER EL BENEFICIO NUMÉRICO O STRING "N/A"
    try:
        precio_compra = float(precio_compra)
        precio_actual = float(precio_actual)
        
        if precio_compra <= 0 or precio_actual <= 0:
            return "N/A"

        acciones = inversion / precio_compra
        beneficio_perdida = (precio_actual - precio_compra) * acciones
        return beneficio_perdida # Devolver el número para la comprobación interna
    except (ValueError, TypeError):
        return "N/A"
        
def formatear_beneficio(beneficio):
    if beneficio == "N/A":
        return "N/A"
    try:
        num = float(beneficio)
        if num >= 0:
            return f"<span style='color:#28a745;'>+{num:,.2f}€</span>" # Verde para ganancias
        else:
            return f"<span style='color:#dc3545;'>{num:,.2f}€</span>" # Rojo para pérdidas
    except (ValueError, TypeError):
        return "N/A"

def obtener_datos_yfinance(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        current_price = info.get("currentPrice")
        if not current_price:
            print(f"⚠️ Advertencia: No se encontró precio actual para {ticker}. Saltando...")
            return None

        # --- Datos Diarios (como estaban) ---
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
        
        # --- Lógica de Detección de Última Operación (Compra o Venta) ---
        comprado_status = "NO"
        precio_compra = "N/A" # Precio de compra de la posición ABIERTA (SI COMPRADO="SI") o CERRADA (SI COMPRADO="NO")
        fecha_compra = "N/A"   # Fecha de compra de la posición ABIERTA (SI COMPRADO="SI") o CERRADA (SI COMPRADO="NO")
        
        precio_venta_cierre = "N/A"
        fecha_venta_cierre = "N/A"
        beneficio_ultima_op = "N/A" # Beneficio de la operación CERRADA
        
        smi_series_copy = hist_extended['SMI'].copy()
        pendientes_smi = smi_series_copy.diff()
        
        # Recorrer hacia atrás buscando el último cruce
        for i in range(len(hist_extended) - 1, 0, -1):
            smi_prev = hist_extended['SMI'].iloc[i - 1]
            pendiente_prev = pendientes_smi.iloc[i - 1]
            pendiente_curr = pendientes_smi.iloc[i]
            
            # Condición de VENTA (Bajando después de subir, sugiere cierre)
            if pendiente_curr < 0 and pendiente_prev >= 0:
                # Si se detecta una señal de venta, significa que la posición anterior (COMPRA) se cierra
                
                precio_venta_cierre = hist_extended['Close'].iloc[i]
                fecha_venta_cierre = hist_extended.index[i].strftime('%d/%m/%Y')

                precio_compra_op_cerrada = "N/A"
                fecha_compra_op_cerrada = "N/A"
                
                # Buscamos la última señal de COMPRA antes de esta VENTA
                for j in range(i - 1, 0, -1):
                    p_curr_compra = pendientes_smi.iloc[j]
                    p_prev_compra = pendientes_smi.iloc[j - 1]
                    smi_prev_compra = hist_extended['SMI'].iloc[j - 1]
                    
                    if p_curr_compra > 0 and p_prev_compra <= 0 and smi_prev_compra < 40:
                        precio_compra_op_cerrada = hist_extended['Close'].iloc[j]
                        fecha_compra_op_cerrada = hist_extended.index[j].strftime('%d/%m/%Y')
                        
                        # Cálculo de Beneficio de la operación CERRADA
                        beneficio_ultima_op = calcular_beneficio_perdida(precio_compra_op_cerrada, precio_venta_cierre)
                        
                        # **CORRECCIÓN DE LÓGICA:**
                        # Se asigna el precio y fecha de la compra de la operación CERRADA
                        # a las variables que el HTML usa para mostrarlas.
                        precio_compra = precio_compra_op_cerrada
                        fecha_compra = fecha_compra_op_cerrada
                        # **FIN DE LA CORRECCIÓN**
                        
                        break
                        
                comprado_status = "NO"
                break
            
            # Condición de COMPRA (Subiendo después de bajar, sugiere apertura)
            elif pendiente_curr > 0 and pendiente_prev <= 0 and smi_prev < 40:
                comprado_status = "SI"
                precio_compra = hist_extended['Close'].iloc[i]
                fecha_compra = hist_extended.index[i].strftime('%d/%m/%Y')
                
                # Al estar COMPRADO, no hay datos de venta/cierre aún, solo el beneficio simulado
                precio_venta_cierre = "N/A"
                fecha_venta_cierre = "N/A"
                beneficio_ultima_op = "N/A"
                break
                
        # --- Cálculo de Beneficio Actual (SI está Comprado) ---
        beneficio_actual = "N/A"
        # Si el estado más reciente es una compra (COMPRADO == "SI"), calculamos el beneficio actual.
        if comprado_status == "SI" and isinstance(precio_compra, (int, float)):
            beneficio_actual = calcular_beneficio_perdida(precio_compra, current_price)


        # --- Modificación: Cálculo de SMI Semanal ---
        hist_weekly = stock.history(period="3y", interval="1wk")
        if hist_weekly.empty:
            smi_weekly = 'N/A'
            estado_smi_weekly = 'N/A'
            # Nuevo campo para el texto de la observación semanal
            observacion_semanal = "No hay datos semanales suficientes."
        else:
            hist_weekly = calculate_smi_tv(hist_weekly)
            smi_weekly_series = hist_weekly['SMI'].dropna()
            smi_weekly = smi_weekly_series.iloc[-1] if not smi_weekly_series.empty else 'N/A'
            
            if isinstance(smi_weekly, (int, float)):
                estado_smi_weekly = "Sobrecompra" if smi_weekly > 40 else ("Sobreventa" if smi_weekly < -40 else "Intermedio")
                
                # Generar el texto de la observación semanal
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
            "COMPRADO": comprado_status,
            "PRECIO_COMPRA": precio_compra,
            "FECHA_COMPRA": fecha_compra,
            "HIST_DF": hist_extended,
            "SOPORTE_1": sr_levels['s1'],
            "SOPORTE_2": sr_levels['s2'],
            "RESISTENCIA_1": sr_levels['r1'],
            "TIPO_EMA": tipo_ema,
            "VALOR_EMA": ema_actual,
            "RESISTENCIA_2": sr_levels['r2'],
            # --- Nuevos Campos Semanales ---
            "SMI_SEMANAL": smi_weekly,
            "ESTADO_SMI_SEMANAL": estado_smi_weekly,
            "ADVERTENCIA_SEMANAL": "NO", # Se inicializa y se modifica en clasificar_empresa
            "OBSERVACION_SEMANAL": observacion_semanal, # Nuevo campo con el texto de la observación semanal
            # --- Nuevos Campos de Operativa ---
            "PRECIO_VENTA_CIERRE": precio_venta_cierre,
            "FECHA_VENTA_CIERRE": fecha_venta_cierre,
            "BENEFICIO_ULTIMA_OP": beneficio_ultima_op, # Beneficio numérico o "N/A"
            "BENEFICIO_ACTUAL": beneficio_actual, # Beneficio numérico o "N/A" (si COMPRADO=SI)
        }

    except Exception as e:
        print(f"❌ Error al obtener datos de {ticker}: {e}. Saltando a la siguiente empresa...")
        return None

def clasificar_empresa(data):
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
                data['COMPRA_SI'] = "COMPRA YA"
                data['VENDE_SI'] = "NO VENDER"
                data['ORDEN_PRIORIDAD'] = prioridad["Posibilidad de Compra Activada"]
            # -----------------------------------
        elif tendencia == "Bajando":
            # --- Lógica de Filtro Semanal ---
            if estado_smi_weekly == "Sobrecompra":
                data['OPORTUNIDAD'] = "Compra RIESGO"
                data['COMPRA_SI'] = "NO RECOMENDAMOS"
                data['VENDE_SI'] = "NO VENDER"
                data['ORDEN_PRIORIDAD'] = prioridad["Compra RIESGO"]
                data['ADVERTENCIA_SEMANAL'] = "SI"
            else:
                data['OPORTUNIDAD'] = "Posibilidad de Compra"
                if current_price > close_yesterday:
                    data['COMPRA_SI'] = "COMPRA YA"
                else:
                    data['COMPRA_SI'] = f"COMPRAR SI SUPERA {formatear_numero(close_yesterday)}€"
                data['VENDE_SI'] = "NO VENDER"
                data['ORDEN_PRIORIDAD'] = prioridad["Posibilidad de Compra"]
            # -----------------------------------
        else:
            data['OPORTUNIDAD'] = "Intermedio"
            data['COMPRA_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['VENDE_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['ORDEN_PRIORIDAD'] = prioridad["Intermedio"]
    
    elif estado_smi == "Intermedio":
        if tendencia == "Bajando":
            data['OPORTUNIDAD'] = "Seguirá bajando"
            data['COMPRA_SI'] = "NO COMPRAR"
            data['VENDE_SI'] = "YA ES TARDE PARA VENDER"
            data['ORDEN_PRIORIDAD'] = prioridad["Seguirá bajando"]
        elif tendencia == "Subiendo":
            data['OPORTUNIDAD'] = "VIGILAR"
            data['COMPRA_SI'] = "NO COMPRAR"
            
            trigger_price = close_yesterday * 0.99
            
            if current_price < trigger_price:
                 data['VENDE_SI'] = "VENDE YA"
            else:
                 data['VENDE_SI'] = f"VENDER SI PIERDE {formatear_numero(trigger_price)}€"
            data['ORDEN_PRIORIDAD'] = prioridad["VIGILAR"]
        else:
            data['OPORTUNIDAD'] = "Intermedio"
            data['COMPRA_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['VENDE_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['ORDEN_PRIORIDAD'] = prioridad["Intermedio"]
            
    elif estado_smi == "Sobrecompra":
        if tendencia == "Subiendo":
            data['OPORTUNIDAD'] = "Riesgo de Venta"
            data['COMPRA_SI'] = "NO COMPRAR"
            data['VENDE_SI'] = f"ZONA DE VENTA<br><span class='small-text'>PRECIO IDEAL VENTA HOY: {high_today:,.2f}€</span>"
            data['ORDEN_PRIORIDAD'] = prioridad["Riesgo de Venta"]
        elif tendencia == "Bajando":
            data['OPORTUNIDAD'] = "Riesgo de Venta Activada"
            data['COMPRA_SI'] = "NO COMPRAR"
            data['VENDE_SI'] = "VENDE AHORA"
            data['ORDEN_PRIORIDAD'] = prioridad["Riesgo de Venta Activada"]
        else:
            data['OPORTUNIDAD'] = "Intermedio"
            data['COMPRA_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['VENDE_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['ORDEN_PRIORIDAD'] = prioridad["Intermedio"]
    
    return data
    
def generar_observaciones(data):
    nombre_empresa = data['NOMBRE_EMPRESA']
    precio_actual = formatear_numero(data['PRECIO_ACTUAL'])
    estado_smi = data['ESTADO_SMI']
    tendencia = data['TENDENCIA_ACTUAL']
    oportunidad = data['OPORTUNIDAD']
    soporte1 = formatear_numero(data['SOPORTE_1'])
    resistencia1 = formatear_numero(data['RESISTENCIA_1'])
    compra_si = data['COMPRA_SI']
    vende_si = data['VENDE_SI']
    tipo_ema = data['TIPO_EMA']
    valor_ema = formatear_numero(data['VALOR_EMA'])
    
    # --- Nuevo: Advertencia Semanal ---
    advertencia_semanal = data['ADVERTENCIA_SEMANAL']

    texto_observacion = f"<strong>Observaciones de {nombre_empresa}:</strong><br>"
    
    # Nuevo texto de advertencia para insertar al inicio
    advertencia_texto = ""
    if advertencia_semanal == "SI":
        advertencia_texto = "<strong style='color:#ffc107;'>ADVERTENCIA SEMANAL: El SMI semanal está en zona de sobrecompra. No se recomienda comprar ya que la subida podría ser muy corta y con alto riesgo.</strong><br>"


    if oportunidad == "Posibilidad de Compra Activada":
        texto = f"El algoritmo se encuentra en una zona de sobreventa y muestra una tendencia alcista en sus últimos valores, lo que activa una señal de compra fuerte. Se recomienda tener en cuenta los niveles de resistencia ({resistencia1}€) para determinar un objetivo de precio. La EMA de 100 periodos se encuentra en {valor_ema}€, actuando como un nivel de {tipo_ema}."
    
    elif oportunidad == "Posibilidad de Compra":
        if "COMPRA YA" in compra_si:
            texto = f"El algoritmo detecta que el valor está en una zona de sobreventa, lo que puede ser un indicador de reversión. El algoritmo ha detectado una oportunidad de compra inmediata para aprovechar un posible rebote.La EMA de 100 periodos se encuentra en {valor_ema}€, actuando como un nivel de {tipo_ema}."
        else:
            texto = f"El algoritmo detecta que el valor está en una zona de sobreventa con una tendencia bajista. Se ha detectado una oportunidad de {compra_si} para un posible rebote. La EMA de 100 periodos se encuentra en {valor_ema}€, actuando como un nivel de {tipo_ema}."
    
    # Nuevo bloque de Riesgo de Compra
    elif oportunidad == "Compra RIESGO":
        texto = f"El algoritmo detectó una señal de compra diaria, pero el **SMI Semanal** se encuentra en zona de **Sobrecompra** ({formatear_numero(data['SMI_SEMANAL'])}). Esto indica que el precio ya ha subido mucho a largo plazo, y la señal de rebote diaria podría ser muy breve. No se recomienda la compra en este momento. La EMA de 100 periodos se encuentra en {valor_ema}€, actuando como un nivel de {tipo_ema}."

    elif oportunidad == "VIGILAR":
        texto = f"El algoritmo se encuentra en una zona intermedia y muestra una tendencia alcista en sus últimos valores. Se sugiere vigilar de cerca, ya que una caída en el precio podría ser una señal de venta. {vende_si}. Se recomienda tener en cuenta los niveles de soporte ({soporte1}€) para saber hasta dónde podría bajar el precio. La EMA de 100 periodos se encuentra en {valor_ema}€, actuando como un nivel de {tipo_ema}."
    
    elif oportunidad == "Riesgo de Venta":
        texto = f"El algoritmo ha entrado en una zona de sobrecompra. Esto genera un riesgo de venta. Se recomienda tener en cuenta los niveles de soporte ({soporte1}€) para saber hasta dónde podría bajar el precio. La EMA de 100 periodos se encuentra en {valor_ema}€, actuando como un nivel de {tipo_ema}."
    
    elif oportunidad == "Riesgo de Venta Activada":
        texto = f"La combinación de una zona de sobrecompra y una tendencia bajista en el algoritmo ha activado una señal de riesgo de venta. Se recomienda tener en cuenta los niveles de soporte ({soporte1}€) para saber hasta dónde podría bajar el precio. La EMA de 100 periodos se encuentra en {valor_ema}€, actuando como un nivel de {tipo_ema}."

    elif oportunidad == "Seguirá bajando":
        texto = f"El algoritmo sugiere que es probable que el precio siga bajando en el corto plazo. No se aconseja ni comprar ni vender. Se recomienda observar los niveles de soporte ({soporte1}€). La EMA de 100 periodos se encuentra en {valor_ema}€, actuando como un nivel de {tipo_ema}."

    elif oportunidad == "Intermedio":
        texto = "El algoritmo no emite recomendaciones de compra o venta en este momento, por lo que lo más prudente es mantenerse al margen. La EMA de 100 periodos se encuentra en {valor_ema}€, actuando como un nivel de {tipo_ema}."
    
    else:
        texto = "El algoritmo se encuentra en una zona de sobreventa y muestra una tendencia alcista en sus últimos valores, lo que activa una señal de compra fuerte. Se recomienda comprar para aprovechar un posible rebote, con un objetivo de precio en la zona de resistencia. La EMA de 100 periodos se encuentra en {valor_ema}€, actuando como un nivel de {tipo_ema}."
    
    # Se añade la advertencia al inicio del texto de la observación
    return f'<p style="text-align:left; color:#000;">{texto_observacion.strip()}{advertencia_texto}{texto.strip()}</p>'


def enviar_email_con_adjunto(texto_generado, asunto_email, nombre_archivo):
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

def generar_reporte():
    try:
        all_tickers = leer_google_sheets()[1:]
        if not all_tickers:
            print("No hay tickers para procesar.")
            return

        datos_completos = []
        for ticker in all_tickers:
            print(f"🔎 Analizando {ticker}...")
            
            try:
                data = obtener_datos_yfinance(ticker)
                if data:
                    datos_completos.append(clasificar_empresa(data))
            except Exception as e:
                print(f"❌ Error al procesar {ticker}: {e}. Saltando a la siguiente empresa...")
                continue
                
            time.sleep(1)

        # --- Lógica de ordenación MODIFICADA para mover "Compra RIESGO" arriba ---
        def obtener_clave_ordenacion(empresa):
            categoria = empresa['OPORTUNIDAD']
            
            # Se ajustan las prioridades para que "Compra RIESGO" esté en el grupo de compra (valores < 3)
            # y darle una prioridad interna de 2.5, justo después de las compras fuertes (1 y 2).
            
            prioridad = {
                "Posibilidad de Compra Activada": 1, # Máxima prioridad de compra
                "Posibilidad de Compra": 2,         # Segunda prioridad de compra
                "Compra RIESGO": 2.5,               # TERCERA prioridad, pero aún en el grupo de Compra.
                "VIGILAR": 3,
                "Riesgo de Venta": 4,
                "Riesgo de Venta Activada": 5,
                "Seguirá bajando": 6,
                "Intermedio": 7,
            }

            orden_interna = prioridad.get(categoria, 99) # Si no está, al final

            return (orden_interna, empresa['NOMBRE_EMPRESA']) # Se ordena por la clave y luego por nombre

        datos_ordenados = sorted(datos_completos, key=obtener_clave_ordenacion)
        
        # --- Fin de la lógica de ordenación MODIFICADA ---
        now_utc = datetime.utcnow()
        hora_actual = (now_utc + timedelta(hours=2)).strftime('%H:%M')
        
        # ******************************************************************************
        # ******************** MODIFICACIÓN DE LA SECCIÓN HTML *************************
        # ******************************************************************************
        html_body = f"""
        <html>
        <head>
            <title>Resumen Diario de Oportunidades - {datetime.today().strftime('%d/%m/%Y')} {hora_actual}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-color: #f8f9fa;
                    margin: 0;
                    padding: 0; /* Cambio: Eliminamos padding para centrar mejor */
                    display: flex;
                    justify-content: center; /* Centrar horizontalmente */
                    align-items: flex-start; /* Alinear arriba */
                    min-height: 100vh; /* Ocupar toda la altura de la ventana */
                }}
                .main-container {{
                    max-width: 1200px;
                    width: 95%; /* Asegurar que ocupe espacio */
                    margin: 20px auto; /* Cambio: Margen superior para el centrado */
                    background-color: #ffffff;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
                }}
                h2 {{
                    color: #343a40;
                    text-align: center;
                    font-size: 1.5em;
                    margin-bottom: 30px; /* Aumentar margen */
                }}
                p {{
                    color: #6c757d;
                    text-align: center;
                    font-size: 0.9em;
                }}
                
                /* ESTILO DEL CAMPO DE BÚSQUEDA TIPO GOOGLE */
                #search-container {{
                    display: flex; /* Para centrar el input */
                    flex-direction: column;
                    align-items: center;
                    margin-bottom: 50px; /* Más espacio */
                }}
                #searchInput {{
                    width: 70%; /* Cambio: Más ancho */
                    max-width: 600px;
                    padding: 15px 20px; /* Cambio: Más padding, más grande */
                    font-size: 1.2em; /* Cambio: Fuente más grande */
                    border: 1px solid #ced4da;
                    border-radius: 24px; /* Bordes redondeados */
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); /* Sombra para efecto 3D */
                    box-sizing: border-box;
                    transition: box-shadow 0.3s ease-in-out;
                    text-align: center;
                }}
                #searchInput:focus {{
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2); /* Sombra al enfocar */
                    outline: none;
                }}
                /* FIN DEL ESTILO DEL CAMPO DE BÚSQUEDA TIPO GOOGLE */

                .table-container {{
                    overflow-x: auto;
                    overflow-y: auto;
                    height: 70vh;
                    position: relative;
                    /* Cambio: Ocultar la tabla al inicio */
                    display: none; 
                }}
                table {{
                    width: 100%;
                    table-layout: fixed;
                    margin: 10px auto 0 auto;
                    border-collapse: collapse;
                    font-size: 0.85em;
                }}
                th, td {{
                    border: 1px solid #e9ecef;
                    padding: 6px;
                    text-align: center;
                    vertical-align: middle;
                    white-space: normal;
                    line-height: 1.2;
                }}
                th {{
                    background-color: #e9ecef;
                    color: #495057;
                    font-weight: 600;
                    position: sticky;
                    top: 0;
                    z-index: 10;
                    white-space: nowrap;
                }}
                .compra {{ color: #28a745; font-weight: bold; }}
                .venta {{ color: #dc3545; font-weight: bold; }}
                .riesgo-compra {{ color: #ffc107; font-weight: bold; }} /* Nuevo estilo para Compra RIESGO */
                .comprado-si {{ background-color: #28a745; color: white; font-weight: bold; }}
                .bg-green {{ background-color: #d4edda; color: #155724; }}
                .bg-red {{ background-color: #f8d7da; color: #721c24; }}
                .bg-highlight {{ background-color: #28a745; color: white; font-weight: bold; }}
                .text-center {{ text-align: center; }}
                .disclaimer {{ font-size: 0.8em; text-align: center; color: #6c757d; margin-top: 15px; }}
                .small-text {{ font-size: 0.7em; color: #6c757d; }}
                .green-cell {{ background-color: #d4edda; }}
                .red-cell {{ background-color: #f8d7da; }}
                .yellow-cell {{ background-color: #fff3cd; }} /* Nuevo estilo para celda de riesgo */
                .separator-row td {{ background-color: #e9ecef; height: 3px; padding: 0; border: none; }}
                .category-header td {{
                    background-color: #495057;
                    color: white;
                    font-size: 1.1em;
                    font-weight: bold;
                    text-align: center;
                    padding: 10px;
                    border: none;
                    /* Cambio: Ocultar al inicio */
                    display: none;
                }}
                .observaciones-row td {{
                    background-color: #f9f9f9;
                    text-align: left;
                    font-size: 0.8em;
                    border: 1px solid #e9ecef;
                }}
                .stacked-text {{
                    line-height: 1.2;
                    font-size: 0.8em;
                }}
                .vigilar {{ color: #ffc107; font-weight: bold; }}
                
                .collapsible-row {{
                    display: none;
                }}
                .expand-button {{
                    cursor: pointer;
                    color: #007bff;
                    font-weight: bold;
                    text-decoration: underline;
                }}
                .position-box {{
                    border: 1px solid #dee2e6;
                    border-radius: 6px;
                    padding: 10px;
                    margin: 5px 0;
                    text-align: left;
                    background-color: #ffffff;
                }}
            </style>
        </head>
        <body>
            <div class="main-container">
                <h2 class="text-center">Resumen Diario de Oportunidades</h2>
                
                <div id="search-container">
                    <input type="text" id="searchInput" placeholder="Buscar empresa por nombre o ticker (Ej: Inditex, SAN.MC)...">
                </div>
                
                <div id="scroll-top" style="overflow-x: auto; display: none;">
                    <div style="min-width: 1400px;">&nbsp;</div>
                </div>
                
                <div class="table-container" id="tableContainer">
                    <table id="myTable">
                        <thead>
                            <tr>
                                <th>Empresa (Precio)</th>
                                <th>Tendencia Actual</th>
                                <th>Oportunidad</th>
                                <th>Compra si...</th>
                                <th>Vende si...</th>
                                <th>Análisis detallado</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        if not datos_ordenados:
            html_body += """
                            <tr><td colspan="6">No se encontraron empresas con datos válidos hoy.</td></tr>
            """
        else:
            previous_orden_grupo = None
            for i, data in enumerate(datos_ordenados):
                
                current_orden_grupo = obtener_clave_ordenacion(data)[0]
                
                # Lógica para determinar el encabezado de categoría
                es_primera_fila = previous_orden_grupo is None
                es_cambio_grupo = current_orden_grupo != previous_orden_grupo
                
                if es_primera_fila or es_cambio_grupo:
                    
                    # MODIFICACIÓN DE LA LÓGICA DE ENCABEZADO
                    if current_orden_grupo in [1, 2, 2.5]: # Grupo de Compra (incluye Compra RIESGO con 2.5)
                        if previous_orden_grupo is None or previous_orden_grupo not in [1, 2, 2.5]:
                            # Se añade la clase 'category-header-compra' para un control más fino en JS
                            html_body += """
                                <tr class="category-header category-header-compra"><td colspan="6">OPORTUNIDADES DE COMPRA</td></tr>
                            """
                    elif current_orden_grupo in [3, 4, 5]: # Grupo de Venta/Vigilancia
                        if previous_orden_grupo is None or previous_orden_grupo not in [3, 4, 5]:
                            # Se añade la clase 'category-header-vigilar'
                            html_body += """
                                <tr class="category-header category-header-vigilar"><td colspan="6">ATENTOS A VENDER/VIGILANCIA</td></tr>
                            """
                    elif current_orden_grupo in [6, 7]: # Grupo Intermedio
                        if previous_orden_grupo is None or previous_orden_grupo not in [6, 7]:
                            # Se añade la clase 'category-header-intermedio'
                            html_body += """
                                <tr class="category-header category-header-intermedio"><td colspan="6">OTRAS EMPRESAS SIN MOVIMIENTOS</td></tr>
                            """
                            
                    # Poner un separador si no es la primera fila y hay cambio de grupo
                    if not es_primera_fila and es_cambio_grupo:
                        html_body += """
                            <tr class="separator-row"><td colspan="6"></td></tr>
                        """

                # Lógica de corrección para el enlace
                nombre_empresa_url = None
                for nombre, ticker_val in tickers.items():
                    if ticker_val == data['TICKER']:
                        nombre_empresa_url = nombre
                        break
                
                if nombre_empresa_url:
                    empresa_link = f'https://ibexia.es/category/{nombre_empresa_url.lower()}/'
                else:
                    empresa_link = '#'
                
                nombre_con_precio = f"<a href='{empresa_link}' target='_blank' style='text-decoration:none; color:inherit;'><div class='stacked-text'><b>{data['NOMBRE_EMPRESA']}</b><br>({formatear_numero(data['PRECIO_ACTUAL'])}€)</div></a>"

                # Ajuste de clases para el estado 'Compra RIESGO' (texto amarillo, celda amarilla)
                if "compra" in data['OPORTUNIDAD'].lower() and "riesgo" not in data['OPORTUNIDAD'].lower():
                    clase_oportunidad = "compra"
                    celda_empresa_class = "green-cell"
                elif "venta" in data['OPORTUNIDAD'].lower():
                    clase_oportunidad = "venta"
                    celda_empresa_class = "red-cell"
                elif "vigilar" in data['OPORTUNIDAD'].lower():
                    clase_oportunidad = "vigilar"
                    celda_empresa_class = ""
                elif "riesgo" in data['OPORTUNIDAD'].lower():
                    clase_oportunidad = "riesgo-compra"
                    celda_empresa_class = "yellow-cell"
                else:
                    clase_oportunidad = ""
                    celda_empresa_class = ""
                
                
                observaciones = generar_observaciones(data)
                
                # --- NUEVA SECCIÓN DE DATOS DE OPERATIVA (DENTRO DEL COLLAPSIBLE) ---
                
                # Bloque de la última operación CERRADA (Venta)
                if data['BENEFICIO_ULTIMA_OP'] != "N/A":
                    ultima_op_html = f"""
                        <div class='position-box' style='background-color: #e9ecef;'>
                            <b>ÚLTIMA OPERACIÓN CERRADA:</b>
                            <ul>
                                <li><strong>Compra:</strong> {formatear_numero(data['PRECIO_COMPRA'])}€ ({data['FECHA_COMPRA']})</li>
                                <li><strong>Venta/Cierre:</strong> {formatear_numero(data['PRECIO_VENTA_CIERRE'])}€ ({data['FECHA_VENTA_CIERRE']})</li>
                                <li><strong>Beneficio Cerrado:</strong> {formatear_beneficio(data['BENEFICIO_ULTIMA_OP'])} (Base 10.000€)</li>
                            </ul>
                        </div>
                    """
                else:
                    ultima_op_html = "<p style='font-size:0.9em; margin-top:5px;'>No se detectó un ciclo completo (Compra->Venta) reciente.</p>"

                # Bloque de Posición ABIERTA (Compra Actual)
                if data['COMPRADO'] == "SI":
                    posicion_actual_html = f"""
                        <div class='position-box' style='border: 2px solid #28a745; background-color: #d4edda;'>
                            <b>POSICIÓN ACTUAL: COMPRADO (✅)</b>
                            <ul>
                                <li><strong>Precio de Entrada:</strong> {formatear_numero(data['PRECIO_COMPRA'])}€ ({data['FECHA_COMPRA']})</li>
                                <li><strong>Precio Actual:</strong> {formatear_numero(data['PRECIO_ACTUAL'])}€</li>
                                <li><strong>Beneficio Actual (Simulado):</strong> {formatear_beneficio(data['BENEFICIO_ACTUAL'])} (Base 10.000€)</li>
                            </ul>
                        </div>
                    """
                else:
                    posicion_actual_html = """
                        <div class='position-box' style='border: 2px solid #dc3545; background-color: #f8d7da;'>
                            <b>POSICIÓN ACTUAL: NO COMPRADO (❌)</b>
                        </div>
                    """
                
                
                # --- FILAS DE REPORTE CON OBSERVACIÓN SEMANAL EN DETALLE ---
                html_body += f"""
                            <tr class="main-row" data-index="{i}" data-name="{data['NOMBRE_EMPRESA'].upper()}" data-ticker="{data['TICKER'].upper()}">
                                <td class="{celda_empresa_class}">{nombre_con_precio}</td>
                                <td>{data['TENDENCIA_ACTUAL']}</td>
                                <td class="{clase_oportunidad}">{data['OPORTUNIDAD']}</td>
                                <td>{data['COMPRA_SI']}</td>
                                <td>{data['VENDE_SI']}</td>
                                <td><span class="expand-button" onclick="toggleDetails({i})">Ver más...</span></td>
                            </tr>
                            <tr class="collapsible-row detailed-row-{i}">
                                <td colspan="6">
                                    <div style="display:flex; justify-content:space-between; align-items:flex-start; padding: 10px; flex-wrap: wrap;">
                                        
                                        <div style="flex-basis: 30%; text-align:left; padding-right: 10px;">
                                            {posicion_actual_html}
                                        </div>

                                        <div style="flex-basis: 30%; text-align:left; padding-right: 10px;">
                                            {ultima_op_html}
                                        </div>
                                        
                                        <div style="flex-basis: 30%; text-align:left; font-size:0.9em;">
                                            <div class='position-box'>
                                                <b>EMA/SR</b><br>
                                                EMA ({data['TIPO_EMA']}): <span style="font-weight:bold;">{formatear_numero(data['VALOR_EMA'])}€</span><br>
                                                S1/R1: {formatear_numero(data['SOPORTE_1'])}€ / {formatear_numero(data['RESISTENCIA_1'])}€
                                            </div>
                                            <div class='position-box' style="margin-top:10px;">
                                                <b>Análisis Semanal (SMI)</b><br>
                                                {data['OBSERVACION_SEMANAL']}
                                            </div>
                                        </div>
                                        
                                    </div>
                                </td>
                            </tr>
                            <tr class="observaciones-row detailed-row-{i}">
                                <td colspan="6">{observaciones}</td>
                            </tr>
                """
                previous_orden_grupo = current_orden_grupo
        
        html_body += """
                        </tbody>
                    </table>
                </div>
                
                <br>
                <p class="disclaimer"><strong>Aviso:</strong> El algoritmo de trading se basa en indicadores técnicos y no garantiza la rentabilidad. Utiliza esta información con tu propio análisis y criterio. ¡Feliz trading!</p>
            </div>

            <script>
                // Se utiliza una variable global o una referencia de cierre para el temporizador
                let filterTimeout;
                const tableContainer = document.getElementById("tableContainer");
                const scrollTop = document.getElementById('scroll-top');
                const searchInput = document.getElementById("searchInput");
                const table = document.getElementById("myTable");
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.getElementsByTagName("tr"));

                // Función de filtrado
                function filterTable() {
                    clearTimeout(filterTimeout); // Limpiar el temporizador anterior
                    
                    filterTimeout = setTimeout(() => {
                        const filter = searchInput.value.toUpperCase().trim();
                        const showTable = filter.length > 0;
                        
                        // Mostrar u ocultar la tabla y el scroll superior
                        tableContainer.style.display = showTable ? "block" : "none";
                        if (scrollTop) {
                            scrollTop.style.display = showTable ? "block" : "none";
                        }
                        
                        // Si no hay filtro, salimos
                        if (!showTable) {
                            // Al no haber filtro, todos los elementos están ocultos por el CSS inicial.
                            return; 
                        }

                        let lastCategoryDisplayed = null;

                        for (let i = 0; i < rows.length; i++) {
                            const row = rows[i];
                            
                            // 1. Manejar Separadores
                            if (row.classList.contains("separator-row")) {
                                row.style.display = "none";
                                continue;
                            }

                            // 2. Manejar Filas de Categoría: inicialmente se ocultan por CSS y se muestran si su grupo tiene filas visibles
                            if (row.classList.contains("category-header")) {
                                row.style.display = "none";
                                continue;
                            }

                            // 3. Manejar Filas Detalle/Observaciones: se mantienen ocultas
                            if (row.classList.contains("collapsible-row") || row.classList.contains("observaciones-row")) {
                                row.style.display = "none";
                                continue;
                            }
                            
                            // 4. Procesar Filas Principales
                            if (row.classList.contains("main-row")) {
                                const name = row.getAttribute('data-name');
                                const ticker = row.getAttribute('data-ticker');
                                
                                const isMatch = (name.indexOf(filter) > -1) || (ticker.indexOf(filter) > -1);

                                if (isMatch) {
                                    row.style.display = "table-row";
                                    // Marcar que esta categoría debe mostrarse (se procesará después del loop)
                                    const currentCategory = row.previousElementSibling;
                                    
                                    if (currentCategory && currentCategory.classList.contains("category-header")) {
                                        lastCategoryDisplayed = currentCategory;
                                    }

                                } else {
                                    row.style.display = "none";
                                }
                            }
                        }
                        
                        // 5. Segunda pasada para mostrar las cabeceras de categoría si tienen al menos una fila visible
                        const categoryHeaders = document.querySelectorAll('.category-header');
                        categoryHeaders.forEach(header => {
                            let nextSibling = header.nextElementSibling;
                            let hasVisibleRows = false;
                            while(nextSibling && !nextSibling.classList.contains('category-header')) {
                                if (nextSibling.classList.contains('main-row') && nextSibling.style.display !== 'none') {
                                    hasVisibleRows = true;
                                    break;
                                }
                                nextSibling = nextSibling.nextElementSibling;
                            }
                            header.style.display = hasVisibleRows ? "table-row" : "none";
                        });

                        // 6. Tercera pasada para mostrar los separadores si hay un cambio de categoría visible
                        const separatorRows = document.querySelectorAll('.separator-row');
                        separatorRows.forEach(separator => {
                            const prev = separator.previousElementSibling;
                            const next = separator.nextElementSibling;
                            
                            const prevVisible = prev && prev.style.display === "table-row" && prev.classList.contains("category-header");
                            const nextVisible = next && next.style.display === "table-row" && next.classList.contains("category-header");

                            // Si el separador está entre dos categorías *visibles* diferentes, lo mostramos.
                            if (prevVisible && nextVisible) {
                                separator.style.display = "table-row";
                            } else {
                                separator.style.display = "none";
                            }
                        });


                    }, 200); // Pequeño retraso para evitar ejecuciones rápidas
                }
                
                // Función de acordeón para las filas individuales
                function toggleDetails(index) {
                    // Seleccionar la fila detallada (collapsible) y la fila de observaciones. Ambas usan la clase 'detailed-row-{index}'
                    var detailedRows = document.querySelectorAll('.detailed-row-' + index);
                    
                    detailedRows.forEach(row => {
                        // El estilo inicial es 'display: none;', se usa 'table-row' para que se muestre correctamente en la tabla.
                        if (row) {
                            row.style.display = row.style.display === "table-row" ? "none" : "table-row";
                        }
                    });
                }
                
                // Asegurar que el script se ejecute cuando el DOM esté listo
                document.addEventListener('DOMContentLoaded', function() {
                    if (searchInput) {
                        searchInput.addEventListener("keyup", filterTable);
                        searchInput.focus(); // Enfocar el campo de búsqueda al cargar
                    }
                    
                    // Sincronizar el scroll lateral
                    if (tableContainer && scrollTop) {
                        scrollTop.addEventListener('scroll', () => {
                            tableContainer.scrollLeft = scrollTop.scrollLeft;
                        });
                        
                        tableContainer.addEventListener('scroll', () => {
                            scrollTop.scrollLeft = tableContainer.scrollLeft;
                        });
                    }
                });
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
        # Nota: La excepción de Google Sheets por falta de variables de entorno (GOOGLE_APPLICATION_CREDENTIALS) 
        # sigue activa en leer_google_sheets().
        print(f"❌ Error al ejecutar el script principal: {e}")

if __name__ == '__main__':
    generar_reporte()
