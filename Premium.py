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
        precio_compra = "N/A" 
        fecha_compra = "N/A"   
        
        precio_venta_cierre = "N/A"
        fecha_venta_cierre = "N/A"
        beneficio_ultima_op = "N/A" 
        
        smi_series_copy = hist_extended['SMI'].copy()
        pendientes_smi = smi_series_copy.diff()
        
        # Recorrer hacia atrás buscando el último cruce
        for i in range(len(hist_extended) - 1, 0, -1):
            smi_prev = hist_extended['SMI'].iloc[i - 1]
            pendiente_prev = pendientes_smi.iloc[i - 1]
            pendiente_curr = pendientes_smi.iloc[i]
            
            # Condición de VENTA (Bajando después de subir, sugiere cierre) - Se detecta en el índice 'i'
            if pendiente_curr < 0 and pendiente_prev >= 0:
                
                # --- CORRECCIÓN: Usar precio y fecha del día ANTERIOR (i-1) al cambio de pendiente ---
                precio_venta_cierre = hist_extended['Close'].iloc[i-1]
                fecha_venta_cierre = hist_extended.index[i-1].strftime('%d/%m/%Y')
                # -------------------------------------------------------------------------------------

                precio_compra_op_cerrada = "N/A"
                fecha_compra_op_cerrada = "N/A"
                
                # Buscamos la última señal de COMPRA antes de esta VENTA
                for j in range(i - 1, 0, -1):
                    p_curr_compra = pendientes_smi.iloc[j]
                    p_prev_compra = pendientes_smi.iloc[j - 1]
                    smi_prev_compra = hist_extended['SMI'].iloc[j - 1]
                    
                    # Condición de COMPRA (Subiendo después de bajar) - Se detecta en el índice 'j'
                    if p_curr_compra > 0 and p_prev_compra <= 0 and smi_prev_compra < 40:
                        
                        # --- CORRECCIÓN: Usar precio y fecha del día ANTERIOR (j-1) al cambio de pendiente ---
                        precio_compra_op_cerrada = hist_extended['Close'].iloc[j-1]
                        fecha_compra_op_cerrada = hist_extended.index[j-1].strftime('%d/%m/%Y')
                        # -------------------------------------------------------------------------------------
                        
                        # Cálculo de Beneficio de la operación CERRADA
                        beneficio_ultima_op = calcular_beneficio_perdida(precio_compra_op_cerrada, precio_venta_cierre)
                        
                        # Se asigna el precio y fecha de la compra de la operación CERRADA
                        precio_compra = precio_compra_op_cerrada
                        fecha_compra = fecha_compra_op_cerrada
                        
                        break
                        
                comprado_status = "NO"
                break
            
            # Condición de COMPRA (Subiendo después de bajar, sugiere apertura) - Se detecta en el índice 'i'
            elif pendiente_curr > 0 and pendiente_prev <= 0 and smi_prev < 40:
                comprado_status = "SI"
                
                # --- CORRECCIÓN: Usar precio y fecha del día ANTERIOR (i-1) al cambio de pendiente ---
                precio_compra = hist_extended['Close'].iloc[i-1]
                fecha_compra = hist_extended.index[i-1].strftime('%d/%m/%Y')
                # -------------------------------------------------------------------------------------
                
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
                    observacion_semanal = f"El **indicador Semanal** ({formatear_numero(smi_weekly)}) está en zona de **Sobrecompra**. Sugiere que el precio ya ha subido mucho a largo plazo."
                elif estado_smi_weekly == "Sobreventa":
                    observacion_semanal = f"El **indicador Semanal** ({formatear_numero(smi_weekly)}) está en zona de **Sobreventa**. Sugiere potencial de subida a largo plazo."
                else:
                    observacion_semanal = f"El **indicador Semanal** ({formatear_numero(smi_weekly)}) está en zona **Intermedia**."
                    
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
            "ADVERTENCIA_SEMANAL": "NO", 
            "OBSERVACION_SEMANAL": observacion_semanal, 
            # --- Nuevos Campos de Operativa ---
            "PRECIO_VENTA_CIERRE": precio_venta_cierre,
            "FECHA_VENTA_CIERRE": fecha_venta_cierre,
            "BENEFICIO_ULTIMA_OP": beneficio_ultima_op,
            "BENEFICIO_ACTUAL": beneficio_actual, 
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
    
    # --- Sustitución de 'SMI' por 'algoritmo' o 'indicador' ---
    texto_observacion = f"<strong>Observaciones de {nombre_empresa}:</strong><br>"
    
    # Nuevo texto de advertencia para insertar al inicio
    advertencia_texto = ""
    if advertencia_semanal == "SI":
        advertencia_texto = "<strong style='color:#ffc107;'>ADVERTENCIA SEMANAL: El indicador semanal está en zona de sobrecompra. No se recomienda comprar ya que la subida podría ser muy corta y con alto riesgo.</strong><br>"


    if oportunidad == "Posibilidad de Compra Activada":
        texto = f"El algoritmo se encuentra en una zona de sobreventa y muestra una tendencia alcista en sus últimos valores, lo que activa una señal de compra fuerte. Se recomienda tener en cuenta los niveles de resistencia ({resistencia1}€) para determinar un objetivo de precio. La EMA de 100 periodos se encuentra en {valor_ema}€, actuando como un nivel de {tipo_ema}."
    
    elif oportunidad == "Posibilidad de Compra":
        if "COMPRA YA" in compra_si:
            texto = f"El algoritmo detecta que el valor está en una zona de sobreventa, lo que puede ser un indicador de reversión. El algoritmo ha detectado una oportunidad de compra inmediata para aprovechar un posible rebote.La EMA de 100 periodos se encuentra en {valor_ema}€, actuando como un nivel de {tipo_ema}."
        else:
            texto = f"El algoritmo detecta que el valor está en una zona de sobreventa con una tendencia bajista. Se ha detectado una oportunidad de {compra_si} para un posible rebote. La EMA de 100 periodos se encuentra en {valor_ema}€, actuando como un nivel de {tipo_ema}."
    
    # Nuevo bloque de Riesgo de Compra
    elif oportunidad == "Compra RIESGO":
        texto = f"El algoritmo detectó una señal de compra diaria, pero el **indicador Semanal** se encuentra en zona de **Sobrecompra** ({formatear_numero(data['SMI_SEMANAL'])}). Esto indica que el precio ya ha subido mucho a largo plazo, y la señal de rebote diaria podría ser muy breve. No se recomienda la compra en este momento. La EMA de 100 periodos se encuentra en {valor_ema}€, actuando como un nivel de {tipo_ema}."

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
        texto = f"El algoritmo se encuentra en una zona de sobreventa y muestra una tendencia alcista en sus últimos valores, lo que activa una señal de compra fuerte. Se recomienda comprar para aprovechar un posible rebote, con un objetivo de precio en la zona de resistencia. La EMA de 100 periodos se encuentra en {valor_ema}€, actuando como un nivel de {tipo_ema}."
    
    # Se añade la advertencia al inicio del texto de la observación
    return f'<p style="text-align:left; color:#000; margin: 0 0 5px 0;">{texto_observacion.strip()}{advertencia_texto}{texto.strip()}</p>'


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
        # Se guarda el HTML COMPLETO, incluyendo head y body, para el envío por email
        html_completo = f"""
        <html>
        <head>
            <title>{asunto_email}</title>
            </head>
        <body>
            {texto_generado}
        </body>
        </html>
        """
        with open(ruta_archivo, "w", encoding="utf-8") as f:
            # Ahora guardamos el HTML completo que se generó y se pasó como argumento
            # NOTA: La variable 'texto_generado' en esta función ahora SOLO contiene el cuerpo HTML.
            # Se ha reajustado 'html_completo' arriba para envolverlo.
            f.write(html_completo)
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
            
# --------------------------------------------------------------------------------------
# ------------------ NUEVA FUNCIÓN AÑADIDA PARA LA SEGUNDA TABLA ---------------------
# --------------------------------------------------------------------------------------

def generar_tabla_posiciones_abiertas(datos_completos):
    
    # 1. Filtrar solo las empresas que tienen COMPRADO == "SI"
    posiciones_abiertas = [d for d in datos_completos if d.get('COMPRADO') == "SI"]
    
    if not posiciones_abiertas:
        # **MODIFICACIÓN: Devolver un bloque HTML con un mensaje en lugar de una cadena vacía**
        return """
            <h3 style="text-align: center; color: #1A237E; margin-top: 20px; margin-bottom: 10px; font-size: 1.2em; border-bottom: 1px solid #e9ecef; padding-bottom: 5px;">
                <i class="fas fa-check-circle" style="color:#28a745; margin-right: 5px;"></i>
                Posiciones Abiertas (Cartera IBEXIA)
            </h3>
            <p style="text-align: center; font-size: 1.0em; color: #dc3545; font-weight: bold; margin-bottom: 10px; padding: 5px; border: 1px dashed #dc3545; background-color: #f8d7da;">
                Actualmente no hay posiciones abiertas según el algoritmo.
            </p>
        """
        
    # 2. Ordenar por FECHA_COMPRA (la fecha está en formato DD/MM/YYYY)
    def key_sort_date(item):
        fecha_str = item.get('FECHA_COMPRA')
        try:
            return datetime.strptime(fecha_str, '%d/%m/%Y')
        except ValueError:
            # Poner al final las que no tienen fecha válida o es "N/A"
            return datetime.min 
            
    # Ordenar por fecha de compra, de la más antigua a la más reciente (ascendente)
    posiciones_ordenadas = sorted(posiciones_abiertas, key=key_sort_date)
    
    # 3. Generar el contenido HTML de la tabla
    
    # **MODIFICACIÓN 1: Añadir el texto de advertencia sobre el desplazamiento**
    html_table = """
        <h3 style="text-align: center; color: #1A237E; margin-top: 20px; margin-bottom: 10px; font-size: 1.2em; border-bottom: 1px solid #e9ecef; padding-bottom: 5px;">
            <i class="fas fa-check-circle" style="color:#28a745; margin-right: 5px;"></i>
            Posiciones Abiertas (Cartera IBEXIA)
        </h3>
        <p style="text-align: center; font-size: 0.9em; color: #dc3545; font-weight: bold; margin-bottom: 10px;">
            ⚠️ Desliza hacia abajo dentro de la caja para ver todas las empresas en las que estamos invertidos.
        </p>
        <div class="open-positions-container" style="overflow-x: auto; max-width: 100%; height: 250px; overflow-y: scroll; border: 1px solid #dee2e6;">
            <table style="min-width: 600px; width: 100%; table-layout: auto; border: 0; font-size: 0.95em;">
                <thead>
                    <tr style="background-color: #f0f8ff;">
                        <th style="width: 20%; padding: 5px;">EMPRESA (TICKER)</th>
                        <th style="width: 15%; padding: 5px;">FECHA ENTRADA</th>
                        <th style="width: 15%; padding: 5px;">PRECIO ENTRADA</th>
                        <th style="width: 15%; padding: 5px;">PRECIO ACTUAL</th>
                        <th style="width: 20%; padding: 5px;">BENEFICIO A DÍA DE HOY</th>
                        <th style="width: 15%; padding: 5px;">ESTADO</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for data in posiciones_ordenadas:
        
        # 4. **MODIFICACIÓN 4: Lógica para la columna ESTADO**
        oportunidad = data['OPORTUNIDAD'].lower()
        
        # Criterios de peor caso a mejor caso
        if "venta activada" in oportunidad:
            recomendacion = "VENDEREMOS HOY"
            clase_rec = "venta"
        elif "riesgo de venta" in oportunidad:
            recomendacion = "VALORANDO VENDER AHORA"
            clase_rec = "vigilar" # Amarillo/Naranja
        elif "vigilar" in oportunidad or "intermedio" in oportunidad or "seguirá bajando" in oportunidad:
            recomendacion = "NOS MANTENEMOS CON PRECAUCIÓN"
            clase_rec = "vigilar"
        elif "compra" in oportunidad:
            recomendacion = "NOS MANTENEMOS"
            clase_rec = "compra" # Verde
        else:
            recomendacion = "NOS MANTENEMOS"
            clase_rec = "compra"

        
        # Obtener el nombre de la empresa sin el precio (lo pondremos en el tooltip/enlace)
        nombre_empresa_url = None
        for nombre, ticker_val in tickers.items():
            if ticker_val == data['TICKER']:
                nombre_empresa_url = nombre
                break
        
        if nombre_empresa_url:
            empresa_link = f'https://ibexia.es/category/{nombre_empresa_url.lower()}/'
        else:
            empresa_link = '#'
            
        # Generar la fila de la tabla
        html_table += f"""
                    <tr>
                        <td style="text-align: left; font-weight: bold; padding: 3px 5px;">
                            <a href='{empresa_link}' target='_blank' style='text-decoration:none; color: #1A237E;'>
                                {data['NOMBRE_EMPRESA']} 
                                <span style='color: #6c757d; font-weight: normal; font-size: 0.9em;'>({data['TICKER']})</span>
                            </a>
                        </td>
                        <td style="padding: 3px 5px;">{data['FECHA_COMPRA']}</td>
                        <td style="padding: 3px 5px;">{formatear_numero(data['PRECIO_COMPRA'])}€</td>
                        <td style="padding: 3px 5px;"><span class="compra" style="color: #1A237E;">{formatear_numero(data['PRECIO_ACTUAL'])}€</span></td>
                        <td style="padding: 3px 5px;">{formatear_beneficio(data['BENEFICIO_ACTUAL'])}</td>
                        <td style="padding: 3px 5px;"><span class="{clase_rec}" style="font-weight: bold;">{recomendacion}</span></td>
                    </tr>
        """
        
    html_table += """
                </tbody>
            </table>
        </div>

    """
    
    return html_table

# --------------------------------------------------------------------------------------
# ---------------- FIN DE LA NUEVA FUNCIÓN AÑADIDA PARA LA SEGUNDA TABLA ---------------
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# ---------------------- NUEVA FUNCIÓN DE ANÁLISIS DE TEXTO (MODIFICADA) ---------------
# --------------------------------------------------------------------------------------

def generar_analisis_texto_empresa(data):
    """Genera un bloque de texto HTML detallado para una sola empresa, encapsulando la mayor parte para desplegar con JS."""
    
    # 1. Recuperar datos y formatear
    ticker = data['TICKER']
    nombre_empresa = data['NOMBRE_EMPRESA']
    precio_actual = formatear_numero(data['PRECIO_ACTUAL'])
    tendencia = data['TENDENCIA_ACTUAL']
    oportunidad = data['OPORTUNIDAD']
    compra_si = data['COMPRA_SI']
    vende_si = data['VENDE_SI']
    soporte1 = formatear_numero(data['SOPORTE_1'])
    resistencia1 = formatear_numero(data['RESISTENCIA_1'])
    tipo_ema = data['TIPO_EMA']
    valor_ema = formatear_numero(data['VALOR_EMA'])
    smi_hoy = formatear_numero(data['SMI_HOY'])
    estado_smi = data['ESTADO_SMI']
    
    comprado = data['COMPRADO'] == "SI"
    
    # Datos de Posición/Operativa
    fecha_compra = data['FECHA_COMPRA']
    precio_compra = formatear_numero(data['PRECIO_COMPRA'])
    beneficio_actual_formateado = formatear_beneficio(data['BENEFICIO_ACTUAL'])
    beneficio_ultima_op_formateado = formatear_beneficio(data['BENEFICIO_ULTIMA_OP'])
    
    # URL de la empresa para el H2
    nombre_empresa_url = None
    for nombre, ticker_val in tickers.items():
        if ticker_val == data['TICKER']:
            nombre_empresa_url = nombre
            break
    
    if nombre_empresa_url:
        empresa_link = f'https://ibexia.es/category/{nombre_empresa_url.lower()}/'
    else:
        empresa_link = '#'
        
    # Lógica para el bloque de recomendación principal
    color_text_operativa = "#ffffff" 
    
    if comprado:
        estado_operativa = "POSICIÓN ABIERTA (COMPRADO)"
        color_bg_operativa = "#28a745" # Verde
        recomendacion_principal = "MANTENER"
        if "venta activada" in oportunidad.lower():
            recomendacion_principal = "🚨 VENTA HOY"
            color_bg_operativa = "#dc3545" # Rojo
        elif "riesgo de venta" in oportunidad.lower():
            recomendacion_principal = "⚠️ RIESGO DE VENTA"
            color_bg_operativa = "#ffc107" # Amarillo
            color_text_operativa = "#000000"
    else:
        estado_operativa = "POSICIÓN CERRADA"
        color_bg_operativa = "#dc3545" # Rojo
        recomendacion_principal = oportunidad.upper().replace('POSIBILIDAD DE ', '').replace('ACTIVADA', '').strip()
        if "compra activada" in oportunidad.lower() or "posibilidad de compra" in oportunidad.lower():
            recomendacion_principal = "🚀 COMPRA RECOMENDADA"
            color_bg_operativa = "#28a745" # Verde
        elif "compra riesgo" in oportunidad.lower():
             recomendacion_principal = "⚠️ COMPRA RIESGO"
             color_bg_operativa = "#ffc107"
             color_text_operativa = "#000000"
        elif "vigilar" in oportunidad.lower():
             recomendacion_principal = "VIGILAR"
             color_bg_operativa = "#1A237E" # Azul oscuro para neutral/vigilar
        elif "seguirá bajando" in oportunidad.lower() or "intermedio" in oportunidad.lower():
             recomendacion_principal = "NEUTRAL"
             color_bg_operativa = "#6c757d"


    # 2. Estilos y Contenedor para la MINIFICHA (Parte Visible)
    border_color = color_bg_operativa
    
    html_minificha = f"""
    <div class="empresa-analisis-block" id="block-{ticker}" data-ticker="{ticker}" data-nombre="{nombre_empresa}" data-oportunidad="{oportunidad}" style="border: 1px solid #dee2e6; margin-bottom: 8px; border-radius: 5px; background-color: #f8f9fa; box-shadow: 0 1px 2px rgba(0,0,0,0.03); line-height: 1.2;">
        
        <div class="minificha-header" style="display: flex; align-items: center; justify-content: space-between; padding: 6px 10px; border-bottom: 1px solid #e9ecef;">
            
            <h4 style="margin: 0; font-size: 1.0em; font-weight: bold; color: #1A237E;">
                {nombre_empresa} <span style="font-weight: normal; color: #6c757d; font-size: 0.9em;">({ticker})</span>
            </h4>
            
            <div class="current-price" style="font-size: 1.1em; font-weight: bold; color: #495057;">
                {precio_actual}€
            </div>
            
        </div>
        
        <div class="minificha-body" style="display: flex; align-items: center; justify-content: space-between; padding: 6px 10px;">
            
            <div class="status-indicator" style="background-color: {color_bg_operativa}; color: {color_text_operativa}; padding: 3px 8px; border-radius: 3px; font-weight: bold; font-size: 0.9em; text-align: center; min-width: 150px;">
                {recomendacion_principal}
            </div>
            
            <div class="op-status" style="font-size: 0.9em; text-align: right; min-width: 150px;">
                {f"<strong>Posición:</strong> {beneficio_actual_formateado}" if comprado else f"<strong>Última Op:</strong> {beneficio_ultima_op_formateado}"}
            </div>
            
            <button onclick="toggleDetail('detail-{ticker}', this)" style="background-color: #007bff; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer; font-size: 0.9em; font-weight: bold; margin-left: 10px; min-width: 120px;">
                Ampliar Info <i class="fas fa-chevron-down" style="margin-left: 5px;"></i>
            </button>
            
        </div>
        
        <div id="detail-{ticker}" class="full-detail" style="display: none; border-top: 1px dashed #e9ecef; padding: 10px;">
        
            <p style="margin: 0 0 8px 0; font-size: 0.95em;">
                <strong style="font-size: 1.05em;">ESTADO DE OPERATIVA:</strong> 
                <span style="background-color: {color_bg_operativa}; color: {color_text_operativa}; padding: 2px 5px; border-radius: 3px; font-weight: bold;">
                    {estado_operativa}
                </span>
            </p>

            {generar_observaciones(data)}
            
            <hr style="border: 0; border-top: 1px dashed #ccc; margin: 8px 0;">
            
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between; font-size: 0.95em;">
                
                <div style="flex: 1 1 48%; min-width: 200px; margin-bottom: 5px; padding-right: 5px;">
                    <h5 style="color: #495057; font-size: 1em; border-bottom: 1px solid #ccc; padding-bottom: 3px; margin-top: 0px; margin-bottom: 3px;">
                        <i class="fas fa-chart-line" style="color:#007bff; margin-right: 3px;"></i>
                        DATOS CLAVE (DIARIO)
                    </h5>
                    <p>
                        Precio actual: <strong>{precio_actual}€</strong>. SMI: **{estado_smi}** ({smi_hoy}).<br>
                        Tendencia: **{tendencia}**. Oportunidad: <strong>{oportunidad}</strong>.
                    </p>
                    <p style="margin-top: 5px;">
                        <strong>Compra:</strong> {compra_si}.<br>
                        <strong>Venta:</strong> {vende_si}.
                    </p>
                </div>
                
                <div style="flex: 1 1 48%; min-width: 200px; margin-bottom: 5px;">
                    <h5 style="color: #495057; font-size: 1em; border-bottom: 1px solid #ccc; padding-bottom: 3px; margin-top: 0px; margin-bottom: 3px;">
                        <i class="fas fa-map-marker-alt" style="color:#007bff; margin-right: 3px;"></i>
                        NIVELES
                    </h5>
                    <p>
                        **Soporte**: <strong>{soporte1}€</strong> | **Resistencia**: <strong>{resistencia1}€</strong>.<br>
                        EMA 100 ({valor_ema}€) actúa como **{tipo_ema}**.
                    </p>
                    <p style="margin-top: 5px;">
                        <i class="fas fa-sync" style="color:#6c757d; margin-right: 3px;"></i> 
                        {data['OBSERVACION_SEMANAL'].replace('SMI', 'indicador')}
                    </p>
                </div>
                
                <div style="flex: 1 1 100%; margin-top: 8px;">
                    <h5 style="color: #495057; font-size: 1em; border-bottom: 1px solid #ccc; padding-bottom: 3px; margin-top: 5px; margin-bottom: 3px;">
                        <i class="fas fa-hand-holding-usd" style="color:#007bff; margin-right: 3px;"></i>
                        POSICIÓN ACTUAL
                    </h5>
                    
                    <div style="border: 1px solid #dee2e6; padding: 5px; border-radius: 4px; background-color: #ffffff;">
                        
                        {f"""
                        <p style="margin: 0; padding: 0;">
                            <strong>Entrada:</strong> {precio_compra}€ (Fecha: {fecha_compra})<br>
                            <strong>Beneficio Actual (Simulado):</strong> {beneficio_actual_formateado}
                        </p>
                        """ if comprado else 
                        f"""
                        <p style="margin: 0; padding: 0;">
                            <strong>No hay inversión abierta.</strong> Última operación ({data['FECHA_VENTA_CIERRE']}) resultó en un beneficio de {beneficio_ultima_op_formateado}.
                        </p>
                        """}
                         <p style="margin: 5px 0 0 0; text-align: right;">
                             <a href='{empresa_link}' target='_blank' style='text-decoration:none; color:#dc3545; font-weight: bold; font-size: 0.95em;'>
                                 Ampliar Análisis y Gráfico <i class="fas fa-external-link-alt"></i>
                             </a>
                         </p>
                        
                    </div>
                </div>
            </div>
        </div>
        
    </div>
    """
    
    return html_minificha

# --------------------------------------------------------------------------------------
# -------------------- FIN DE LA NUEVA FUNCIÓN DE ANÁLISIS DE TEXTO --------------------
# --------------------------------------------------------------------------------------


def generar_reporte():
    try:
        # Aquí se mantiene la lectura de la hoja de Google
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

        # --- Lógica de ordenación MODIFICADA (se mantiene) ---
        def obtener_clave_ordenacion(empresa):
            categoria = empresa['OPORTUNIDAD']
            
            prioridad = {
                "Posibilidad de Compra Activada": 1, 
                "Posibilidad de Compra": 2,         
                "Compra RIESGO": 2.5,               
                "VIGILAR": 3,
                "Riesgo de Venta": 4,
                "Riesgo de Venta Activada": 5,
                "Seguirá bajando": 6,
                "Intermedio": 7,
            }

            orden_interna = prioridad.get(categoria, 99) 

            # Ordenar por oportunidad de compra/venta, y dentro de cada grupo, alfabéticamente por nombre
            return (orden_interna, empresa['NOMBRE_EMPRESA']) 

        datos_ordenados = sorted(datos_completos, key=obtener_clave_ordenacion)
        
        # --- Fin de la lógica de ordenación MODIFICADA ---
        now_utc = datetime.utcnow()
        hora_actual = (now_utc + timedelta(hours=2)).strftime('%H:%M')
        fecha_actual_str = datetime.today().strftime('%d de %B %Y').replace(datetime.today().strftime('%B'), ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'][datetime.today().month-1])

        # ******************************************************************************
        # ******************** NUEVA SECCIÓN HTML BASADA EN TEXTO **********************
        # ******************************************************************************
        
        # 1. ESTILOS CSS INLINE (MODIFICADOS PARA MINIMIZAR ALTURA Y TAMAÑO DE FUENTE)
        html_styles = f"""
            <style>
                * {{ box-sizing: border-box !important; }} 
                .main-container {{
                    max-width: 1200px;
                    width: 95%;
                    margin: 10px auto;
                    background-color: #ffffff;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.4;
                    font-size: 0.9em; /* Tamaño de fuente base pequeño */
                }}
                h1 {{
                    text-align: center;
                    font-size: 1.5em; 
                    color: #1A237E;
                    margin-bottom: 5px;
                    margin-top: 5px;
                }}
                h2, h3 {{
                    margin-top: 15px; 
                    margin-bottom: 8px;
                }}
                p {{
                    color: #495057;
                    text-align: left;
                    font-size: 1em; /* Respeta el tamaño base de 0.9em */
                    margin: 0 0 5px 0; 
                }}
                strong {{
                    font-weight: 700;
                    color: #212529;
                }}
                /* Estilos del Buscador - ESTILO PROFESIONAL Y GRANDE */
                #search-input-container {{
                    text-align: center;
                    margin-bottom: 20px; 
                    padding: 10px 0;
                }}
                #company-search {{
                    padding: 12px 20px; 
                    border: 1px solid #ced4da; 
                    border-radius: 6px; 
                    width: 100%;
                    max-width: 700px; /* Más ancho */
                    font-size: 1.1em; /* Fuente grande */
                    outline: none;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); 
                    transition: box-shadow 0.3s ease-in-out;
                }}
                #company-search:focus {{
                    border-color: #1A237E; 
                    box-shadow: 0 0 0 0.2rem rgba(26, 35, 126, .25);
                }}
                /* Estilos de la tabla de posiciones */
                .open-positions-container table td, .open-positions-container table th {{
                    padding: 3px 5px !important;
                    line-height: 1.3;
                    font-size: 0.9em;
                }}
                /* Estilo de los iconos de Chevron para el botón */
                .fa-chevron-up, .fa-chevron-down {{
                    transition: transform 0.3s ease-in-out;
                }}
            </style>
        """
        
        # 2. CUERPO HTML (Contenido insertable en WordPress)
        html_content = f"""
            <div class="main-container">
                <h1>ANÁLISIS DIARIO IBEXIA</h1>
                <p style="text-align: center; font-size: 1.0em; color: #8b0000; font-weight: bold; margin-bottom: 5px;">
                    Fecha de Actualización: {fecha_actual_str} | Hora: {hora_actual} (CET)
                </p>
                <hr style="border: 0; border-top: 1px solid #1A237E; margin: 15px 0;">
                
                <div id="search-input-container">
                    <input type="text" id="company-search" placeholder="Buscar por Nombre de Empresa o Ticker..." onkeyup="filterCompanies()" />
                </div>
                
        """
        
        if not datos_ordenados:
            html_content += """
                <p style="text-align: center; font-size: 1.0em; color: #dc3545; font-weight: bold; padding: 10px; border: 1px solid #dc3545; background-color: #f8d7da;">
                    No se encontraron empresas con datos válidos hoy.
                </p>
            """
        else:
            previous_orden_grupo = None
            
            for i, data in enumerate(datos_ordenados):
                
                current_orden_grupo = obtener_clave_ordenacion(data)[0]
                
                # Lógica para encabezados de grupo (H3)
                es_primera_fila = previous_orden_grupo is None
                es_cambio_grupo = current_orden_grupo != previous_orden_grupo
                
                if es_primera_fila or es_cambio_grupo:
                    
                    if current_orden_grupo in [1, 2, 2.5]:
                        if previous_orden_grupo is None or previous_orden_grupo not in [1, 2, 2.5]:
                            html_content += """
                                <h3 style="color: #28a745; font-size: 1.2em; margin-top: 20px; border-bottom: 2px solid #28a745; padding-bottom: 3px;">
                                    <i class="fas fa-arrow-up" style="margin-right: 5px;"></i>
                                    OPORTUNIDADES DE COMPRA DETECTADAS
                                </h3>
                                <div class="grid-container" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px;">
                            """
                        # Cierre del div si se sale de este grupo
                        if previous_orden_grupo is not None and previous_orden_grupo not in [1, 2, 2.5]:
                            html_content += "</div>"

                    elif current_orden_grupo in [3, 4, 5]:
                        if previous_orden_grupo is None or previous_orden_grupo not in [3, 4, 5]:
                            # Cierre del div si se sale del grupo anterior
                            if previous_orden_grupo in [1, 2, 2.5, 6, 7] or previous_orden_grupo is None and i > 0:
                                html_content += "</div>"
                                
                            html_content += """
                                <h3 style="color: #ffc107; font-size: 1.2em; margin-top: 20px; border-bottom: 2px solid #ffc107; padding-bottom: 3px;">
                                    <i class="fas fa-exclamation-triangle" style="margin-right: 5px;"></i>
                                    ATENTOS A VENDER / VIGILANCIA
                                </h3>
                                <div class="grid-container" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px;">
                            """
                        # Cierre del div si se sale de este grupo
                        if previous_orden_grupo is not None and previous_orden_grupo not in [3, 4, 5]:
                            html_content += "</div>"

                    elif current_orden_grupo in [6, 7]:
                        if previous_orden_grupo is None or previous_orden_grupo not in [6, 7]:
                            # Cierre del div si se sale del grupo anterior
                            if previous_orden_grupo in [1, 2, 2.5, 3, 4, 5] or previous_orden_grupo is None and i > 0:
                                html_content += "</div>"
                                
                            html_content += """
                                <h3 style="color: #6c757d; font-size: 1.2em; margin-top: 20px; border-bottom: 2px solid #6c757d; padding-bottom: 3px;">
                                    <i class="fas fa-minus-circle" style="margin-right: 5px;"></i>
                                    OTRAS EMPRESAS SIN MOVIMIENTOS RELEVANTES
                                </h3>
                                <div class="grid-container" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px;">
                            """

                # Generar la minificha para la empresa
                html_content += generar_analisis_texto_empresa(data)
                
                previous_orden_grupo = current_orden_grupo
                
            # Cierre del último grid-container
            if datos_ordenados:
                html_content += "</div>"
        
        # Agregar la tabla de Posiciones Abiertas al final
        html_content += generar_tabla_posiciones_abiertas(datos_completos)

        html_content += """
                <hr style="border: 0; border-top: 1px dashed #ccc; margin: 15px 0;">
                <p class="disclaimer" style="text-align: center; font-size: 0.8em; color: #6c757d;">
                    <strong>Aviso Legal:</strong> La información contenida en este análisis se proporciona únicamente con fines educativos e informativos y no constituye asesoramiento de inversión, fiscal o legal. Las decisiones de inversión son de exclusiva responsabilidad del usuario.
                </p>
            </div>
        """
        
        # 3. SCRIPT JAVASCRIPT (Para el buscador y el desplegable)
        html_script = """
            <script src="https://kit.fontawesome.com/a076d05399.js" crossorigin="anonymous"></script>
            <script>
                // Función para desplegar/ocultar el detalle de la empresa
                function toggleDetail(detailId, button) {
                    var detail = document.getElementById(detailId);
                    var icon = button.querySelector('i');
                    if (detail.style.display === "none" || detail.style.display === "") {
                        detail.style.display = "block";
                        button.innerHTML = 'Cerrar Info <i class="fas fa-chevron-up" style="margin-left: 5px;"></i>';
                    } else {
                        detail.style.display = "none";
                        button.innerHTML = 'Ampliar Info <i class="fas fa-chevron-down" style="margin-left: 5px;"></i>';
                    }
                }
                
                // Función de filtrado de empresas por nombre o ticker
                function filterCompanies() {
                    var input, filter, blocks;
                    input = document.getElementById('company-search');
                    filter = input.value.toUpperCase();
                    blocks = document.getElementsByClassName('empresa-analisis-block');
                    
                    for (var i = 0; i < blocks.length; i++) {
                        var block = blocks[i];
                        var ticker = block.getAttribute('data-ticker').toUpperCase();
                        var nombre = block.getAttribute('data-nombre').toUpperCase();
                        
                        // Si el filtro coincide con el ticker o el nombre de la empresa
                        if (ticker.indexOf(filter) > -1 || nombre.indexOf(filter) > -1) {
                            block.style.display = "block"; // Mostrar la minificha
                        } else {
                            block.style.display = "none"; // Ocultar la minificha
                        }
                    }
                    
                    // Lógica para ocultar/mostrar los encabezados de grupo (H3)
                    var h3s = document.querySelectorAll('.main-container h3');
                    var gridContainers = document.getElementsByClassName('grid-container');
                    
                    // Recorrer los contenedores de grid
                    for (var j = 0; j < gridContainers.length; j++) {
                        var container = gridContainers[j];
                        var hasVisibleChild = false;
                        
                        // Verificar si algún bloque dentro del grid-container es visible
                        var childBlocks = container.querySelectorAll('.empresa-analisis-block');
                        for (var k = 0; k < childBlocks.length; k++) {
                            if (childBlocks[k].style.display !== "none") {
                                hasVisibleChild = true;
                                break;
                            }
                        }
                        
                        // El encabezado H3 que precede inmediatamente al contenedor
                        var previousSibling = container.previousElementSibling;
                        while (previousSibling && previousSibling.tagName !== 'H3') {
                            previousSibling = previousSibling.previousElementSibling;
                        }
                        
                        if (previousSibling && previousSibling.tagName === 'H3') {
                            if (hasVisibleChild) {
                                previousSibling.style.display = "block";
                                container.style.display = "grid"; // Mostrar el grid
                            } else {
                                previousSibling.style.display = "none";
                                container.style.display = "none"; // Ocultar el grid
                            }
                        }
                    }
                }
            </script>
        """
        
        # ******************************************************************************
        # *************** FIN DE LA NUEVA SECCIÓN HTML BASADA EN TEXTO *****************
        # ******************************************************************************
        
        # El HTML COMPLETO para el correo
        html_para_email_body = f"""
            {html_styles}
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
            {html_content}
            {html_script} 
        """

        
        asunto = f"🔔 Alertas y Oportunidades IBEXIA: {len(datos_ordenados)} análisis detallados hoy {datetime.today().strftime('%d/%m/%Y')}"
        nombre_archivo_base = f"analisis_ibexia_{datetime.today().strftime('%Y%m%d')}"

        # Enviamos el cuerpo HTML (sin tags <html>/<body>, que se añaden en la función de envío)
        enviar_email_con_adjunto(html_para_email_body, asunto, nombre_archivo_base)
        
        # Devolver el bloque único con <style>, <link>, contenido y <script> para la inserción en WordPress.
        return f"""{html_styles}<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">{html_content}{html_script}"""

    except Exception as e:
        print(f"❌ Error al ejecutar el script principal: {e}")
        return None

if __name__ == '__main__':
    generar_reporte()
