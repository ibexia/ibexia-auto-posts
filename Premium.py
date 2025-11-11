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
            data['COMPRA_SI'] = "NO COMPRAR"
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


def generar_html_posiciones(datos_completos, fecha_actual_str):
    posiciones_abiertas = [data for data in datos_completos if data['COMPRADO'] == "SI"]
    
    # --- MODIFICACIÓN 1: Actualizar Título con la fecha y hora de la generación ---
    titulo_personalizado = f"Posiciones Abiertas (Cartera IBEXIA) Actualizado a fecha {fecha_actual_str}."

    if not posiciones_abiertas:
        return f""" <h3 style="text-align: center; color: #1A237E; margin-top: 20px; margin-bottom: 10px; font-size: 1.2em; border-bottom: 1px solid #e9ecef; padding-bottom: 5px;"> <i class="fas fa-check-circle" style="color:#6c757d; margin-right: 5px;"></i> {titulo_personalizado} </h3> <p style="text-align: center; font-size: 0.9em; color: #6c757d; margin-bottom: 10px;"> Actualmente no hay posiciones abiertas según el algoritmo. </p> """
    # -----------------------------------------------------------------------------

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
    # **MODIFICACIÓN 1: Usar el título personalizado**
    html_table = f"""
    <h3 style="text-align: center; color: #1A237E; margin-top: 20px; margin-bottom: 10px; font-size: 1.2em; border-bottom: 1px solid #e9ecef; padding-bottom: 5px;">
        <i class="fas fa-check-circle" style="color:#28a745; margin-right: 5px;"></i> {titulo_personalizado}
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
                    <th style="width: 15%; padding: 5px;">BENEFICIO / PÉRDIDA</th>
                    <th style="width: 20%; padding: 5px;">RECOMENDACIÓN HOY</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for data in posiciones_ordenadas:
        # --- MODIFICACIÓN DE ENLACE INICIA AQUÍ ---
        # Extraer el ticker base (ej. 'PHM' de 'PHM.MC')
        ticker_yf = data['TICKER']
        ticker_base = ticker_yf.split('.')[0]
        # Crear el enlace con el formato requerido: ibexia.es/TICKER_BASE/
        empresa_link = f"https://ibexia.es/{ticker_base.upper()}/" 
        # --- MODIFICACIÓN DE ENLACE TERMINA AQUÍ ---
        
        recomendacion = data['OPORTUNIDAD']
        if 'compra' in recomendacion.lower():
            clase_rec = 'compra-op'
        elif 'venta' in recomendacion.lower():
            clase_rec = 'venta-op'
        elif 'vigilar' in recomendacion.lower():
            clase_rec = 'vigilar-op'
        else:
            clase_rec = 'neutro-op'

        html_table += f"""
        <tr>
            <td style="text-align: left; font-weight: bold; padding: 3px 5px;">
                <a href='{empresa_link}' target='_blank' style='text-decoration:none; color: #1A237E;'>
                    {data['NOMBRE_EMPRESA']} <span style='color: #6c757d; font-weight: normal; font-size: 0.9em;'>({data['TICKER']})</span>
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
# ---------------------- FUNCIÓN DE ANÁLISIS DE TEXTO (MODIFICADA) ---------------------
# --------------------------------------------------------------------------------------
def generar_analisis_texto_empresa(data, is_expanded_default, ficha_color_index, fecha_corta_str):
    """Genera un bloque de texto HTML detallado para una sola empresa. Acepta un nuevo parámetro ficha_color_index para el fondo alterno."""
    
    # 1. Recuperar datos y formatear
    ticker_yf = data['TICKER'] # TICKER_YC is like 'PHM.MC'
    nombre_empresa = data['NOMBRE_EMPRESA']
    precio_actual = formatear_numero(data['PRECIO_ACTUAL'])
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
    fecha_compra = data['FECHA_COMPRA']
    precio_compra = formatear_numero(data['PRECIO_COMPRA'])
    beneficio_actual_formateado = formatear_beneficio(data['BENEFICIO_ACTUAL'])
    
    # --- CAMBIO 2: Enlace a ibexia.es/TICKER_BASE/ ---
    # Extraer el ticker base (ej. 'PHM' de 'PHM.MC')
    ticker_base = ticker_yf.split('.')[0]
    # Crear el enlace con el formato requerido
    empresa_link = f"https://ibexia.es/{ticker_base.upper()}/" 

    # Lógica de colores y textos
    color_text_operativa = "#ffffff"
    if "compra" in oportunidad.lower() or "compra riesgo" in oportunidad.lower():
        recomendacion_principal = "COMPRA"
        color_bg_operativa = "#28a745" # Verde
    elif "venta" in oportunidad.lower() or "vigilar" in oportunidad.lower():
        recomendacion_principal = "VENTA / VIGILANCIA"
        color_bg_operativa = "#dc3545" # Rojo
    elif "seguirá bajando" in oportunidad.lower() or "intermedio" in oportunidad.lower():
        recomendacion_principal = "NEUTRAL"
        color_bg_operativa = "#6c757d" # Gris neutral

    # --- CAMBIO 1 (Parte 1): Determinar el estado inicial del detalle ---
    # La variable is_expanded_default SIEMPRE SERÁ True, forzando la apertura.
    display_detail = "block" if is_expanded_default else "none"
    icon_class = "fa-chevron-up" if is_expanded_default else "fa-chevron-down"
    button_text = "Cerrar Información" if is_expanded_default else "Ampliar Información"
    
    # Determinar la clase de fondo de la ficha
    ficha_class = f"empresa-analisis-block-{ficha_color_index}"
    
    # --- MODIFICACIÓN 2: Texto de actualización en la cabecera de la ficha ---
    html_cabecera_actualizacion = f"""
    <div style="font-size: 0.8em; text-align: right; color: #495057; padding: 5px 10px 0 10px; font-weight: bold;">
        Actualizado a las {fecha_corta_str}
    </div>
    """

    # 2. Estilos y Contenedor para la MINIFICHA (Parte Visible y Cuadrada)
    # El aspecto cuadrado se maneja con la clase CSS .empresa-analisis-block
    html_minificha = f"""
    <div class="{ficha_class}" id="block-{ticker_yf}" data-ticker="{ticker_yf}" data-nombre="{nombre_empresa}" data-oportunidad="{oportunidad}">
        {html_cabecera_actualizacion}
        <div class="minificha-header">
            <h4 style="margin: 0; font-size: 1.0em; font-weight: bold; color: #1A237E;">
                {nombre_empresa} <span style="font-weight: normal; color: #6c757d; font-size: 0.9em;">({ticker_yf})</span>
            </h4>
        </div>
        <div class="minificha-body-resumen">
            <div class="resumen-item current-price">
                <span style="font-size: 1.1em; font-weight: bold; color: #495057;">{precio_actual}€</span>
            </div>
            <div class="resumen-item op-status-resumen" style="background-color: {color_bg_operativa}; color: {color_text_operativa};">
                {recomendacion_principal}
            </div>
            <div class="resumen-item comprado-status" style="font-size: 0.9em; font-weight: bold;">
                {f"En Cartera: <span style='color:#28a745;'>SÍ</span>" if comprado else f"En Cartera: <span style='color:#dc3545;'>NO</span>"}
            </div>
            <a href='{empresa_link}' target='_blank' class="resumen-item chart-link">
                Ver Análisis Detallado y Gráfico <i class="fas fa-external-link-alt" style="margin-left: 5px;"></i>
            </a>
        </div>

        <div id="detail-{ticker_yf}" class="detail-container" style="display: {display_detail};">
            <div class="detail-content">
                
                {generar_observaciones(data)}

                <h5 style="color: #495057; font-size: 1em; border-bottom: 1px solid #ccc; padding-bottom: 3px; margin-top: 5px; margin-bottom: 3px;">
                    <i class="fas fa-chart-line" style="color:#007bff; margin-right: 3px;"></i> NIVELES CLAVE
                </h5>
                <div style="border: 1px solid #dee2e6; padding: 5px; border-radius: 4px; background-color: #ffffff;">
                    <p style="margin: 0; padding: 0;"><strong>SMI/Estado:</strong> {smi_hoy} ({estado_smi})<br>
                    <strong>Soporte/Resistencia 1:</strong> {soporte1}€ / {resistencia1}€<br>
                    <strong>EMA 100 ({tipo_ema}):</strong> {valor_ema}€</p>
                </div>
                
                <h5 style="color: #495057; font-size: 1em; border-bottom: 1px solid #ccc; padding-bottom: 3px; margin-top: 5px; margin-bottom: 3px;">
                    <i class="fas fa-hand-holding-usd" style="color:#007bff; margin-right: 3px;"></i> POSICIÓN ACTUAL
                </h5>
                <div style="border: 1px solid #dee2e6; padding: 5px; border-radius: 4px; background-color: #ffffff;">
                    {f"""
                    <p style="margin: 0; padding: 0;">
                        <strong>Entrada:</strong> {precio_compra}€ (Fecha: {fecha_compra})<br>
                        <strong>Beneficio Actual (Simulado):</strong> {beneficio_actual_formateado}
                    </p>
                    """ if comprado else f"""
                    <p style="margin: 0; padding: 0;">
                        <strong>No hay inversión abierta.</strong> Última operación ({data['FECHA_VENTA_CIERRE']}) resultó en un beneficio de {formatear_beneficio(data['BENEFICIO_ULTIMA_OP'])}.
                    </p>
                    """}
                </div>
            </div>
        </div>
        <div class="minificha-footer">
            <button class="expand-button" onclick="toggleDetail('detail-{ticker_yf}', this)" data-ticker="{ticker_yf}">
                {button_text} <i class="fas {icon_class}" style="margin-left: 5px;"></i>
            </button>
        </div>
    </div>
    """
    return html_minificha

# --------------------------------------------------------------------------------------
# -------------------- FIN DE LA FUNCIÓN DE ANÁLISIS DE TEXTO --------------------------
# --------------------------------------------------------------------------------------


def enviar_email_con_adjunto(asunto_email):
    """
    Envía un correo electrónico a través de Brevo (Sendinblue) con el archivo PHP adjunto,
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
    
    # --- LA RUTA DEL ARCHIVO A ADJUNTAR ES AHORA widget-data.php ---
    ruta_archivo_adjunto = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'widget-data.php')
    nombre_adjunto = 'widget-data.php'
    
    # 2. Comprobación y Construcción del mensaje MIME
    if not os.path.exists(ruta_archivo_adjunto):
        print(f"❌ ERROR: El archivo {nombre_adjunto} no existe. No se puede adjuntar.")
        return

    msg = MIMEMultipart()
    msg['From'] = remitente_header 
    msg['To'] = destinatario
    msg['Subject'] = asunto_email

    # Cuerpo del email (Solo texto, ya que el adjunto es el importante)
    msg.attach(MIMEText(f"Adjunto el archivo de datos {nombre_adjunto} con el análisis diario. Sustituye este archivo en tu plugin de WordPress.", 'plain'))

    # Adjuntar el archivo PHP
    try:
        with open(ruta_archivo_adjunto, "rb") as attachment:
            part = MIMEBase("application", "x-php") # Tipo MIME correcto para PHP
            part.set_payload(attachment.read())

        # Codificación y cabeceras para el adjunto
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {nombre_adjunto}",
        )
        msg.attach(part)
    except Exception as e:
        print(f"❌ Error al adjuntar el archivo {nombre_adjunto}: {e}")
        return

    # 3. Conexión al servidor Brevo SMTP
    try:
        print(f"🌐 Intentando conectar a Brevo SMTP: {servidor_smtp}:{puerto_smtp}")
        servidor = smtplib.SMTP(servidor_smtp, puerto_smtp)
        servidor.starttls()

        print(f"🔑 Intentando iniciar sesión con el usuario: {remitente_login}")
        servidor.login(remitente_login, password)

        print(f"✉️ Enviando correo a: {destinatario} desde: {remitente_visible_email}")
        servidor.sendmail(remitente_visible_email, destinatario, msg.as_string())
        servidor.quit()
        print("✅ Correo enviado exitosamente a Brevo con el archivo PHP adjunto.")

    except smtplib.SMTPAuthenticationError:
        print(f"❌ ERROR de Autenticación SMTP. Verifica las credenciales de Brevo.")
    except Exception as e:
        print(f"❌ ERROR al conectar o enviar por SMTP: {e}")


def generar_reporte():
    try:
        # --- MODIFICACIÓN: Obtener la fecha y hora de inicio del reporte ---
        now = datetime.now()
        # Formato para el título de la tabla: "11 de noviembre a las 14:03 horas"
        fecha_actual_str = now.strftime('%d de ') + now.strftime('%B').lower() + now.strftime(' a las %H:%M horas')
        # Formato para la cabecera de la ficha: "14:03 del dia 11/11/25"
        fecha_corta_str = now.strftime('%H:%M del día %d/%m/%y')
        # -------------------------------------------------------------------
        
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
                print(f"Error procesando {ticker}: {e}")
                time.sleep(1) # Espera un poco antes de seguir

        if not datos_completos:
            print("No se generó análisis para ninguna empresa.")
            return

        # 1. Ordenar datos (Primero Compra, luego Vigilar/Venta, luego Neutral)
        datos_ordenados = sorted(datos_completos, key=lambda x: x['ORDEN_PRIORIDAD'])
        
        # 2. Generar el HTML de las posiciones abiertas (Tabla de Inversión)
        # --- MODIFICACIÓN: Pasar fecha_actual_str a la función ---
        html_posiciones = generar_html_posiciones(datos_ordenados, fecha_actual_str)

        # 3. Generar el HTML de las fichas de análisis (Acordeón)
        html_content = f"""<div class="main-widget-container"><div id="analisis-diario-widget">"""
        
        # Sección de la tabla de posiciones (si aplica)
        html_content += html_posiciones
        
        # Bloque de Título
        html_content += f"""<h1 style="text-align: center; font-size: 1.5em; color: #1A237E; margin-bottom: 5px; margin-top: 5px;">RESUMEN DE OPORTUNIDADES HOY {datetime.today().strftime('%d/%m/%Y')}</h1>"""
        
        # Buscador
        html_content += """
        <div id="search-input-container">
            <input type="text" id="company-search" placeholder="Buscar por Ticker o Nombre..." onkeyup="filterCompanies()">
        </div>
        """

        # Generar las fichas agrupadas
        previous_orden_grupo = None
        group_counters = {}
        
        for i, data in enumerate(datos_ordenados):
            current_orden_grupo = data['ORDEN_PRIORIDAD']

            if current_orden_grupo not in group_counters:
                group_counters[current_orden_grupo] = 0

            if previous_orden_grupo is None or current_orden_grupo != previous_orden_grupo:
                
                # Cierra el contenedor del grupo anterior (grid-container y collapsible-content)
                if previous_orden_grupo is not None:
                    html_content += "</div>" # Cierra el grid-container
                    html_content += "</div>" # Cierra el .collapsible-content wrapper

                # Definir título y estilos del nuevo grupo
                if current_orden_grupo in [1, 2, 8]: # Compra
                    titulo = "OPORTUNIDADES DE COMPRA DETECTADAS"
                    clase = "h3-compra"
                    icono = "fas fa-leaf"
                elif current_orden_grupo in [3, 4, 5]: # Vigilar/Venta
                    titulo = "ATENTOS A VENDER / VIGILANCIA DE RIESGO"
                    clase = "h3-vigilar"
                    icono = "fas fa-eye"
                elif current_orden_grupo in [6, 7]: # Neutral/Otros
                    titulo = "OTRAS EMPRESAS SIN MOVIMIENTOS RELEVANTES"
                    clase = "h3-neutral"
                    icono = "fas fa-ellipsis-h"
                else:
                    titulo = "OTROS ANÁLISIS"
                    clase = "h3-neutral"
                    icono = "fas fa-info-circle"
                    
                # Abrir el nuevo encabezado (H3)
                html_content += f"""
                <h3 class="{clase}" data-group="{current_orden_grupo}" style="cursor: default;">
                    <i class="{icono}" style="margin-right: 10px;"></i> {titulo}
                </h3>
                """
                
                # Collapsible wrapper for the content (FICHES)
                # Se fuerza a display: grid (abierto) para todos los grupos.
                display_style = "display: grid;"
                html_content += f"""
                <div id="collapsible-content-{current_orden_grupo}" class="collapsible-content" style="{display_style}">
                    <div class="grid-container" data-group="{current_orden_grupo}">
                """
                previous_orden_grupo = current_orden_grupo

            # *** CAMBIO 1 (Parte 2): Forzar que todas las fichas salgan abiertas por defecto ***
            is_expanded_default = True
            
            # *** MODIFICACIÓN: Alternancia de fondo entre 4 colores ***
            ficha_color_index = (i % 4) + 1 # Cicla entre 1, 2, 3, 4

            # Generar la minificha para la empresa
            # --- MODIFICACIÓN: Pasar fecha_corta_str a la función ---
            html_content += generar_analisis_texto_empresa(data, is_expanded_default, ficha_color_index, fecha_corta_str)
            
            group_counters[current_orden_grupo] += 1
            
        # Cierre final de los últimos contenedores
        if previous_orden_grupo is not None:
            html_content += "</div>" # Cierra el grid-container
            html_content += "</div>" # Cierra el .collapsible-content wrapper
            
        html_content += "</div>" # Cierra el div#analisis-diario-widget

        # 4. Estilos CSS (Completo)
        html_styles = """
            <style>
                .main-widget-container { font-family: 'Arial', sans-serif; max-width: 100%; margin: auto; padding: 10px; background-color: #f7f7f7; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                h1 { text-align: center; font-size: 1.5em; color: #1A237E; margin-bottom: 5px; margin-top: 5px; }
                h3 { margin-top: 20px; margin-bottom: 8px; padding: 5px 10px; border-radius: 4px; font-size: 1.2em; font-weight: bold; display: flex; align-items: center; cursor: default; /* Eliminamos el cursor pointer en todos */ }
                .h3-compra { background-color: #e6f7ee; /* Verde suave */ color: #1a7e4b; /* Verde oscuro */ border-left: 5px solid #28a745; }
                .h3-vigilar { background-color: #fff9e6; /* Amarillo suave */ color: #997b00; /* Naranja oscuro */ border-left: 5px solid #ffc107; }
                .h3-neutral { background-color: #f2f2f2; /* Gris suave */ color: #495057; /* Gris oscuro */ border-left: 5px solid #6c757d; }
                p { color: #495057; text-align: left; font-size: 1em; margin: 0 0 5px 0; }
                strong { font-weight: 700; color: #212529; }
                
                /* Estilos del Buscador - GRANDE Y LLAMATIVO */
                #search-input-container { text-align: center; margin-bottom: 20px; padding: 10px 0; background-color: #f0f8ff; /* Fondo muy suave para destacar el buscador */ border-radius: 8px; border: 1px solid #1A237E; }
                #company-search { padding: 15px 25px; /* Más grande */ width: 90%; max-width: 500px; border-radius: 30px; border: 2px solid #1A237E; font-size: 1.1em; outline: none; transition: border-color 0.3s, box-shadow 0.3s; }
                #company-search:focus { border-color: #007bff; box-shadow: 0 0 8px rgba(0, 123, 255, 0.2); }

                /* Estilos del Grid de Fichas */
                .grid-container { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 15px; margin-top: 10px; }
                
                /* Estilos de Fichas Individuales */
                .empresa-analisis-block-1, .empresa-analisis-block-2, .empresa-analisis-block-3, .empresa-analisis-block-4 { border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); overflow: hidden; display: flex; flex-direction: column; transition: transform 0.2s; border: 1px solid #ddd; }
                .empresa-analisis-block-1 { background-color: #e6f7ee; } /* Verde claro */
                .empresa-analisis-block-2 { background-color: #fff9e6; } /* Amarillo claro */
                .empresa-analisis-block-3 { background-color: #f0f8ff; } /* Azul claro */
                .empresa-analisis-block-4 { background-color: #f2f2f2; } /* Gris claro */
                
                .empresa-analisis-block-1:hover, .empresa-analisis-block-2:hover, .empresa-analisis-block-3:hover, .empresa-analisis-block-4:hover { transform: translateY(-3px); box-shadow: 0 6px 12px rgba(0,0,0,0.15); }

                /* Minificha (Parte Visible) */
                .minificha-header { padding: 8px 10px; border-bottom: 1px solid #e9ecef; background-color: inherit; /* Hereda del padre para el fondo alterno */ }
                .minificha-body-resumen { flex-grow: 1; display: grid; grid-template-columns: 1fr 1fr; gap: 5px; padding: 10px; text-align: center; align-items: center; align-content: start; }
                .resumen-item { padding: 5px; border-radius: 3px; }
                .current-price { grid-column: 1 / 3; font-size: 1.1em; }
                .op-status-resumen { font-weight: bold; font-size: 0.9em; grid-column: 1 / 3; /* Ocupa todo el ancho */ }
                .chart-link { text-decoration: none; color: #007bff; font-weight: bold; font-size: 0.85em; grid-column: 1 / 3; /* Ocupa todo el ancho */ }
                .chart-link:hover { text-decoration: underline; }
                
                /* Detalle (Parte Expandible) */
                .detail-container { padding: 10px; border-top: 1px solid #e9ecef; background-color: white; }
                .detail-container p { font-size: 0.9em; line-height: 1.4; }
                
                .minificha-footer { padding: 8px 10px; border-top: 1px solid #e9ecef; text-align: center; background-color: #f8f9fa; /* Gris claro para el footer */ }
                .expand-button { width: 100%; background-color: #1A237E; color: white; border: none; padding: 8px 10px; border-radius: 4px; cursor: pointer; font-size: 0.95em; font-weight: bold; transition: background-color 0.3s; }
                .expand-button:hover { background-color: #0056b3; }
                
                /* Estilos de la tabla de posiciones (Mantenidos) */
                .open-positions-container table td, .open-positions-container table th { padding: 3px 5px; border-bottom: 1px solid #dee2e6; text-align: center; vertical-align: middle; }
                .open-positions-container table { border-collapse: collapse; }
                .open-positions-container thead th { position: sticky; top: 0; background-color: #f0f8ff; z-index: 10; font-weight: bold; font-size: 0.9em; }

            </style>
        """

        # 5. Scripts (Completo)
        html_script = """
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    console.log("Widget Análisis Diario: Fichas cargadas y listas.");
                    // Forzar el filtro al inicio para mantener todas las fichas abiertas
                    filterCompanies(); 
                });

                function toggleDetail(detailId, button) {
                    var detail = document.getElementById(detailId);
                    if (detail.style.display === "none" || detail.style.display === "") {
                        detail.style.display = "block";
                        button.innerHTML = 'Cerrar Información <i class="fas fa-chevron-up" style="margin-left: 5px;"></i>';
                    } else {
                        detail.style.display = "none";
                        button.innerHTML = 'Ampliar Información <i class="fas fa-chevron-down" style="margin-left: 5px;"></i>';
                    }
                }

                // Función de filtrado de empresas (modificada para secciones y fiches)
                function filterCompanies() {
                    var input, filter, blocks;
                    input = document.getElementById('company-search');
                    filter = input.value.toUpperCase();
                    
                    blocks = document.querySelectorAll('.empresa-analisis-block-1, .empresa-analisis-block-2, .empresa-analisis-block-3, .empresa-analisis-block-4');
                    var groupsVisibility = {}; // Para rastrear si un grupo tiene elementos visibles

                    // 1. Ocultar/Mostrar bloques individuales
                    for (var i = 0; i < blocks.length; i++) {
                        var block = blocks[i];
                        var ticker = block.getAttribute('data-ticker').toUpperCase();
                        var nombre = block.getAttribute('data-nombre').toUpperCase();
                        var contentDiv = block.closest('.collapsible-content');
                        if (!contentDiv) continue;
                        
                        var groupId = contentDiv.id.replace('collapsible-content-', '');
                        
                        if (ticker.indexOf(filter) > -1 || nombre.indexOf(filter) > -1) {
                            block.style.display = "flex"; // Mostrar
                            groupsVisibility[groupId] = true;
                            
                            // *** Cambio para mantener ABIERTAS por defecto, solo se contraen si se filtra ***
                            var detailId = 'detail-' + block.getAttribute('data-ticker');
                            var detail = document.getElementById(detailId);
                            var button = block.querySelector('.expand-button');
                            
                            if (filter === "") {
                                // Mantiene Abierto por defecto
                                if (detail.style.display === "none" || detail.style.display === "") {
                                    toggleDetail(detailId, button);
                                }
                            } else {
                                // Cierra si se filtra y no hay necesidad de abrir
                                if (detail.style.display === "block") {
                                    toggleDetail(detailId, button);
                                }
                            }
                            // **************************************************************************

                        } else {
                            block.style.display = "none"; // Ocultar
                        }
                    }

                    // 2. Ocultar/Mostrar secciones (H3 y .collapsible-content) basadas en el resultado del filtro
                    var h3s = document.querySelectorAll('.h3-compra, .h3-vigilar, .h3-neutral');
                    for (var j = 0; j < h3s.length; j++) {
                        var h3 = h3s[j];
                        var groupId = h3.getAttribute('data-group');
                        var content = document.getElementById('collapsible-content-' + groupId);
                        
                        if (groupsVisibility[groupId] || filter === "") {
                            // Mostrar si hay elementos visibles o si no hay filtro (caso inicial)
                            h3.style.display = "flex";
                            content.style.display = "grid"; // Forzamos grid
                        } else {
                            // Ocultar si el filtro está activo y no hay coincidencias en el grupo
                            h3.style.display = "none";
                            content.style.display = "none";
                        }
                    }

                    // Asegurar que las secciones (H3 y content) estén visibles (porque no hay filtro)
                    for (var k = 0; k < h3s.length; k++) {
                        var h3_k = h3s[k];
                        var content_k = document.getElementById('collapsible-content-' + h3_k.getAttribute('data-group'));
                        if (filter === "") {
                            h3_k.style.display = "flex";
                            content_k.style.display = "grid";
                        }
                    }
                }
            </script>
        """

        # ******************************************************************************
        # *************** ZONA CRUCIAL: GENERACIÓN DEL ARCHIVO PHP *********************
        # ******************************************************************************
        
        # 1. CONSTRUIR EL HTML COMPLETO PARA EL WIDGET
        # Esta es la cadena final que contiene estilos, link, contenido y scripts.
        html_para_widget = f"""{html_styles}<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">{html_content}{html_script}"""
        
        # 2. FORMATEAR EL CONTENIDO PARA EL ARCHIVO PHP (Aseguramos el RETURN)
        # Escapamos las comillas simples ('') dentro del HTML para evitar romper la sintaxis PHP.
        html_escaped = html_para_widget.replace("'", "\\'") 
        
        # ESTO GENERA EL CONTENIDO FINAL CON LA SENTENCIA PHP 'return'
        php_content_to_write = f"""<?php 
// ESTE ARCHIVO FUE GENERADO AUTOMÁTICAMENTE POR PREMIUM.py EL {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}

return '
{html_escaped}
';
"""
        
        # 3. ESCRIBIR EL ARCHIVO widget-data.php
        # Esto calcula la ruta relativa al script Premium.py y lo guarda en la misma carpeta.
        widget_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'widget-data.php')
        
        try:
            with open(widget_data_path, 'w', encoding='utf-8') as f:
                f.write(php_content_to_write)
            print(f"✅ Archivo widget-data.php generado y actualizado exitosamente en: {widget_data_path}")
        except Exception as e:
            print(f"❌ ERROR al escribir widget-data.php en {widget_data_path}: {e}")
            
        # ******************************************************************************
        # ************************* FIN DE LA ZONA CRUCIAL *****************************
        # ******************************************************************************

        # El asunto del correo
        asunto = f"🔔 Alertas y Oportunidades IBEXIA: {len(datos_ordenados)} análisis detallados hoy {datetime.today().strftime('%d/%m/%Y')}"

        # Enviamos el archivo PHP que acabamos de generar
        enviar_email_con_adjunto(asunto)
        
        # Devolver el bloque único con <style>, <link>, contenido y <script> para la inserción en WordPress.
        return html_para_widget

    except Exception as e:
        print(f"❌ Error al ejecutar el reporte: {e}")
        return f"Error en la generación del reporte: {e}"

if __name__ == '__main__':
    print("--- INICIANDO PROCESO DE ANÁLISIS ---")
    resultado = generar_reporte()
    print("--- PROCESO COMPLETADO ---")
