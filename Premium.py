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
    
    # Nuevo campo para el filtrado en el JS
    data['FILTRO_GRUPO'] = 'INTERMEDIO'

    if estado_smi == "Sobreventa":
        if tendencia == "Subiendo":
            # --- Lógica de Filtro Semanal ---
            if estado_smi_weekly == "Sobrecompra":
                data['OPORTUNIDAD'] = "Compra RIESGO"
                data['COMPRA_SI'] = "NO RECOMENDAMOS" # La subida puede ser corta
                data['VENDE_SI'] = "NO VENDER"
                data['ORDEN_PRIORIDAD'] = prioridad["Compra RIESGO"]
                data['ADVERTENCIA_SEMANAL'] = "SI"
                data['FILTRO_GRUPO'] = 'COMPRA'
            else:
                data['OPORTUNIDAD'] = "Posibilidad de Compra Activada"
                data['COMPRA_SI'] = "COMPRA YA"
                data['VENDE_SI'] = "NO VENDER"
                data['ORDEN_PRIORIDAD'] = prioridad["Posibilidad de Compra Activada"]
                data['FILTRO_GRUPO'] = 'COMPRA'
            # -----------------------------------
        elif tendencia == "Bajando":
            # --- Lógica de Filtro Semanal ---
            if estado_smi_weekly == "Sobrecompra":
                data['OPORTUNIDAD'] = "Compra RIESGO"
                data['COMPRA_SI'] = "NO RECOMENDAMOS"
                data['VENDE_SI'] = "NO VENDER"
                data['ORDEN_PRIORIDAD'] = prioridad["Compra RIESGO"]
                data['ADVERTENCIA_SEMANAL'] = "SI"
                data['FILTRO_GRUPO'] = 'COMPRA'
            else:
                data['OPORTUNIDAD'] = "Posibilidad de Compra"
                if current_price > close_yesterday:
                    data['COMPRA_SI'] = "COMPRA YA"
                else:
                    data['COMPRA_SI'] = f"COMPRAR SI SUPERA {formatear_numero(close_yesterday)}€"
                data['VENDE_SI'] = "NO VENDER"
                data['ORDEN_PRIORIDAD'] = prioridad["Posibilidad de Compra"]
                data['FILTRO_GRUPO'] = 'COMPRA'
            # -----------------------------------
        else:
            data['OPORTUNIDAD'] = "Intermedio"
            data['COMPRA_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['VENDE_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['ORDEN_PRIORIDAD'] = prioridad["Intermedio"]
            data['FILTRO_GRUPO'] = 'VIGILAR' # Se considera vigilancia si está plano en sobreventa/sobrecompra
    
    elif estado_smi == "Intermedio":
        if tendencia == "Bajando":
            data['OPORTUNIDAD'] = "Seguirá bajando"
            data['COMPRA_SI'] = "NO COMPRAR"
            data['VENDE_SI'] = "YA ES TARDE PARA VENDER"
            data['ORDEN_PRIORIDAD'] = prioridad["Seguirá bajando"]
            data['FILTRO_GRUPO'] = 'VIGILAR'
        elif tendencia == "Subiendo":
            data['OPORTUNIDAD'] = "VIGILAR"
            data['COMPRA_SI'] = "NO COMPRAR"
            
            trigger_price = close_yesterday * 0.99
            
            if current_price < trigger_price:
                 data['VENDE_SI'] = "VENDE YA"
            else:
                 data['VENDE_SI'] = f"VENDER SI PIERDE {formatear_numero(trigger_price)}€"
            data['ORDEN_PRIORIDAD'] = prioridad["VIGILAR"]
            data['FILTRO_GRUPO'] = 'VIGILAR'
        else:
            data['OPORTUNIDAD'] = "Intermedio"
            data['COMPRA_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['VENDE_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['ORDEN_PRIORIDAD'] = prioridad["Intermedio"]
            data['FILTRO_GRUPO'] = 'VIGILAR'
            
    elif estado_smi == "Sobrecompra":
        if tendencia == "Subiendo":
            data['OPORTUNIDAD'] = "Riesgo de Venta"
            data['COMPRA_SI'] = "NO COMPRAR"
            data['VENDE_SI'] = f"ZONA DE VENTA<br><span class='small-text'>PRECIO IDEAL VENTA HOY: {high_today:,.2f}€</span>"
            data['ORDEN_PRIORIDAD'] = prioridad["Riesgo de Venta"]
            data['FILTRO_GRUPO'] = 'VENTA'
        elif tendencia == "Bajando":
            data['OPORTUNIDAD'] = "Riesgo de Venta Activada"
            data['COMPRA_SI'] = "NO COMPRAR"
            data['VENDE_SI'] = "VENDE AHORA"
            data['ORDEN_PRIORIDAD'] = prioridad["Riesgo de Venta Activada"]
            data['FILTRO_GRUPO'] = 'VENTA'
        else:
            data['OPORTUNIDAD'] = "Intermedio"
            data['COMPRA_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['VENDE_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['ORDEN_PRIORIDAD'] = prioridad["Intermedio"]
            data['FILTRO_GRUPO'] = 'VIGILAR'
    
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
        advertencia_texto = "<strong style='color:#dc3545;'>ADVERTENCIA SEMANAL: El SMI semanal está en zona de sobrecompra. No se recomienda comprar ya que la subida podría ser muy corta y con alto riesgo.</strong><br>"


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
        # DEVOLVEMOS UN MENSAJE EN LUGAR DE UNA CADENA VACÍA PARA MOSTRAR EN LA PESTAÑA
        return """
            <p style="text-align: center; font-size: 1.2em; color: #dc3545; font-weight: bold; margin-top: 30px;">
                ❌ No hay posiciones abiertas actualmente.
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
        <p style="text-align: center; font-size: 1.0em; color: #6c757d; font-weight: normal; margin-bottom: 20px;">
            ⚠️ Desliza horizontalmente y verticalmente dentro de la tabla para ver todos los datos.
        </p>
        <div class="open-positions-container" style="overflow-x: auto; max-width: 100%; height: auto; max-height: 400px; overflow-y: scroll; border: 1px solid #dee2e6; border-radius: 8px;">
            <table style="min-width: 700px; width: 100%; table-layout: auto; border: 0;">
                <thead>
                    <tr style="background-color: #1A237E; color: white;">
                        <th style="width: 20%; position: sticky; top: 0; z-index: 1;">EMPRESA (TICKER)</th>
                        <th style="width: 15%; position: sticky; top: 0; z-index: 1;">FECHA ENTRADA</th>
                        <th style="width: 15%; position: sticky; top: 0; z-index: 1;">PRECIO ENTRADA</th>
                        <th style="width: 15%; position: sticky; top: 0; z-index: 1;">PRECIO ACTUAL</th>
                        <th style="width: 20%; position: sticky; top: 0; z-index: 1;">BENEFICIO A DÍA DE HOY</th>
                        <th style="width: 15%; position: sticky; top: 0; z-index: 1;">ESTADO RECOMENDADO</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for data in posiciones_ordenadas:
        
        # 4. Lógica para la columna ESTADO
        oportunidad = data['OPORTUNIDAD'].lower()
        
        # Criterios de peor caso a mejor caso
        if "venta activada" in oportunidad:
            recomendacion = "VENDEREMOS HOY"
            clase_rec = "venta-op"
        elif "riesgo de venta" in oportunidad:
            recomendacion = "VALORANDO VENDER"
            clase_rec = "vigilar-op" # Amarillo/Naranja
        elif "vigilar" in oportunidad or "intermedio" in oportunidad or "seguirá bajando" in oportunidad:
            recomendacion = "NOS MANTENEMOS"
            clase_rec = "vigilar-op"
        elif "compra" in oportunidad:
            recomendacion = "NOS MANTENEMOS FUERTE"
            clase_rec = "compra-op" # Verde
        else:
            recomendacion = "NOS MANTENEMOS"
            clase_rec = "compra-op"

        
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
                    <tr style="background-color: #ffffff;">
                        <td style="text-align: left; font-weight: bold;">
                            <a href='{empresa_link}' target='_blank' style='text-decoration:none; color: #1A237E;'>
                                {data['NOMBRE_EMPRESA']} 
                                <span style='color: #6c757d; font-weight: normal; font-size: 0.9em;'>({data['TICKER']})</span>
                            </a>
                        </td>
                        <td>{data['FECHA_COMPRA']}</td>
                        <td>{formatear_numero(data['PRECIO_COMPRA'])}€</td>
                        <td><span class="compra-op-price" style="color: #1A237E;">{formatear_numero(data['PRECIO_ACTUAL'])}€</span></td>
                        <td>{formatear_beneficio(data['BENEFICIO_ACTUAL'])}</td>
                        <td><span class="{clase_rec}" style="font-weight: bold; padding: 4px 8px; border-radius: 5px; color: white;">{recomendacion}</span></td>
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

def generar_tabla_analisis(datos_ordenados):
    """Genera la tabla principal de análisis (visible en 'Ver Todos' y los filtros)"""
    
    # 1. ESTILOS CSS INLINE para las tablas (se mantienen aquí para la generación modular)
    html_table_styles = """
        <div class="table-container" id="tableContainer" style="display:none;">
            <table id="myTable">
                <thead>
                    <tr>
                        <th style="width: 25%;">Empresa (Precio)</th>
                        <th style="width: 10%;">Tendencia</th>
                        <th style="width: 15%;">Oportunidad</th>
                        <th style="width: 25%;">Compra si...</th>
                        <th style="width: 25%;">Vende si...</th>
                        <th style="width: 5%;">Detalle</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    if not datos_ordenados:
        html_table_styles += """
                        <tr><td colspan="6">No se encontraron empresas con datos válidos o no hay resultados para el filtro.</td></tr>
        """
    else:
        previous_orden_grupo = None
        for i, data in enumerate(datos_ordenados):
            
            # La clave de ordenación también se usará para el filtro. 
            # Los números 1, 2 y 2.5 son de COMPRA. 3, 6, 7 son VIGILAR. 4 y 5 son VENTA.
            orden_tuple = obtener_clave_ordenacion(data)
            current_orden_grupo = orden_tuple[0]
            
            # Asignar una ID de filtrado basada en la lógica de clasificación
            filtro_id = data.get('FILTRO_GRUPO', 'INTERMEDIO')
            
            # Lógica para determinar el encabezado de categoría (solo si el modo es 'Ver Todos' o 'Buscar')
            es_primera_fila = previous_orden_grupo is None
            es_cambio_grupo = current_orden_grupo != previous_orden_grupo
            
            # El sistema de categorías solo se activa en el modo "Ver Todos" (que es el modo por defecto de esta función)
            if es_primera_fila or es_cambio_grupo:
                
                if current_orden_grupo in [1, 2, 2.5]: 
                    if previous_orden_grupo is None or previous_orden_grupo not in [1, 2, 2.5]:
                        html_table_styles += f"""
                            <tr class="category-header category-header-compra" data-filtro="COMPRA"><td colspan="6">OPORTUNIDADES DE COMPRA</td></tr>
                        """
                elif current_orden_grupo in [4, 5]:
                    if previous_orden_grupo is None or previous_orden_grupo not in [4, 5]:
                        html_table_styles += f"""
                            <tr class="category-header category-header-venta" data-filtro="VENTA"><td colspan="6">RIESGOS DE VENTA</td></tr>
                        """
                elif current_orden_grupo in [3, 6, 7, 99]: # 99 es para empresas sin clasificación o con datos incompletos
                    if previous_orden_grupo is None or previous_orden_grupo not in [3, 6, 7, 99]:
                        html_table_styles += f"""
                            <tr class="category-header category-header-vigilar" data-filtro="VIGILAR"><td colspan="6">VALORES A VIGILAR/INTERMEDIOS</td></tr>
                        """
                        
                # Poner un separador si no es la primera fila y hay cambio de grupo
                if not es_primera_fila and es_cambio_grupo:
                    html_table_styles += """
                        <tr class="separator-row"><td colspan="6"></td></tr>
                    """

            # Lógica para el enlace
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

            # Ajuste de clases para el estado
            if "compra" in data['OPORTUNIDAD'].lower() and "riesgo" not in data['OPORTUNIDAD'].lower():
                clase_oportunidad = "compra"
                celda_empresa_class = "green-cell"
            elif "venta" in data['OPORTUNIDAD'].lower():
                clase_oportunidad = "venta"
                celda_empresa_class = "red-cell"
            elif "vigilar" in data['OPORTUNIDAD'].lower():
                clase_oportunidad = "vigilar"
                celda_empresa_class = "yellow-cell-light" # Cambio a color claro para no confundir con riesgo
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
                    <div class='position-box-op' style='background-color: #e9ecef;'>
                        <b>ÚLTIMA OPERACIÓN CERRADA:</b>
                        <ul class="op-list">
                            <li><strong>Compra:</strong> {formatear_numero(data['PRECIO_COMPRA'])}€ ({data['FECHA_COMPRA']})</li>
                            <li><strong>Venta/Cierre:</strong> {formatear_numero(data['PRECIO_VENTA_CIERRE'])}€ ({data['FECHA_VENTA_CIERRE']})</li>
                            <li><strong>Beneficio Cerrado:</strong> {formatear_beneficio(data['BENEFICIO_ULTIMA_OP'])} (Base 10.000€)</li>
                        </ul>
                    </div>
                """
            else:
                ultima_op_html = "<div class='position-box-op' style='background-color: #e9ecef;'><p style='font-size:0.9em; margin-top:5px;'>No se detectó un ciclo completo (Compra->Venta) reciente.</p></div>"

            # Bloque de Posición ABIERTA (Compra Actual)
            if data['COMPRADO'] == "SI":
                posicion_actual_html = f"""
                    <div class='position-box-op' style='border: 2px solid #28a745; background-color: #d4edda;'>
                        <b>POSICIÓN ACTUAL: COMPRADO (✅)</b>
                        <ul class="op-list">
                            <li><strong>Precio de Entrada:</strong> {formatear_numero(data['PRECIO_COMPRA'])}€ ({data['FECHA_COMPRA']})</li>
                            <li><strong>Precio Actual:</strong> {formatear_numero(data['PRECIO_ACTUAL'])}€</li>
                            <li><strong>Beneficio Actual (Simulado):</strong> {formatear_beneficio(data['BENEFICIO_ACTUAL'])} (Base 10.000€)</li>
                        </ul>
                    </div>
                """
            else:
                posicion_actual_html = """
                    <div class='position-box-op' style='border: 2px solid #dc3545; background-color: #f8d7da;'>
                        <b>POSICIÓN ACTUAL: NO COMPRADO (❌)</b>
                    </div>
                """
            
            
            # --- FILAS DE REPORTE CON OBSERVACIÓN SEMANAL EN DETALLE ---
            html_table_styles += f"""
                        <tr class="main-row" data-index="{i}" data-name="{data['NOMBRE_EMPRESA'].upper()}" data-ticker="{data['TICKER'].upper()}" data-filtro="{filtro_id}">
                            <td class="{celda_empresa_class}">{nombre_con_precio}</td>
                            <td>{data['TENDENCIA_ACTUAL']}</td>
                            <td class="{clase_oportunidad}">{data['OPORTUNIDAD']}</td>
                            <td>{data['COMPRA_SI']}</td>
                            <td>{data['VENDE_SI']}</td>
                            <td><span class="expand-button" onclick="toggleDetails({i})">Ver más...</span></td>
                        </tr>
                        <tr class="collapsible-row detailed-row-{i}" data-filtro="{filtro_id}">
                            <td colspan="6">
                                <div class="collapsible-content-container">
                                    
                                    <div class="col-op-data">
                                        {posicion_actual_html}
                                    </div>

                                    <div class="col-op-data">
                                        {ultima_op_html}
                                    </div>
                                    
                                    <div class="col-op-data">
                                        <div class='position-box-op'>
                                            <b>EMA/SR</b><br>
                                            EMA ({data['TIPO_EMA']}): <span style="font-weight:bold;">{formatear_numero(data['VALOR_EMA'])}€</span><br>
                                            S1/R1: {formatear_numero(data['SOPORTE_1'])}€ / {formatear_numero(data['RESISTENCIA_1'])}€
                                        </div>
                                        <div class='position-box-op' style="margin-top:10px;">
                                            <b>Análisis Semanal (SMI)</b><br>
                                            {data['OBSERVACION_SEMANAL']}
                                        </div>
                                    </div>
                                    
                                </div>
                            </td>
                        </tr>
                        <tr class="observaciones-row detailed-row-{i}" data-filtro="{filtro_id}">
                            <td colspan="6">{observaciones}</td>
                        </tr>
            """
            previous_orden_grupo = current_orden_grupo
    
    html_table_styles += """
                    </tbody>
                </table>
            </div>
    """
    return html_table_styles


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

            return (orden_interna, empresa['NOMBRE_EMPRESA'])

        datos_ordenados = sorted(datos_completos, key=obtener_clave_ordenacion)
        
        now_utc = datetime.utcnow()
        hora_actual = (now_utc + timedelta(hours=2)).strftime('%H:%M')
        
        
        # ******************************************************************************
        # ******************** MODIFICACIÓN DE LA SECCIÓN HTML *************************
        # ******************************************************************************
        
        # 1. ESTILOS CSS INLINE (NECESARIOS)
        html_styles = f"""
            <style>
            * {{
                box-sizing: border-box !important; 
            }}
            
            .main-container {{
                max-width: 1200px;
                width: 95%;
                margin: 20px auto;
                background-color: #f8f9fa; 
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            /* Estilo del título IBEXIA.ES (Tipo Google - Ajustado a ibexiaES) */
            h2 {{
                text-align: center;
                font-size: 3.5em; /* Tamaño grande */
                font-weight: 300;
                margin-top: 20px;
                margin-bottom: 20px; 
                letter-spacing: -0.05em;
            }}
            .google-style {{
                font-family: 'Product Sans', 'Arial', sans-serif; 
                line-height: 1; 
            }}
            .google-style span {{
                font-weight: 700;
                border-radius: 25px; 
                display: inline-block; 
                padding: 0 1px;
            }}
            .google-style .i1, .google-style .b1, .google-style .e1, 
            .google-style .x1, .google-style .i2, .google-style .a1 {{ color: #26A699; }} 
            .google-style .es-final {{
                font-size: 0.5em; 
                font-weight: 100;
                color: #00BCD4; 
                vertical-align: bottom; 
                margin-left: -0.1em; 
            }}
            p {{
                color: #6c757d;
                text-align: left;
                font-size: 0.9em;
            }}
            
            /* Contenedor de Pestañas/Bloques */
            #tab-group {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 10px;
                margin: 20px 0;
                border-bottom: 2px solid #dee2e6;
                padding-bottom: 10px;
            }}
            .tab-button {{
                text-decoration: none;
                padding: 12px 18px;
                border: 1px solid #dee2e6;
                border-radius: 8px 8px 0 0;
                font-size: 0.9em;
                font-weight: 600;
                color: #495057;
                background-color: #e9ecef;
                transition: all 0.3s ease;
                cursor: pointer;
                flex-grow: 1; /* Permite que los botones crezcan si es necesario */
                max-width: 200px;
                text-align: center;
            }}
            .tab-button.active {{
                color: white;
                background-color: #26A699; /* Color principal de ibexiaES */
                border-color: #26A699;
                border-bottom: none;
                box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
            }}
            .tab-button:hover:not(.active) {{
                background-color: #d8e6f1;
            }}
            .tab-content {{
                padding: 15px 0;
                display: none;
            }}
            .tab-content.active {{
                display: block;
            }}
            
            /* ESTILO DEL CAMPO DE BÚSQUEDA TIPO GOOGLE */
            #search-container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                margin-bottom: 20px;
            }}
            #searchInput {{
                width: 90%; 
                max-width: 600px;
                padding: 12px 20px;
                font-size: 1.1em;
                border: 1px solid #dfe1e5;
                border-radius: 24px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }}
            #searchInput:focus {{
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
                border-color: #4285F4; 
                outline: none;
            }}
            #update-date {{
                font-size: 0.85em;
                color: #6c757d; 
                margin-top: 10px; 
                text-align: center;
            }}
            .update-info-2h {{
                color: #1A237E; 
                font-weight: bold;
                display: block; /* Mover a línea separada en móviles */
            }}
            
            /* Estilos de la Tabla Principal */
            .table-container {{
                overflow-x: auto;
                overflow-y: auto;
                height: auto; 
                max-height: 70vh; /* Límite de altura para la tabla principal */
                position: relative;
                display: none; 
                max-width: 100%;
                border: 1px solid #dee2e6; /* Borde añadido */
                border-radius: 8px;
            }}
            table {{
                width: 100%;
                min-width: 800px; /* Ancho mínimo para la tabla principal */
                table-layout: auto;
                border-collapse: collapse;
                font-size: 0.85em;
            }}
            th, td {{
                border: 1px solid #e9ecef;
                padding: 8px 6px; /* Aumentar un poco el padding */
                text-align: center;
                vertical-align: middle;
                white-space: normal;
                line-height: 1.3;
            }}
            th {{
                background-color: #1A237E; /* Azul oscuro */
                color: white;
                font-weight: 600;
                position: sticky;
                top: 0;
                z-index: 10;
                white-space: nowrap;
            }}
            .compra {{ color: #155724; font-weight: bold; }}
            .venta {{ color: #721c24; font-weight: bold; }}
            .riesgo-compra {{ color: #856404; font-weight: bold; }}
            .vigilar {{ color: #004085; font-weight: bold; }}
            
            .green-cell {{ background-color: #d4edda; }}
            .red-cell {{ background-color: #f8d7da; }}
            .yellow-cell {{ background-color: #fff3cd; }}
            .yellow-cell-light {{ background-color: #e2f7ff; }} /* Nuevo color para VIGILAR */
            
            .category-header td {{
                background-color: #495057;
                color: white;
                font-size: 1.0em;
                font-weight: bold;
                text-align: center;
                padding: 8px;
                border: none;
            }}
            
            .observaciones-row td {{
                background-color: #f9f9f9;
                text-align: left;
                font-size: 0.8em;
                border: 1px solid #e9ecef;
            }}
            .stacked-text {{
                line-height: 1.2;
                font-size: 0.85em;
            }}
            .expand-button {{
                cursor: pointer;
                color: #26A699;
                font-weight: bold;
                text-decoration: underline;
                white-space: nowrap;
            }}
            
            /* Estilos para el contenido detallado */
            .collapsible-content-container {{
                display:flex; 
                justify-content:space-around; 
                align-items:flex-start; 
                padding: 10px; 
                flex-wrap: wrap;
                gap: 10px;
            }}
            .col-op-data {{
                flex: 1 1 300px; /* Flex-grow, flex-shrink, base width */
                min-width: 250px;
                text-align: left; 
            }}
            .position-box-op {{
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 8px;
                margin: 5px 0;
                text-align: left;
                background-color: #ffffff;
                font-size: 0.9em;
            }}
            .op-list {{
                list-style: none;
                padding-left: 10px;
                margin: 5px 0;
            }}

            /* Estilos para la tabla de Posiciones Abiertas */
            .open-positions-container table th {{
                background-color: #1A237E; /* Azul oscuro corporativo */
                color: white;
                font-weight: 700;
            }}
            .open-positions-container table td {{
                font-size: 0.9em;
                padding: 10px 6px;
                text-align: center;
            }}
            .venta-op {{ background-color: #dc3545; color: white; }}
            .vigilar-op {{ background-color: #ffc107; color: #343a40; }}
            .compra-op {{ background-color: #28a745; color: white; }}

            
            /* Media Queries para Responsividad */
            @media (max-width: 900px) {{
                .tab-button {{
                    font-size: 0.8em;
                    padding: 10px 10px;
                    max-width: none;
                }}
                h2 {{
                    font-size: 2.5em;
                }}
                .collapsible-content-container {{
                    flex-direction: column;
                    gap: 5px;
                }}
                .col-op-data {{
                    min-width: 100%;
                    flex-basis: 100%;
                }}
                .table-container table {{
                    min-width: 600px; /* Reducir un poco el mínimo */
                }}
                .update-info-2h {{
                    margin-left: 0;
                    margin-top: 5px;
                }}
            }}
            
            @media (max-width: 600px) {{
                .tab-button {{
                    font-size: 0.7em;
                    padding: 8px 6px;
                }}
                .stacked-text {{
                    font-size: 0.8em;
                }}
                .table-container table {{
                    min-width: 500px;
                }}
            }}
            

        </style>
        """
        
        # 2. CUERPO HTML (Contenido insertable en WordPress)
        html_content = f"""
            <div class="main-container">
                <h2 class="google-style">
                    <span class="i1">i</span>
                    <span class="b1">b</span>
                    <span class="e1">e</span>
                    <span class="x1">x</span>
                    <span class="i2">i</span>
                    <span class="a1">a</span>
                    <span class="es-final">ES</span>
                </h2>
                
                <div id="update-date">
                    <span>
                        Fecha de actualización: {datetime.today().strftime('%d de %B %Y').replace(datetime.today().strftime('%B'), ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'][datetime.today().month-1])} hora {hora_actual} (CET)
                    </span>
                    <span class="update-info-2h">
                        Los resultados están actualizados cada dos horas.
                    </span>
                </div>
                
                <div id="tab-group">
                    <button class="tab-button active" onclick="showTab('buscar-valor', this)">🔎 Buscar Valor</button>
                    <button class="tab-button" onclick="showTab('ver-todos', this)">📊 Ver Todos</button>
                    <button class="tab-button" onclick="showTab('posiciones-abiertas', this)">✅ Posiciones Abiertas</button>
                    <button class="tab-button" onclick="showTab('oportunidades-compra', this)">💰 Oport. Compra</button>
                    <button class="tab-button" onclick="showTab('riesgos-venta', this)">🚨 Riesgos Venta</button>
                    <button class="tab-button" onclick="showTab('valores-vigilar', this)">⚠️ Valores a Vigilar</button>
                </div>
                
                <div id="buscar-valor" class="tab-content active">
                    <div id="search-container">
                        <input type="text" id="searchInput" placeholder="Escribe el nombre o ticker (Ej: Inditex, SAN.MC)...">
                    </div>
                    <div id="search-results-table">
                        </div>
                </div>

                <div id="ver-todos" class="tab-content">
                    <p style="text-align:center; font-style:italic;">Mostrando el análisis completo de todos los valores, ordenado por prioridad de Compra a Venta.</p>
                </div>
                
                <div id="posiciones-abiertas" class="tab-content">
                    <h3 style="text-align: center; color: #1A237E; margin-top: 10px; margin-bottom: 20px; font-size: 1.5em; border-bottom: 2px solid #e9ecef; padding-bottom: 5px;">
                        <i class="fas fa-check-circle" style="color:#28a745; margin-right: 10px;"></i>
                        Cartera Actual (IBEXIAes)
                    </h3>
                    {generar_tabla_posiciones_abiertas(datos_completos)}
                </div>

                <div id="oportunidades-compra" class="tab-content">
                    <p style="text-align:center; font-style:italic;">Mostrando solo empresas con señal de **Posibilidad de Compra** (incluyendo Riesgo).</p>
                </div>
                <div id="riesgos-venta" class="tab-content">
                    <p style="text-align:center; font-style:italic;">Mostrando solo empresas en zona de **Riesgo de Venta**.</p>
                </div>
                <div id="valores-vigilar" class="tab-content">
                    <p style="text-align:center; font-style:italic;">Mostrando valores en situación **Intermedia** o de **Vigilancia**.</p>
                </div>

                <div id="main-analysis-container">
                    {generar_tabla_analisis(datos_ordenados)}
                </div>

            </div>
        """
        
        # 3. SCRIPT JAVASCRIPT (NECESARIO)
        html_script = """
            <script>
                // Función de acordeón para las filas individuales (Debe ser global)
                function toggleDetails(index) {
                    var detailedRows = document.querySelectorAll('.detailed-row-' + index);
                    
                    detailedRows.forEach(row => {
                        // Usar 'table-row' para filas de tabla
                        row.style.display = row.style.display === "table-row" ? "none" : "table-row";
                    });
                }
                
                // Variable global para almacenar el estado de la tabla (Ver Todos o Filtro)
                let currentViewMode = 'buscar-valor';
                
                // --- LÓGICA DE PESTAÑAS Y VISTAS ---
                function showTab(tabId, buttonElement) {
                    // Ocultar todos los contenidos de las pestañas
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                    // Desactivar todos los botones
                    document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));

                    // Mostrar el contenido de la pestaña activa y activar el botón
                    document.getElementById(tabId).classList.add('active');
                    if (buttonElement) {
                        buttonElement.classList.add('active');
                    }
                    
                    // Actualizar el modo de vista actual
                    currentViewMode = tabId;
                    
                    // Mostrar u ocultar la tabla principal y el contenedor de búsqueda según la pestaña
                    const tableContainer = document.getElementById("tableContainer");
                    const searchInput = document.getElementById("searchInput");
                    const mainAnalysisContainer = document.getElementById("main-analysis-container");

                    if (tabId === 'buscar-valor') {
                        // En Buscar Valor, solo se muestra la tabla si hay texto en el input
                        tableContainer.style.display = (searchInput.value.trim().length > 0) ? "block" : "none";
                        searchInput.focus();
                    } else if (tabId === 'posiciones-abiertas') {
                        // En Posiciones Abiertas, se oculta la tabla principal y se muestra el contenido HTML estático
                        tableContainer.style.display = "none";
                    } else {
                        // Para Ver Todos y Filtros (Compra, Venta, Vigilar), la tabla principal siempre es visible
                        tableContainer.style.display = "block";
                        // Limpiar el input de búsqueda en cualquier modo que no sea 'buscar-valor'
                        searchInput.value = ''; 
                        
                        // Aplicar filtro si no es 'Ver Todos'
                        filterView(tabId);
                    }
                    
                    // Asegurarse de que el scroll superior del contenedor principal sea 0 al cambiar de vista
                    mainAnalysisContainer.scrollTop = 0; 
                }

                // --- LÓGICA DE FILTRADO (BUSCADOR Y BOTONES) ---
                let filterTimeout;
                const searchInput = document.getElementById("searchInput");
                const tableContainer = document.getElementById("tableContainer");
                const table = document.getElementById("myTable");
                
                if (table) {
                    const tbody = table.querySelector('tbody');
                    const rows = Array.from(tbody.getElementsByTagName("tr"));
                
                    // 1. FILTRADO POR TEXTO (BUSCADOR)
                    function filterTableByText() {
                        clearTimeout(filterTimeout);
                        
                        filterTimeout = setTimeout(() => {
                            const filter = searchInput.value.toUpperCase().trim();
                            const showTable = filter.length > 0;
                            
                            // Mostrar u ocultar la tabla
                            tableContainer.style.display = showTable ? "block" : "none";
                            if (!showTable) return;

                            for (let i = 0; i < rows.length; i++) {
                                const row = rows[i];
                                
                                // Ocultar filas de detalle, separadores y categorías por defecto
                                if (row.classList.contains("collapsible-row") || 
                                    row.classList.contains("observaciones-row") ||
                                    row.classList.contains("separator-row") || 
                                    row.classList.contains("category-header")) 
                                {
                                    row.style.display = "none";
                                    continue;
                                }
                                
                                // Procesar Filas Principales
                                if (row.classList.contains("main-row")) {
                                    const name = row.getAttribute('data-name');
                                    const ticker = row.getAttribute('data-ticker');
                                    
                                    const isMatch = (name.indexOf(filter) > -1) || (ticker.indexOf(filter) > -1);

                                    row.style.display = isMatch ? "table-row" : "none";
                                }
                            }
                            // Las categorías y separadores no se muestran en el modo búsqueda
                        }, 200);
                    }

                    // 2. FILTRADO POR VISTA (BOTONES)
                    window.filterView = function(tabId) {
                        // Mapeo de IDs de pestaña a valores de data-filtro
                        const filterMap = {
                            'ver-todos': null, // No filtrar, mostrar todo
                            'oportunidades-compra': 'COMPRA',
                            'riesgos-venta': 'VENTA',
                            'valores-vigilar': 'VIGILAR'
                        };
                        const filterValue = filterMap[tabId];

                        for (let i = 0; i < rows.length; i++) {
                            const row = rows[i];
                            const rowFilter = row.getAttribute('data-filtro');

                            // 1. FILAS PRINCIPALES (main-row)
                            if (row.classList.contains("main-row")) {
                                const isMatch = (filterValue === null) || (rowFilter === filterValue);
                                row.style.display = isMatch ? "table-row" : "none";
                            } 
                            // 2. FILAS DE DETALLE (collapsible-row, observaciones-row)
                            else if (row.classList.contains("collapsible-row") || row.classList.contains("observaciones-row")) {
                                // Las filas de detalle deben estar ocultas por defecto en todos los modos, 
                                // y solo se muestran con el botón "Ver más..."
                                row.style.display = "none"; 
                            }
                            // 3. CABECERAS Y SEPARADORES (category-header, separator-row)
                            else if (row.classList.contains("category-header") || row.classList.contains("separator-row")) {
                                // Por defecto, se ocultan y se re-evalúan a continuación, a menos que estemos en 'Ver Todos'
                                row.style.display = "none";
                            }
                        }
                        
                        // Mostrar categorías y separadores solo en la vista 'Ver Todos'
                        if (tabId === 'ver-todos') {
                            showCategoryHeaders();
                            showSeparatorRows();
                        }
                    }
                    
                    // Lógica para mostrar las cabeceras de categoría
                    function showCategoryHeaders() {
                        const categoryHeaders = document.querySelectorAll('.category-header');
                        categoryHeaders.forEach(header => {
                            let nextSibling = header.nextElementSibling;
                            let hasVisibleRows = false;
                            while(nextSibling && !nextSibling.classList.contains('category-header')) {
                                // Si la siguiente fila principal está visible, mostrar la cabecera
                                if (nextSibling.classList.contains('main-row') && nextSibling.style.display !== 'none') {
                                    hasVisibleRows = true;
                                    break;
                                }
                                nextSibling = nextSibling.nextElementSibling;
                            }
                            header.style.display = hasVisibleRows ? "table-row" : "none";
                        });
                    }
                    
                    // Lógica para mostrar los separadores
                    function showSeparatorRows() {
                        const separatorRows = document.querySelectorAll('.separator-row');
                        separatorRows.forEach(separator => {
                            const prev = separator.previousElementSibling;
                            const next = separator.nextElementSibling;
                            
                            // Un separador es visible si la categoría anterior y la siguiente están visibles
                            const prevVisible = prev && prev.style.display === "table-row" && prev.classList.contains("category-header");
                            const nextVisible = next && next.style.display === "table-row" && next.classList.contains("category-header");

                            if (prevVisible && nextVisible) {
                                separator.style.display = "table-row";
                            } else {
                                separator.style.display = "none";
                            }
                        });
                    }
                }
                
                // Asegurar que el script se ejecute cuando el DOM esté listo
                document.addEventListener('DOMContentLoaded', function() {
                    // Inicializar la vista en 'Buscar Valor'
                    showTab('buscar-valor', document.querySelector('.tab-button.active'));

                    if (searchInput) {
                        searchInput.addEventListener("keyup", filterTableByText);
                    }
                });
            </script>
        """
        
        # El HTML COMPLETO para el correo es la concatenación de todo.
        html_para_email_body = f"""
            {html_styles}
            {html_content}
            {html_script}
        """

        
        asunto = f"🔔 Alertas y Oportunidades IBEXIA: {len(datos_ordenados)} oportunidades detectadas hoy {datetime.today().strftime('%d/%m/%Y')}"
        nombre_archivo_base = f"reporte_ibexia_{datetime.today().strftime('%Y%m%d')}"

        enviar_email_con_adjunto(html_para_email_body, asunto, nombre_archivo_base)
        
        # Devolver el bloque único con <style>, contenido y <script> para la inserción en WordPress.
        return html_styles + html_content + html_script 

    except Exception as e:
        print(f"❌ Error al ejecutar el script principal: {e}")
        return None

if __name__ == '__main__':
    generar_reporte()
