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
    # Función auxiliar para calcular el beneficio monetario
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

# ******************************************************************************
# ************** LÓGICA DE COMPRA/VENTA EXTRAÍDA DE LEER_GOOGLE_SHEETS *********************
# ******************************************************************************
def analizar_ultima_operacion(df_hist):
    """
    Simula la operativa basada en el SMI (como en leer_google_sheets.py) 
    para encontrar el estado de la última posición (abierta o cerrada).
    """
    df = df_hist.dropna(subset=['SMI']).copy()
    if len(df) < 3:
        return {
            'comprado_status': "NO", 'precio_compra': "N/A", 'fecha_compra': "N/A",
            'precio_venta_cierre': "N/A", 'fecha_venta_cierre': "N/A", 'beneficio_ultima_op': "N/A"
        }

    smis = df['SMI'].values
    precios = df['Close'].values
    fechas = df.index.strftime('%d/%m/%Y').tolist()
    current_price = precios[-1]
    
    # Calcular la pendiente del SMI
    pendientes_smi = np.diff(smis)
    pendientes_smi = np.insert(pendientes_smi, 0, 0) # Rellenar el primer elemento

    all_compras = []
    all_ventas = []
    posicion_abierta = False
    
    # Iterar hacia adelante para simular la operativa y detectar señales
    for i in range(1, len(df)):
        
        pendiente_actual = pendientes_smi[i]
        pendiente_anterior = pendientes_smi[i-1]
        smi_anterior = smis[i-1]
        
        # Señal de Compra: SMI gira al alza (cruza de <=0 a >0) Y no está en sobrecompra
        if pendiente_actual > 0 and pendiente_anterior <= 0 and smi_anterior < 40:
            if not posicion_abierta:
                posicion_abierta = True
                ultima_compra = {'fecha': fechas[i], 'precio': precios[i]}
                all_compras.append(ultima_compra)
        
        # Señal de Venta: SMI gira a la baja (cruza de >=0 a <0)
        elif pendiente_actual < 0 and pendiente_anterior >= 0:
            if posicion_abierta:
                posicion_abierta = False
                # Se asocia a la última compra abierta
                if all_compras:
                    ultima_venta = {'fecha': fechas[i], 'precio': precios[i], 'compra_asociada': all_compras[-1]}
                    all_ventas.append(ultima_venta)

    # 2. Determinar la última operación
    
    if posicion_abierta and all_compras:
        # La última señal fue de COMPRA y la posición sigue abierta
        compra_actual = all_compras[-1]
        
        # Cálculo del beneficio simulado actual
        beneficio_simulado = calcular_beneficio_perdida(compra_actual['precio'], current_price)

        return {
            'comprado_status': "SI",
            'precio_compra': compra_actual['precio'],
            'fecha_compra': compra_actual['fecha'],
            'precio_venta_cierre': "N/A",
            'fecha_venta_cierre': "N/A",
            'beneficio_ultima_op': beneficio_simulado # Beneficio actual
        }
    
    elif all_ventas:
        # La última señal fue de VENTA y la posición está cerrada
        venta_cerrada = all_ventas[-1]
        compra_asociada = venta_cerrada['compra_asociada']
        
        # Cálculo del beneficio de la última operación cerrada
        beneficio_cerrado = calcular_beneficio_perdida(compra_asociada['precio'], venta_cerrada['precio'])
        
        return {
            'comprado_status': "NO",
            'precio_compra': compra_asociada['precio'], # Muestra la compra asociada a la última venta
            'fecha_compra': compra_asociada['fecha'],
            'precio_venta_cierre': venta_cerrada['precio'],
            'fecha_venta_cierre': venta_cerrada['fecha'],
            'beneficio_ultima_op': beneficio_cerrado
        }
    
    else:
        # No se encontraron señales de compra/venta en el historial
        return {
            'comprado_status': "NO", 'precio_compra': "N/A", 'fecha_compra': "N/A",
            'precio_venta_cierre': "N/A", 'fecha_venta_cierre': "N/A", 'beneficio_ultima_op': "N/A"
        }
# ******************************************************************************
# ************ FIN LÓGICA DE COMPRA/VENTA EXTRAÍDA DE LEER_GOOGLE_SHEETS *******************
# ******************************************************************************
        
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
        
        # ******************************************************************************
        # *** LLAMADA A LA LÓGICA DE DETECCIÓN DE ÚLTIMA OPERACIÓN (COMPRA/VENTA) ***
        # ******************************************************************************
        resultados_operacion = analizar_ultima_operacion(hist_extended)

        comprado_status = resultados_operacion['comprado_status']
        precio_compra = resultados_operacion['precio_compra']
        fecha_compra = resultados_operacion['fecha_compra']
        
        precio_venta_cierre = resultados_operacion['precio_venta_cierre']
        fecha_venta_cierre = resultados_operacion['fecha_venta_cierre']
        beneficio_ultima_op = resultados_operacion['beneficio_ultima_op'] # Resultado de la última operación

        # Si la posición está abierta ('SI'), el beneficio_ultima_op ya contiene el beneficio actual simulado.
        beneficio_actual = beneficio_ultima_op if comprado_status == "SI" else "N/A"

        # ******************************************************************************
        # ************* FIN LLAMADA A LÓGICA DE COMPRA/VENTA ************************
        # ******************************************************************************


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
        print(f"❌ Error al obtener datos de {ticker}: {e}. Saltando a...")
        return None

def clasificar_empresa(datos):
    # ******************************************************************
    # ******* ESTA FUNCIÓN NO SE MODIFICA, SE MANTIENE EL ORIGINAL ******
    # ******************************************************************
    recomendacion = ""
    nivel_confianza = 0
    observacion = ""
    estado_diario = ""
    
    # 1. Recomendación Basada en SMI Diario
    if datos["ESTADO_SMI"] == "Sobreventa" and datos["TENDENCIA_ACTUAL"] == "Subiendo" and datos["PENDIENTE"] > 0.1:
        recomendacion = "COMPRA"
        nivel_confianza += 5
        estado_diario = "Fuerte señal de compra (SMI en sobreventa girando al alza)."
    elif datos["ESTADO_SMI"] == "Sobreventa" and datos["TENDENCIA_ACTUAL"] == "Plano" and datos["PENDIENTE"] >= -0.1:
        recomendacion = "COMPRA (Cautelosa)"
        nivel_confianza += 3
        estado_diario = "Señal de posible compra (SMI en sobreventa, pero sin giro fuerte)."
    elif datos["ESTADO_SMI"] == "Sobrecompra" and datos["TENDENCIA_ACTUAL"] == "Bajando" and datos["PENDIENTE"] < -0.1:
        recomendacion = "VENTA/CIERRE"
        nivel_confianza += 5
        estado_diario = "Fuerte señal de venta o cierre (SMI en sobrecompra girando a la baja)."
    elif datos["ESTADO_SMI"] == "Sobrecompra" and datos["TENDENCIA_ACTUAL"] == "Plano" and datos["PENDIENTE"] <= 0.1:
        recomendacion = "VENTA/CIERRE (Cautelosa)"
        nivel_confianza += 3
        estado_diario = "Señal de posible venta o cierre (SMI en sobrecompra, pero sin caída fuerte)."
    elif datos["ESTADO_SMI"] == "Intermedio":
        if datos["TENDENCIA_ACTUAL"] == "Subiendo" and datos["PENDIENTE"] > 0.1:
            recomendacion = "MANTENER"
            nivel_confianza += 2
            estado_diario = "Tendencia alcista en zona intermedia. Mantener."
        elif datos["TENDENCIA_ACTUAL"] == "Bajando" and datos["PENDIENTE"] < -0.1:
            recomendacion = "MANTENER (Espera)"
            nivel_confianza += 1
            estado_diario = "Tendencia bajista en zona intermedia. Esperar."
        else:
            recomendacion = "NEUTRO"
            nivel_confianza += 0
            estado_diario = "Movimiento plano en zona intermedia. No operar."
    else:
        recomendacion = "NEUTRO"
        nivel_confianza += 0
        estado_diario = "Condiciones no claras para operar."
        
    # 2. Modificación por SMI Semanal (Advertencia)
    if datos["ESTADO_SMI_SEMANAL"] == "Sobrecompra" and recomendacion.startswith("COMPRA"):
        recomendacion = recomendacion.replace("COMPRA", "COMPRA (ALTO RIESGO)")
        nivel_confianza -= 2
        datos["ADVERTENCIA_SEMANAL"] = "SMI Semanal en Sobrecompra"
        
    if datos["ESTADO_SMI_SEMANAL"] == "Sobreventa" and recomendacion.startswith("VENTA"):
        recomendacion = recomendacion.replace("VENTA/CIERRE", "VENTA/CIERRE (BAJO RIESGO)")
        nivel_confianza -= 1
        datos["ADVERTENCIA_SEMANAL"] = "SMI Semanal en Sobreventa" # Esto es menos una advertencia y más una confirmación de soporte

    # 3. Consideración de Posición Abierta
    if datos["COMPRADO"] == "SI":
        # Si ya está comprado, la recomendación se orienta a la gestión de la posición
        if datos["ESTADO_SMI"] == "Sobrecompra" and datos["TENDENCIA_ACTUAL"] == "Bajando":
             recomendacion = "ALERTA CIERRE"
             observacion = f"Posición abierta desde {datos['FECHA_COMPRA']}. SMI indica posible corrección (sobrecompra + bajando)."
        elif recomendacion.startswith("VENTA"):
             recomendacion = "ALERTA CIERRE"
             observacion = f"Posición abierta desde {datos['FECHA_COMPRA']}. SMI diario indica giro de venta."
        else:
             recomendacion = "MANTENER POSICIÓN"
             observacion = f"Posición abierta desde {datos['FECHA_COMPRA']} con beneficio actual: {formatear_beneficio(datos['BENEFICIO_ACTUAL'])}."
             
    # 4. Observaciones Finales
    observacion_final = f"{estado_diario} {datos['OBSERVACION_SEMANAL']}"

    return {
        "RECOMENDACION": recomendacion,
        "NIVEL_CONFIANZA": max(0, nivel_confianza),
        "OBSERVACION": observacion or observacion_final.strip()
    }


def formatear_valor(valor):
    if valor is None:
        return "N/A"
    if isinstance(valor, (int, float)):
        return formatear_numero(valor) + "€"
    return str(valor)

def generar_html(datos_analizados):
    
    # Convertimos a F-string para inyectar la fecha correctamente
    html_template_start = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte de Análisis Bursátil IBEXIA</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; color: #333; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: auto; background-color: #ffffff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); }}
        h1 {{ color: #007bff; text-align: center; margin-bottom: 20px; }}
        .header-info {{ text-align: center; margin-bottom: 30px; font-size: 1.1em; color: #555; }}
        .company-card {{ background-color: #f9f9f9; border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 20px; padding: 20px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); }}
        .company-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
        .company-header h2 {{ margin: 0; color: #333; }}
        .recommendation {{ font-weight: bold; padding: 5px 10px; border-radius: 5px; color: white; }}
        .COMPRA {{ background-color: #28a745; }}
        .VENTA-CIERRE {{ background-color: #dc3545; }}
        .MANTENER, .NEUTRO {{ background-color: #ffc107; color: #333; }}
        .ALERTA-CIERRE {{ background-color: #ff5722; }}
        .COMPRA-ALTO-RIESGO {{ background-color: #ff9800; color: #333; }}
        .VENTA-CIERRE-BAJO-RIESGO {{ background-color: #4CAF50; }}
        
        .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; margin-top: 15px; }}
        .stat-box {{ background-color: #f0f8ff; padding: 10px; border-radius: 5px; border-left: 4px solid #007bff; }}
        .stat-box strong {{ display: block; font-size: 0.9em; color: #555; }}
        .stat-box span {{ font-size: 1.2em; font-weight: bold; color: #333; }}
        
        .data-table-container {{ 
            overflow-x: auto; 
            margin-top: 20px; 
            border: 1px solid #ccc;
            border-radius: 5px;
        }}
        .data-table {{ width: 100%; border-collapse: collapse; white-space: nowrap; }}
        .data-table th, .data-table td {{ border: 1px solid #e9ecef; padding: 12px 15px; text-align: left; }}
        .data-table th {{ background-color: #007bff; color: white; font-weight: bold; text-transform: uppercase; font-size: 0.9em; }}
        .data-table tr:nth-child(even) {{ background-color: #f7f7f7; }}
        .data-table tr:hover {{ background-color: #e9ecef; }}
        .scroll-top {{ overflow-x: scroll; height: 18px; opacity: 0; }}
        
        .filter-container {{ margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center; }}
        .filter-container input[type="text"] {{ padding: 8px; border: 1px solid #ccc; border-radius: 5px; width: 250px; }}

        .operacion-historica {{
            margin-top: 15px;
            padding: 10px;
            background-color: #e9f7ef; /* Light green background */
            border-left: 5px solid #28a745;
            border-radius: 4px;
        }}

        .operacion-historica strong {{
            color: #333;
        }}

        .alerta-semanal {{
            color: #dc3545;
            font-weight: bold;
            margin-left: 10px;
        }}
        
    </style>
</head>
<body>
    <div class="container">
        <h1>🔔 Reporte Diario de Análisis Bursátil IBEXIA</h1>
        <div class="header-info">
            Reporte generado el <strong>{datetime.today().strftime('%d/%m/%Y')}</strong>
        </div>

        <div class="filter-container">
            <input type="text" id="searchInput" placeholder="Filtrar por nombre o ticker...">
            <div>Mostrando {len(datos_analizados)} oportunidades</div>
        </div>
        
        <div id="scrollTop" class="scroll-top">
            <div style="width: 3000px; height: 1px;"></div>
        </div>
        
        <div class="data-table-container" id="tableContainer">
            <table class="data-table" id="data-table">
                <thead>
                    <tr>
                        <th>Empresa</th>
                        <th>Precio Actual</th>
                        <th>Recomendación</th>
                        <th>Confianza (0-5)</th>
                        <th>Estado SMI</th>
                        <th>SMI Hoy</th>
                        <th>SMI Ayer</th>
                        <th>Pendiente</th>
                        <th>Soporte 1</th>
                        <th>Resistencia 1</th>
                        <th>Tipo EMA (100)</th>
                        <th>Valor EMA (100)</th>
                        <th>P. Aplanamiento</th>
                        <th>Estado SMI Semanal</th>
                        <th>Última Compra (Precio)</th>
                        <th>Última Compra (Fecha)</th>
                        <th>Última Venta (Precio)</th>
                        <th>Última Venta (Fecha)</th>
                        <th>Beneficio Operación</th>
                        <th>Observación</th>
                    </tr>
                </thead>
                <tbody id="tableBody">
    """
    
    html_rows = ""
    
    for dato in datos_analizados:
        
        recomendacion_class = dato['RECOMENDACION'].split(' ')[0].replace("/", "-")
        if recomendacion_class == "VENTA":
            recomendacion_class = "VENTA-CIERRE"
        
        # Determine the color and text for the benefit column
        beneficio_op_formateado = formatear_beneficio(dato["BENEFICIO_ULTIMA_OP"])
        
        # Formatting for display
        precio_compra_display = formatear_valor(dato["PRECIO_COMPRA"]) if dato["COMPRADO"] == "SI" else "N/A"
        fecha_compra_display = dato["FECHA_COMPRA"] if dato["COMPRADO"] == "SI" else "N/A"
        precio_venta_display = formatear_valor(dato["PRECIO_VENTA_CIERRE"]) if dato["COMPRADO"] == "NO" else "N/A"
        fecha_venta_display = dato["FECHA_VENTA_CIERRE"] if dato["COMPRADO"] == "NO" else "N/A"

        # If Comprado=SI, the last operation benefit is the current simulated profit
        if dato["COMPRADO"] == "SI":
             # Use the current benefit for the Operación column, as it's the only one
             beneficio_op_formateado = formatear_beneficio(dato["BENEFICIO_ACTUAL"])
             precio_compra_display = formatear_valor(dato["PRECIO_COMPRA"])
             fecha_compra_display = dato["FECHA_COMPRA"]
             precio_venta_display = "N/A"
             fecha_venta_display = "N/A"
             
        # If Comprado=NO, the last operation benefit is the closed profit (BENEFICIO_ULTIMA_OP)
        elif dato["COMPRADO"] == "NO" and dato["FECHA_VENTA_CIERRE"] != "N/A":
             # Last operation was a close, show the purchase associated with that sale
             precio_compra_display = formatear_valor(dato["PRECIO_COMPRA"])
             fecha_compra_display = dato["FECHA_COMPRA"]
             precio_venta_display = formatear_valor(dato["PRECIO_VENTA_CIERRE"])
             fecha_venta_display = dato["FECHA_VENTA_CIERRE"]


        # Check if the SMI is "N/A" to avoid trying to format it
        smi_hoy_formateado = formatear_numero(dato["SMI_HOY"])
        smi_ayer_formateado = formatear_numero(dato["SMI_AYER"])
        
        html_rows += f"""
                    <tr class="{'comprado' if dato['COMPRADO'] == 'SI' else 'no-comprado'}" data-name="{dato['NOMBRE_EMPRESA']}" data-ticker="{dato['TICKER']}">
                        <td><strong>{dato['NOMBRE_EMPRESA']}</strong> ({dato['TICKER']})</td>
                        <td>{formatear_numero(dato['PRECIO_ACTUAL'])}€</td>
                        <td><span class="recommendation {recomendacion_class}">{dato['RECOMENDACION']}</span></td>
                        <td>{dato['NIVEL_CONFIANZA']}/5</td>
                        <td>{dato['ESTADO_SMI']}</td>
                        <td>{smi_hoy_formateado}</td>
                        <td>{smi_ayer_formateado}</td>
                        <td>{formatear_numero(dato['PENDIENTE'])}</td>
                        <td>{formatear_valor(dato['SOPORTE_1'])}</td>
                        <td>{formatear_valor(dato['RESISTENCIA_1'])}</td>
                        <td>{dato['TIPO_EMA']}</td>
                        <td>{formatear_numero(dato['VALOR_EMA'])}€</td>
                        <td>{formatear_valor(dato['PRECIO_APLANAMIENTO'])}</td>
                        <td>{dato['ESTADO_SMI_SEMANAL']} <span class="alerta-semanal">{'!ALERTA' if dato['ADVERTENCIA_SEMANAL'] != 'NO' else ''}</span></td>
                        <td>{precio_compra_display}</td>
                        <td>{fecha_compra_display}</td>
                        <td>{precio_venta_display}</td>
                        <td>{fecha_venta_display}</td>
                        <td>{beneficio_op_formateado}</td>
                        <td>{dato['OBSERVACION']}</td>
                    </tr>
                """

    # Esta sección se mantiene como cadena simple para evitar el SyntaxError de Python con JavaScript/f-strings.
    html_template_end = """
                </tbody>
            </table>
        </div>
        
        <div style="margin-top: 30px;">
            <p><strong>Nota sobre Beneficio Operación:</strong></p>
            <ul>
                <li>Si <strong>Compra</strong> es 'SI', este es el beneficio/pérdida actual de la posición abierta.</li>
                <li>Si <strong>Compra</strong> es 'NO', este es el beneficio/pérdida de la última operación cerrada (Buy & Sell).</li>
            </ul>
        </div>
    </div>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const searchInput = document.getElementById('searchInput');
            const tableBody = document.getElementById('tableBody');
            const rows = tableBody.getElementsByTagName('tr');
            const tableContainer = document.getElementById('tableContainer');
            const scrollTop = document.getElementById('scrollTop');

            searchInput.addEventListener('keyup', function() {
                const filter = searchInput.value.toUpperCase();
                for (let i = 0; i < rows.length; i++) {
                    const row = rows[i];
                    const name = row.getAttribute('data-name').toUpperCase();
                    const ticker = row.getAttribute('data-ticker').toUpperCase();
                    if (name.includes(filter) || ticker.includes(filter)) {
                        row.style.display = "";
                    } else {
                        row.style.display = "none";
                    }
                }
            });

            // Enfocar el campo de búsqueda al cargar (si es posible)
            if (searchInput) {
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
    
    return html_template_start + html_rows + html_template_end


def enviar_email_con_adjunto(html_body, asunto, nombre_archivo_base):
    
    smtp_server = os.getenv('SMTP_SERVER')
    smtp_port_str = os.getenv('SMTP_PORT') 
    email_user = os.getenv('EMAIL_USER')
    email_password = os.getenv('EMAIL_PASSWORD')
    email_recipient = os.getenv('EMAIL_RECIPIENT')

    # CORRECCIÓN DE ROBUSTEZ: Verifica que todas las variables existan antes de intentar usarlas
    if not all([smtp_server, smtp_port_str, email_user, email_password, email_recipient]):
        print("❌ Error: Faltan variables de entorno para el envío de correo. Asegúrate de que SMTP_SERVER, SMTP_PORT, EMAIL_USER, EMAIL_PASSWORD y EMAIL_RECIPIENT están configuradas. Saltando envío.")
        return
        
    # CORRECCIÓN DE ROBUSTEZ: Conversión segura del puerto a entero
    try:
        smtp_port = int(smtp_port_str)
    except ValueError:
        print(f"❌ Error: La variable de entorno SMTP_PORT ('{smtp_port_str}') no es un número entero válido. Saltando envío.")
        return

    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = email_recipient
    msg['Subject'] = asunto

    # Adjuntar el HTML como parte del cuerpo y como archivo
    html_part = MIMEText(html_body, 'html')
    msg.attach(html_part)
    
    archivo_html = MIMEBase('application', 'octet-stream')
    archivo_html.set_payload(html_body.encode('utf-8'))
    encoders.encode_base64(archivo_html)
    archivo_html.add_header('Content-Disposition', f'attachment; filename="{nombre_archivo_base}.html"')
    msg.attach(archivo_html)

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Para seguridad
        server.login(email_user, email_password)
        server.sendmail(email_user, email_recipient, msg.as_string())
        server.quit()
        print(f"✅ Correo enviado con éxito a {email_recipient}")
    except Exception as e:
        print(f"❌ Error al enviar el correo: {e}")

def generar_reporte():
    
    try:
        # Se asume que leer_google_sheets y obtener_datos_yfinance existen
        all_tickers = list(tickers.values())
        datos_analizados = []
        
        for ticker in all_tickers:
            datos = obtener_datos_yfinance(ticker)
            if datos:
                # Realizar el análisis avanzado
                analisis = clasificar_empresa(datos)
                datos.update(analisis)
                datos_analizados.append(datos)
                print(f"✅ Análisis completado para {datos['NOMBRE_EMPRESA']}.")
            # time.sleep(random.randint(2, 5)) # Espera aleatoria para evitar saturar la API

        if not datos_analizados:
            print("No se generaron datos de análisis para ninguna empresa. Abortando reporte.")
            return

        # Ordenar por Recomendación (COMPRA, VENTA/CIERRE, MANTENER, NEUTRO)
        orden_recomendacion = {"COMPRA": 4, "COMPRA (Cautelosa)": 3, "MANTENER": 2, "MANTENER (Espera)": 1, "NEUTRO": 0, "VENTA/CIERRE": -4, "VENTA/CIERRE (Cautelosa)": -3, "ALERTA CIERRE": -2, "COMPRA (ALTO RIESGO)": -1, "VENTA/CIERRE (BAJO RIESGO)": -5, "MANTENER POSICIÓN": 10}
        
        # Uso de .get con un valor por defecto bajo (-10) para manejar cualquier nueva recomendación
        datos_ordenados = sorted(datos_analizados, key=lambda x: orden_recomendacion.get(x["RECOMENDACION"].split('(')[0].strip(), -10), reverse=True)


        # Generar el HTML
        html_body = generar_html(datos_ordenados)
        
        asunto = f"🔔 Alertas y Oportunidades IBEXIA: {len(datos_ordenados)} oportunidades detectadas hoy {datetime.today().strftime('%d/%m/%Y')}"
        nombre_archivo_base = f"reporte_ibexia_{datetime.today().strftime('%Y%m%d')}"

        enviar_email_con_adjunto(html_body, asunto, nombre_archivo_base)

    except Exception as e:
        print(f"❌ Error al ejecutar el script principal: {e}")

if __name__ == '__main__':
    generar_reporte()
