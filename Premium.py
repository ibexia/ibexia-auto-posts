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

# ******************************************************************************
# ************** INICIO DE LA LÓGICA DE SIMULACIÓN (DE leer_google_sheets.py) **
# ******************************************************************************
def calcular_ganancias_simuladas(precios, smis, fechas, capital_inicial=10000):
    compras = []
    ventas = []
    posicion_abierta = False
    
    # Calcular la pendiente del SMI para cada punto
    pendientes_smi = [0] * len(smis)
    for i in range(1, len(smis)):
        pendientes_smi[i] = smis[i] - smis[i-1]

    # Iterar sobre los datos históricos para encontrar señales
    for i in range(2, len(smis)):
        # Señal de compra: la pendiente del SMI cambia de negativa a positiva y no está en sobrecompra
        if i >= 1 and pendientes_smi[i] > 0 and pendientes_smi[i-1] <= 0:
            if not posicion_abierta:
                # Condición de sobrecompra (SMI < 40)
                if smis[i-1] < 40: 
                    posicion_abierta = True
                    # El precio de la compra se registra al cierre del día anterior (i-1)
                    precio_compra_actual = precios[i-1] 
                    compras.append({'fecha': fechas[i-1], 'precio': precio_compra_actual})
                
        # Señal de venta: la pendiente del SMI cambia de positiva a negativa
        elif i >= 1 and pendientes_smi[i] < 0 and pendientes_smi[i-1] >= 0:
            if posicion_abierta:
                posicion_abierta = False
                # La venta se registra al cierre del día anterior (i-1)
                precio_venta_actual = precios[i-1]
                ventas.append({'fecha': fechas[i-1], 'precio': precio_venta_actual}) 
                
    return compras, ventas
# ******************************************************************************
# *************** FIN DE LA LÓGICA DE SIMULACIÓN *******************************
# ******************************************************************************

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
        return {'s1': 'N/A', 's2': 'N/A', 'r1': 'N/A', 'r2': 'r2'}
        
def calcular_beneficio_perdida(precio_compra, precio_actual, inversion=10000):
    # Función existente, se mantiene para la columna principal si es necesaria
    try:
        precio_compra = float(precio_compra)
        precio_actual = float(precio_actual)
        
        if precio_compra <= 0 or precio_actual <= 0:
            return "N/A"

        acciones = inversion / precio_compra
        beneficio_perdida = (precio_actual - precio_compra) * acciones
        return f"{beneficio_perdida:,.2f}"
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

        # --- Datos Diarios: Periodo extendido para una mejor simulación ---
        hist_extended = stock.history(period="1y", interval="1d") # Cambiado de 150d a 1y
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
            
        hist_extended = calculate_smi_tv(hist_extended) # Calcula SMI

        # ******************************************************************************
        # ************** CÁLCULO DE OPERACIONES SIMULADAS Y ESTADO ACTUAL **************
        # ******************************************************************************
        # Limpiar NaNs para la simulación
        combined_df = hist_extended[['Close', 'SMI']].dropna()
        precios_sim = combined_df['Close'].tolist()
        smis_sim = combined_df['SMI'].tolist()
        fechas_sim = combined_df.index.strftime('%d/%m/%Y').tolist()
        
        compras_sim, ventas_sim = calcular_ganancias_simuladas(precios_sim, smis_sim, fechas_sim)
        
        comprado_status = "NO"
        precio_compra = "N/A"
        fecha_compra = "N/A"
        
        # Determinar el estado actual (SI/NO COMPRADO)
        if len(compras_sim) > len(ventas_sim):
            comprado_status = "SI"
            # La última compra es la posición abierta
            ultima_compra = compras_sim[-1]
            precio_compra = ultima_compra['precio']
            fecha_compra = ultima_compra['fecha']
        # ******************************************************************************
        # ************ FIN CÁLCULO DE OPERACIONES SIMULADAS Y ESTADO ACTUAL ************
        # ******************************************************************************
        
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
            "COMPRADO": comprado_status, # Valor actualizado
            "PRECIO_COMPRA": precio_compra, # Valor actualizado
            "FECHA_COMPRA": fecha_compra, # Valor actualizado
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
            # --- Nuevos Campos para el Desplegable de Operaciones ---
            "SIM_COMPRAS": compras_sim,
            "SIM_VENTAS": ventas_sim,
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

            elif estado_smi_weekly == "Sobreventa" or estado_smi_weekly == "Intermedio":
                data['OPORTUNIDAD'] = "Posibilidad de Compra Activada"
                data['COMPRA_SI'] = f"COMPRAR A PRECIO DE {formatear_numero(precio_aplanamiento)}€"
                data['VENDE_SI'] = "NO VENDER"
                data['ORDEN_PRIORIDAD'] = prioridad["Posibilidad de Compra Activada"]
                
            else: # N/A
                data['OPORTUNIDAD'] = "Posibilidad de Compra"
                data['COMPRA_SI'] = f"COMPRAR A PRECIO DE {formatear_numero(precio_aplanamiento)}€"
                data['VENDE_SI'] = "NO VENDER"
                data['ORDEN_PRIORIDAD'] = prioridad["Posibilidad de Compra"]
                
        else: # Bajando o Plano
            data['OPORTUNIDAD'] = "Posibilidad de Compra"
            data['COMPRA_SI'] = f"COMPRAR A PRECIO DE {formatear_numero(precio_aplanamiento)}€"
            data['VENDE_SI'] = "NO VENDER"
            data['ORDEN_PRIORIDAD'] = prioridad["Posibilidad de Compra"]
            
    elif estado_smi == "Sobrecompra":
        if tendencia == "Bajando":
            data['OPORTUNIDAD'] = "Riesgo de Venta Activada"
            # Nuevo criterio de Venta: si el precio de aplanamiento es menor al de ayer
            if isinstance(precio_aplanamiento, (int, float)) and precio_aplanamiento < current_price:
                data['VENDE_SI'] = f"VENDER A PRECIO DE {formatear_numero(precio_aplanamiento)}€"
            else:
                data['VENDE_SI'] = f"VENDER INMEDIATAMENTE"
            data['COMPRA_SI'] = "NO COMPRAR"
            data['ORDEN_PRIORIDAD'] = prioridad["Riesgo de Venta Activada"]
        else: # Subiendo o Plano
            data['OPORTUNIDAD'] = "Riesgo de Venta"
            if isinstance(precio_aplanamiento, (int, float)) and precio_aplanamiento < current_price:
                data['VENDE_SI'] = f"VENDER A PRECIO DE {formatear_numero(precio_aplanamiento)}€"
            else:
                data['VENDE_SI'] = "VIGILAR"
            data['COMPRA_SI'] = "NO COMPRAR"
            data['ORDEN_PRIORIDAD'] = prioridad["Riesgo de Venta"]
            
    else: # Intermedio
        if tendencia == "Bajando":
            data['OPORTUNIDAD'] = "Seguirá bajando"
            data['COMPRA_SI'] = "NO COMPRAR"
            data['VENDE_SI'] = "VIGILAR"
            data['ORDEN_PRIORIDAD'] = prioridad["Seguirá bajando"]
        elif tendencia == "Subiendo":
            data['OPORTUNIDAD'] = "VIGILAR"
            data['COMPRA_SI'] = "VIGILAR"
            data['VENDE_SI'] = "VIGILAR"
            data['ORDEN_PRIORIDAD'] = prioridad["VIGILAR"]
        else: # Plano
            data['OPORTUNIDAD'] = "Intermedio"
            data['COMPRA_SI'] = "VIGILAR"
            data['VENDE_SI'] = "VIGILAR"
            data['ORDEN_PRIORIDAD'] = prioridad["Intermedio"]

    return data


def enviar_email_con_adjunto(cuerpo_html, asunto, nombre_archivo_base):
    # --- CONFIGURACIÓN PERSONAL (NO MODIFICADA) ---
    sender_email = os.getenv('SENDER_EMAIL')
    receiver_email = os.getenv('RECEIVER_EMAIL')
    smtp_server = os.getenv('SMTP_SERVER')
    smtp_port = int(os.getenv('SMTP_PORT', 587))
    smtp_user = os.getenv('SMTP_USER')
    smtp_password = os.getenv('SMTP_PASSWORD')

    if not all([sender_email, receiver_email, smtp_server, smtp_user, smtp_password]):
        print("❌ Error: Faltan variables de entorno para el envío de correo.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = asunto
    msg["From"] = sender_email
    msg["To"] = receiver_email
    
    # Parte HTML
    msg.attach(MIMEText(cuerpo_html, "html"))
    
    # Parte del archivo adjunto (con extensión .html)
    archivo_adjunto_nombre = f"{nombre_archivo_base}.html"
    
    part = MIMEBase("application", "octet-stream")
    part.set_payload(cuerpo_html.encode('utf-8'))
    
    encoders.encode_base64(part)
    
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {archivo_adjunto_nombre}",
    )
    
    msg.attach(part)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print(f"✅ Correo enviado con éxito a {receiver_email}. Archivo adjunto: {archivo_adjunto_nombre}")
    except Exception as e:
        print(f"❌ Error al enviar el correo: {e}")

# ******************************************************************************
# *************** FUNCION PRINCIPAL DE GENERACIÓN DE REPORTE *******************
# ******************************************************************************
def generar_reporte():
    try:
        # Obtener datos de Google Sheets (la lista de tickers)
        # Se asume que leer_google_sheets() devuelve la lista de tickers
        # Si se usa un diccionario predefinido, se usa ese.
        # Aquí se usa el diccionario 'tickers' como base.
        tickers_list = list(tickers.values())
        
        datos_completos = []
        for nombre, ticker in tickers.items():
            datos = obtener_datos_yfinance(ticker)
            if datos:
                datos = clasificar_empresa(datos)
                datos_completos.append(datos)
        
        if not datos_completos:
            print("No se pudo obtener datos para ningún ticker. Abortando.")
            return

        # Ordenar los datos por prioridad
        # Se excluyen los de "Compra RIESGO" de la ordenación principal
        datos_ordenados = sorted(
            [d for d in datos_completos if d.get('ORDEN_PRIORIDAD', 999) != 8],
            key=lambda x: x.get('ORDEN_PRIORIDAD', 999)
        )
        
        # Se añaden al final los de "Compra RIESGO" sin ordenación
        datos_ordenados.extend([d for d in datos_completos if d.get('ORDEN_PRIORIDAD', 999) == 8])

        # ******************************************************************************
        # *************** INICIO DE LA MODIFICACIÓN DE LA SECCIÓN HTML *****************
        # ******************************************************************************
        html_body = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <title>Reporte de Oportunidades IBEXIA - {datetime.today().strftime('%d/%m/%Y')}</title>
            <style>
                body {{ font-family: 'Arial', sans-serif; background-color: #f4f7f9; color: #333; margin: 0; padding: 20px; }}
                .container {{ max-width: 1400px; margin: 0 auto; background-color: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); }}
                h1 {{ color: #007bff; border-bottom: 3px solid #007bff; padding-bottom: 10px; margin-bottom: 20px; text-align: center; }}
                h2 {{ color: #555; margin-top: 25px; border-left: 5px solid #ffc107; padding-left: 10px; }}
                .search-box {{ margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }}
                .search-box input {{ padding: 10px; border: 1px solid #ccc; border-radius: 5px; font-size: 16px; flex-grow: 1; }}
                .table-scroll-top {{ overflow-x: auto; height: 10px; margin-bottom: -10px; }}
                .table-container {{ overflow-x: auto; max-height: 70vh; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; table-layout: fixed; }}
                thead th {{ background-color: #007bff; color: white; padding: 12px 8px; text-align: left; cursor: pointer; position: sticky; top: 0; z-index: 10; }}
                tbody tr:nth-child(even) {{ background-color: #f9f9f9; }}
                tbody tr:hover {{ background-color: #e9ecef; }}
                td {{ padding: 10px 8px; border-bottom: 1px solid #ddd; word-wrap: break-word; }}
                .clickable {{ cursor: pointer; user-select: none; }}
                .details-row td {{ background-color: #fcfcfc; border-top: 2px solid #ddd; padding: 15px; }}
                .label-compra {{ background-color: #28a745; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }}
                .label-venta {{ background-color: #dc3545; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }}
                .label-vigilar {{ background-color: #ffc107; color: #333; padding: 4px 8px; border-radius: 4px; font-weight: bold; }}
                .warning {{ color: #dc3545; font-weight: bold; }}
                .info {{ color: #007bff; font-weight: bold; }}
                .success {{ color: #28a745; font-weight: bold; }}
                .sort-arrow {{ margin-left: 5px; }}
                .sticky-col {{ position: sticky; left: 0; background-color: #fff; z-index: 20; box-shadow: 2px 0 5px rgba(0,0,0,0.05); }}
                .details-row .sticky-col {{ background-color: #fcfcfc; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Reporte de Oportunidades Algorítmicas IBEXIA</h1>
                <p>Generado el: <strong>{datetime.today().strftime('%d/%m/%Y %H:%M:%S')}</strong></p>
                <div class="search-box">
                    <input type="text" id="searchInput" placeholder="Buscar por Nombre, Ticker u Oportunidad...">
                    <button onclick="clearSearch()">Limpiar Filtros</button>
                </div>
                
                <div class="table-scroll-top" id="scroll-top">
                    <div style="width: 150%; height: 1px;"></div>
                </div>

                <div class="table-container" id="table-container">
                    <table id="data-table">
                        <thead>
                            <tr>
                                <th class="sticky-col" onclick="sortTable(0, 'string')">Ticker <span class="sort-arrow"></span></th>
                                <th onclick="sortTable(1, 'string')">Oportunidad <span class="sort-arrow"></span></th>
                                <th onclick="sortTable(2, 'number')">Precio Act. <span class="sort-arrow"></span></th>
                                <th onclick="sortTable(3, 'number')">SMI Hoy <span class="sort-arrow"></span></th>
                                <th onclick="sortTable(4, 'string')">Estado SMI <span class="sort-arrow"></span></th>
                                <th onclick="sortTable(5, 'string')">EMA (100) <span class="sort-arrow"></span></th>
                                <th onclick="sortTable(6, 'number')">Precio Aplan. <span class="sort-arrow"></span></th>
                                <th onclick="sortTable(7, 'number')">B/P (Sim.) <span class="sort-arrow"></span></th>
                                <th onclick="sortTable(8, 'string')">Comprar si... <span class="sort-arrow"></span></th>
                                <th style="width: 50px;">Detalles</th>
                            </tr>
                        </thead>
                        <tbody id="table-body">
                            </tbody>
                    </table>
                </div>
            </div>

            <script>
                const data = JSON.parse('{json.dumps(datos_ordenados).replace("'", r"\'").replace('\\', '\\\\')}');
                let currentSortCol = -1;
                let isAscending = true;
                const CAPITAL_INICIAL = 10000;

                // Función para dar formato de número con separador de miles y decimales
                function formatNumber(num) {{
                    if (num === null || num === undefined || isNaN(num) || num === 'N/A') return 'N/A';
                    let floatNum = parseFloat(num);
                    return floatNum.toLocaleString('es-ES', {{ minimumFractionDigits: 3, maximumFractionDigits: 3 }});
                }}

                function getProfitLoss(precioCompra, precioActual) {{
                    if (precioCompra === 'N/A' || precioActual === 'N/A') return 'N/A';
                    let pc = parseFloat(precioCompra);
                    let pa = parseFloat(precioActual);
                    if (pc <= 0 || pa <= 0) return 'N/A';

                    let acciones = CAPITAL_INICIAL / pc;
                    let beneficio = (pa - pc) * acciones;
                    return beneficio;
                }}
                
                function getStatusLabel(oportunidad) {{
                    if (oportunidad.includes("Compra Activada")) return '<span class="label-compra">¡COMPRA ACTIVA!</span>';
                    if (oportunidad.includes("Compra RIESGO")) return '<span class="label-compra">COMPRA RIESGO</span>';
                    if (oportunidad.includes("Posibilidad de Compra")) return '<span class="label-vigilar">POSIBLE COMPRA</span>';
                    if (oportunidad.includes("Riesgo de Venta Activada")) return '<span class="label-venta">¡VENTA ACTIVA!</span>';
                    if (oportunidad.includes("Riesgo de Venta")) return '<span class="label-vigilar">RIESGO VENTA</span>';
                    if (oportunidad.includes("Seguirá bajando")) return '<span class="label-venta">BAJISTA</span>';
                    return '<span class="label-vigilar">VIGILAR</span>';
                }}

                function getEmaStatus(tipoEma, valorEma) {{
                    if (tipoEma === 'Soporte') return '<span class="success">Soporte</span> (' + formatNumber(valorEma) + '€)';
                    if (tipoEma === 'Resistencia') return '<span class="warning">Resistencia</span> (' + formatNumber(valorEma) + '€)';
                    return formatNumber(valorEma);
                }}

                function generateTable(dataArray) {{
                    const tbody = document.getElementById('table-body');
                    tbody.innerHTML = ''; // Limpiar la tabla
                    
                    dataArray.forEach(data => {{
                        const profitLossSim = getProfitLoss(data.PRECIO_COMPRA, data.PRECIO_ACTUAL);
                        const profitLossSimFormatted = formatNumber(profitLossSim);
                        const profitLossClass = profitLossSim !== 'N/A' ? (profitLossSim >= 0 ? 'success' : 'warning') : 'info';

                        let rowHTML = `
                            <tr>
                                <td class="sticky-col"><strong class="info">${data.TICKER}</strong><br>${data.NOMBRE_EMPRESA}</td>
                                <td>${getStatusLabel(data.OPORTUNIDAD)}</td>
                                <td data-value="${data.PRECIO_ACTUAL}">${formatNumber(data.PRECIO_ACTUAL)}€</td>
                                <td data-value="${data.SMI_HOY}">${formatNumber(data.SMI_HOY)}</td>
                                <td>${data.ESTADO_SMI} (${data.TENDENCIA_ACTUAL})</td>
                                <td>${getEmaStatus(data.TIPO_EMA, data.VALOR_EMA)}</td>
                                <td data-value="${data.PRECIO_APLANAMIENTO}">${formatNumber(data.PRECIO_APLANAMIENTO)}€</td>
                                <td data-value="${profitLossSim}">${data.COMPRADO === 'SI' ? '<span class="' + profitLossClass + '">' + profitLossSimFormatted + '€</span>' : 'N/A'}</td>
                                <td><span class="${data.COMPRA_SI === 'NO COMPRAR' || data.VENDE_SI === 'VIGILAR' ? 'warning' : 'success'}">${data.COMPRA_SI === 'VIGILAR' ? data.VENDE_SI : data.COMPRA_SI}</span></td>
                                <td style="text-align: center;">
                                    <button onclick="toggleDetails('${data.TICKER}')">Ver más</button>
                                </td>
                            </tr>
                        `;

                        // ******************************************************************************
                        // *************** GENERACIÓN DE LA FILA DE DETALLES (VER MÁS) ********************
                        // ******************************************************************************

                        // --- Generar Historial de Operaciones ---
                        let operacionesDetalle = '';
                        let numOperacionesCompletadas = Math.min(data.SIM_COMPRAS.length, data.SIM_VENTAS.length);
                        let compras = data.SIM_COMPRAS;
                        let ventas = data.SIM_VENTAS;

                        // 1. Posición Abierta (Si la hay)
                        if (data.COMPRADO === "SI") {
                            let ultimaCompra = compras[compras.length - 1];
                            let precioActual = parseFloat(data.PRECIO_ACTUAL);
                            let precioCompra = parseFloat(ultimaCompra.precio);
                            let gananciaActual = getProfitLoss(precioCompra, precioActual);
                            let estadoGanancia = gananciaActual !== 'N/A' ? (gananciaActual >= 0 ? "Ganancia" : "Pérdida") : 'N/A';
                            let estiloGanancia = gananciaActual !== 'N/A' ? (gananciaActual >= 0 ? '#28a745' : '#dc3545') : '#007bff';

                            operacionesDetalle += `
                                <li style="color: #007bff; font-weight: bold; margin-bottom: 5px;">
                                    POSICIÓN ABIERTA: Entrada en ${ultimaCompra.fecha} a 
                                    <strong>${formatNumber(ultimaCompra.precio)}€</strong>. 
                                    Ganancia/Pérdida actual: 
                                    <strong style="color: ${estiloGanancia};">${formatNumber(gananciaActual)}€</strong> (${estadoGanancia}).
                                </li>`;
                        }

                        // 2. Operaciones Cerradas (Mostrar las más recientes primero)
                        for (let i = numOperacionesCompletadas - 1; i >= 0; i--) { 
                            let compra = compras[i];
                            let venta = ventas[i];
                            let ganancia = getProfitLoss(compra.precio, venta.precio);
                            let estadoGanancia = ganancia !== 'N/A' ? (ganancia >= 0 ? "Ganancia" : "Pérdida") : 'N/A';
                            let estiloGanancia = ganancia !== 'N/A' ? (ganancia >= 0 ? '#28a745' : '#dc3545') : '#555';

                            operacionesDetalle += `
                                <li style="margin-left: 15px; border-left: 2px solid #ccc; padding-left: 10px; margin-top: 5px;">
                                    Operación CERRADA: <strong>Entrada</strong> en ${compra.fecha} a 
                                    <strong>${formatNumber(compra.precio)}€</strong>. 
                                    <strong>Salida</strong> en ${venta.fecha} a 
                                    <strong>${formatNumber(venta.precio)}€</strong>. 
                                    Beneficio: <strong style="color: ${estiloGanancia};">${formatNumber(ganancia)}€</strong> (${estadoGanancia}).
                                </li>`;
                        }

                        let historialOpHtml = '<ul style="list-style-type: none; padding-left: 0;">';
                        if (operacionesDetalle) {
                            historialOpHtml += operacionesDetalle;
                        } else {
                            historialOpHtml += '<li>No se encontraron operaciones de Compra/Venta en el período de simulación (1 año) según el Algoritmo SMI.</li>';
                        }
                        historialOpHtml += '</ul>';

                        // Construir el contenido del desplegable
                        let detailsContent = `
                            <div style="display: flex; gap: 30px; justify-content: space-between;">
                                <div style="flex: 1; min-width: 250px;">
                                    <p style="margin-top: 0;"><strong>ESTADO DE COMPRA ALGORÍTMICO:</strong> 
                                        <span style="font-weight: bold; font-size: 1.1em; color: ${data.COMPRADO === 'SI' ? '#28a745' : '#dc3545'};">${data.COMPRADO}</span>
                                    </p>
                                    ${data.COMPRADO === 'SI' ? `<p><strong>Última Entrada (Algoritmo):</strong> ${data.FECHA_COMPRA} a ${formatNumber(data.PRECIO_COMPRA)}€</p>` : '<p><strong>Última Entrada:</strong> N/A (Posición cerrada)</p>'}
                                    
                                    <p><strong>Soportes / Resistencias (Local):</strong></p>
                                    <ul style="list-style-type: none; padding-left: 0;">
                                        <li><strong>S1:</strong> ${data.SOPORTE_1 ? formatNumber(data.SOPORTE_1) + '€' : 'N/A'}</li>
                                        <li><strong>S2:</strong> ${data.SOPORTE_2 ? formatNumber(data.SOPORTE_2) + '€' : 'N/A'}</li>
                                        <li><strong>R1:</strong> ${data.RESISTENCIA_1 ? formatNumber(data.RESISTENCIA_1) + '€' : 'N/A'}</li>
                                        <li><strong>R2:</strong> ${data.RESISTENCIA_2 ? formatNumber(data.RESISTENCIA_2) + '€' : 'N/A'}</li>
                                    </ul>
                                </div>
                                <div style="flex: 2; min-width: 600px; border-left: 1px solid #eee; padding-left: 20px;">
                                    <p style="margin-top: 0; font-weight: bold; border-bottom: 1px solid #eee;">Historial de Compras y Ventas (Simulación SMI - Capital por operación: ${formatNumber(CAPITAL_INICIAL)}€)</p>
                                    ${historialOpHtml}
                                </div>
                            </div>
                            ${data.ADVERTENCIA_SEMANAL === 'SI' ? '<div style="margin-top: 15px; padding: 10px; background-color: #fff3cd; border-radius: 5px; color: #856404; border: 1px solid #ffeeba;"><strong>ADVERTENCIA SEMANAL:</strong> ' + data.OBSERVACION_SEMANAL + '</div>' : (data.OBSERVACION_SEMANAL ? '<div style="margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;"><strong>OBSERVACIÓN SEMANAL:</strong> ' + data.OBSERVACION_SEMANAL + '</div>' : '')}
                        `;
                        
                        rowHTML += '<tr class="details-row" id="details-' + data.TICKER + '" style="display: none;"><td colspan="10" style="padding: 15px; border-top: 2px solid #ddd; background-color: #fcfcfc;">' + detailsContent + '</td></tr>';
                        
                        tbody.insertAdjacentHTML('beforeend', rowHTML);
                    }});
                }
                
                function toggleDetails(ticker) {{
                    const detailsRow = document.getElementById('details-' + ticker);
                    if (detailsRow) {{
                        detailsRow.style.display = detailsRow.style.display === 'none' ? 'table-row' : 'none';
                    }}
                }}

                // --- Funciones de Búsqueda y Ordenación (Se mantienen) ---

                function filterTable() {{
                    const input = document.getElementById('searchInput');
                    const filter = input.value.toUpperCase();
                    const tbody = document.getElementById('table-body');
                    const tr = tbody.getElementsByTagName('tr');
                    
                    for (let i = 0; i < tr.length; i++) {{
                        let row = tr[i];
                        if (row.classList.contains('details-row')) continue; // Saltar filas de detalle

                        let display = 'none';
                        // Buscar en Ticker (col 0), Oportunidad (col 1), y Nombre de Empresa
                        const tickerCell = row.cells[0];
                        const oportunidadCell = row.cells[1];
                        
                        // Obtener texto de la celda de oportunidad (sin las etiquetas HTML)
                        const oportunidadText = oportunidadCell.textContent || oportunidadCell.innerText;

                        if (tickerCell.innerHTML.toUpperCase().indexOf(filter) > -1 || oportunidadText.toUpperCase().indexOf(filter) > -1) {{
                            display = '';
                        }}
                        
                        // Aplicar la visibilidad a la fila principal y a su fila de detalles
                        row.style.display = display;
                        const detailsRow = document.getElementById('details-' + tickerCell.querySelector('strong').textContent);
                        if (detailsRow) {{
                            detailsRow.style.display = 'none'; // Siempre ocultar el detalle al filtrar
                        }}
                    }}
                }}

                function clearSearch() {{
                    document.getElementById('searchInput').value = '';
                    filterTable();
                }}

                function sortTable(n, type) {{
                    const table = document.getElementById("data-table");
                    const tbody = table.querySelector("tbody");
                    const rows = Array.from(tbody.querySelectorAll("tr:not(.details-row)"));
                    
                    if (currentSortCol === n) {{
                        isAscending = !isAscending;
                    }} else {{
                        isAscending = true;
                        currentSortCol = n;
                    }}

                    // Remover flechas de ordenación antiguas
                    table.querySelectorAll('.sort-arrow').forEach(span => span.textContent = '');

                    // Establecer nueva flecha de ordenación
                    const header = table.querySelectorAll('th')[n];
                    const arrowSpan = header.querySelector('.sort-arrow');
                    arrowSpan.textContent = isAscending ? ' ▲' : ' ▼';


                    rows.sort((rowA, rowB) => {{
                        let cellA = rowA.cells[n];
                        let cellB = rowB.cells[n];
                        let valA, valB;

                        if (type === 'number') {{
                            valA = parseFloat(cellA.getAttribute('data-value') || 0);
                            valB = parseFloat(cellB.getAttribute('data-value') || 0);
                        }} else if (type === 'string') {{
                            valA = (cellA.textContent || cellA.innerText).toUpperCase();
                            valB = (cellB.textContent || cellB.innerText).toUpperCase();
                        }}
                        
                        let comparison = 0;
                        if (valA > valB) {{
                            comparison = 1;
                        }} else if (valA < valB) {{
                            comparison = -1;
                        }}
                        
                        return isAscending ? comparison : comparison * -1;
                    }});

                    // Reconstruir el tbody con las filas ordenadas y sus filas de detalle
                    tbody.innerHTML = '';
                    rows.forEach(row => {{
                        const ticker = row.cells[0].querySelector('strong').textContent;
                        const detailsRow = document.getElementById('details-' + ticker);
                        tbody.appendChild(row);
                        if (detailsRow) {{
                            tbody.appendChild(detailsRow);
                        }}
                    }});
                }}

                // --- Inicialización ---
                document.addEventListener('DOMContentLoaded', () => {{
                    generateTable(data);
                    
                    const searchInput = document.getElementById('searchInput');
                    searchInput.addEventListener('keyup', filterTable);
                    
                    // Configurar el enfoque de la búsqueda
                    if (searchInput) {{
                        searchInput.focus(); // Enfocar el campo de búsqueda al cargar
                    }}
                    
                    // Sincronizar el scroll lateral
                    const tableContainer = document.getElementById('table-container');
                    const scrollTop = document.getElementById('scroll-top');
                    
                    if (tableContainer && scrollTop) {{
                        scrollTop.addEventListener('scroll', () => {{
                            tableContainer.scrollLeft = scrollTop.scrollLeft;
                        }});
                        
                        tableContainer.addEventListener('scroll', () => {{
                            scrollTop.scrollLeft = tableContainer.scrollLeft;
                        }});
                    }}
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
        print(f"❌ Error al ejecutar el script principal: {e}")

if __name__ == '__main__':
    generar_reporte()
