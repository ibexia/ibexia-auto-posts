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
    'Bankinter': 'BKT.MC',
    'BBVA': 'BBVA.MC',
    'Berkeley': 'BKY.MC',
    'Biotechnology': 'BST.MC',
    'CaixaBank': 'CABK.MC',
    'Cellnex': 'CLNX.MC',
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
        
        comprado_status = "NO"
        precio_compra = "N/A"
        fecha_compra = "N/A"
        
        smi_series_copy = hist_extended['SMI'].copy()
        pendientes_smi = smi_series_copy.diff()
        
        for i in range(len(hist_extended) - 1, 0, -1):
            smi_prev = hist_extended['SMI'].iloc[i - 1]
            pendiente_prev = pendientes_smi.iloc[i - 1]
            pendiente_curr = pendientes_smi.iloc[i]
            
            if pendiente_curr < 0 and pendiente_prev >= 0:
                comprado_status = "NO"
                precio_compra = hist_extended['Close'].iloc[i]
                fecha_compra = hist_extended.index[i].strftime('%d/%m/%Y')
                break
            
            elif pendiente_curr > 0 and pendiente_prev <= 0 and smi_prev < 40:
                comprado_status = "SI"
                precio_compra = hist_extended['Close'].iloc[i]
                fecha_compra = hist_extended.index[i].strftime('%d/%m/%Y')
                break

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
            "RESISTENCIA_2": sr_levels['r2']
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

    prioridad = {
        "Posibilidad de Compra Activada": 1,
        "Posibilidad de Compra": 2,
        "VIGILAR": 3,
        "Riesgo de Venta": 4,
        "Riesgo de Venta Activada": 5,
        "Seguirá bajando": 6,
        "Intermedio": 7
    }

    if estado_smi == "Sobreventa":
        if tendencia == "Subiendo":
            data['OPORTUNIDAD'] = "Compra Activada"
            data['COMPRA_SI'] = "COMPRA YA"
            data['VENDE_SI'] = "NO VENDER"
            data['ORDEN_PRIORIDAD'] = prioridad["Posibilidad de Compra Activada"]
        elif tendencia == "Bajando":
            data['OPORTUNIDAD'] = "Posibilidad de Compra"
            if current_price > close_yesterday:
                data['COMPRA_SI'] = "COMPRA YA"
            else:
                data['COMPRA_SI'] = f"COMPRAR SI SUPERA {formatear_numero(close_yesterday)}€"
            data['VENDE_SI'] = "NO VENDER"
            data['ORDEN_PRIORIDAD'] = prioridad["Posibilidad de Compra"]
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
    
    texto_observacion = f"<strong>Observaciones de {nombre_empresa}:</strong><br>"

    if oportunidad == "Posibilidad de Compra Activada":
        texto = f"El algoritmo se encuentra en una zona de sobreventa y muestra una tendencia alcista en sus últimos valores, lo que activa una señal de compra fuerte. Se recomienda tener en cuenta los niveles de resistencia ({resistencia1}€) para determinar un objetivo de precio. La EMA de 100 periodos se encuentra en {valor_ema}€, actuando como un nivel de {tipo_ema}."
    
    elif oportunidad == "Posibilidad de Compra":
        if "COMPRA YA" in compra_si:
            texto = f"El algoritmo detecta que el valor está en una zona de sobreventa, lo que puede ser un indicador de reversión. El algoritmo ha detectado una oportunidad de compra inmediata para aprovechar un posible rebote.La EMA de 100 periodos se encuentra en {valor_ema}€, actuando como un nivel de {tipo_ema}."
        else:
            texto = f"El algoritmo detecta que el valor está en una zona de sobreventa con una tendencia bajista. Se ha detectado una oportunidad de {compra_si} para un posible rebote. La EMA de 100 periodos se encuentra en {valor_ema}€, actuando como un nivel de {tipo_ema}."
    
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
    
    return f'<p style="text-align:left; color:#000;">{texto_observacion.strip()}{texto.strip()}</p>'


def enviar_email_con_adjunto(html_body, asunto_email):
    remitente = "xumkox@gmail.com"
    destinatario = "xumkox@gmail.com"
    password = "kdgz lvdo wqvt vfkt"
    msg = MIMEMultipart()
    msg['From'] = remitente
    msg['To'] = destinatario
    msg['Subject'] = asunto_email

    html_filename = "analisis-empresas.html"
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(html_body)

    with open(html_filename, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f"attachment; filename={html_filename}",
    )
    msg.attach(part)

    try:
        servidor = smtplib.SMTP('smtp.gmail.com', 587)
        servidor.starttls()
        servidor.login(remitente, password)
        servidor.sendmail(remitente, destinatario, msg.as_string())
        servidor.quit()
        print(f"✅ Correo enviado con el asunto: {asunto_email}")
        os.remove(html_filename)
    except Exception as e:
        print("❌ Error al enviar el correo:", e)

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

        # --- Lógica de ordenación integrada ---
        def obtener_clave_ordenacion(empresa):
            categoria = empresa['OPORTUNIDAD']
            
            orden_grupo = 99
            orden_interna = float('inf')
            
            if categoria in ["Posibilidad de Compra Activada", "Posibilidad de Compra"]:
                orden_grupo = 1
            elif categoria in ["VIGILAR", "Riesgo de Venta", "Riesgo de Venta Activada", "Seguirá bajando"]:
                orden_grupo = 2
            elif categoria == "Intermedio":
                orden_grupo = 3

            return (empresa['ORDEN_PRIORIDAD'], empresa['NOMBRE_EMPRESA'])

        datos_ordenados = sorted(datos_completos, key=obtener_clave_ordenacion)
        
        # --- Fin de la lógica de ordenación integrada ---
        now_utc = datetime.utcnow()
        hora_actual = (now_utc + timedelta(hours=2)).strftime('%H:%M')
        
        html_body = f"""
        <html>
        <head>
            <title>Resumen Diario de Oportunidades - {datetime.today().strftime('%d/%m/%Y')} {hora_actual}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-color: #f8f9fa;
                    margin: 0;
                    padding: 10px;
                }}
                .main-container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: #ffffff;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
                }}
                h2 {{
                    color: #343a40;
                    text-align: center;
                    font-size: 1.5em;
                    margin-bottom: 10px;
                }}
                p {{
                    color: #6c757d;
                    text-align: center;
                    font-size: 0.9em;
                }}
                #search-container {{
                    margin-bottom: 15px;
                }}
                #searchInput {{
                    width: 100%;
                    padding: 8px;
                    font-size: 0.9em;
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    box-sizing: border-box;
                }}
                .table-container {{
                    overflow-x: auto;
                    overflow-y: auto;
                    height: 70vh; /* Altura un poco más baja para que no sea tan grande */
                    position: relative;
                }}
                table {{
                    width: 100%;
                    table-layout: fixed;
                    margin: 10px auto 0 auto;
                    border-collapse: collapse;
                    font-size: 0.85em; /* Reducir el tamaño de la fuente de la tabla */
                }}
                th, td {{
                    border: 1px solid #e9ecef;
                    padding: 6px; /* Reducir el padding para celdas más pequeñas */
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
                    white-space: nowrap; /* Para que los títulos de las columnas no se rompan */
                }}
                .compra {{ color: #28a745; font-weight: bold; }}
                .venta {{ color: #dc3545; font-weight: bold; }}
                .comprado-si {{ background-color: #28a745; color: white; font-weight: bold; }}
                .bg-green {{ background-color: #d4edda; color: #155724; }}
                .bg-red {{ background-color: #f8d7da; color: #721c24; }}
                .bg-highlight {{ background-color: #28a745; color: white; font-weight: bold; }}
                .text-center {{ text-align: center; }}
                .disclaimer {{ font-size: 0.8em; text-align: center; color: #6c757d; margin-top: 15px; }}
                .small-text {{ font-size: 0.7em; color: #6c757d; }}
                .green-cell {{ background-color: #d4edda; }}
                .red-cell {{ background-color: #f8d7da; }}
                .separator-row td {{ background-color: #e9ecef; height: 3px; padding: 0; border: none; }}
                .category-header td {{
                    background-color: #495057;
                    color: white;
                    font-size: 1.1em;
                    font-weight: bold;
                    text-align: center;
                    padding: 10px;
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
                    font-size: 0.8em;
                }}
                .vigilar {{ color: #ffc107; font-weight: bold; }}
                
                /* Estilos para las filas colapsables */
                .collapsible-row {
                    display: none;
                }
                .toggle-input {
                    display: none;
                }
                .toggle-label {
                    cursor: pointer;
                    display: block;
                    width: 100%;
                }
                .toggle-input:checked ~ .collapsible-row {
                    display: table-row;
                }
            </style>                
        </head>
        <body>
            <div class="main-container">
                <h2 class="text-center">Resumen Diario de Oportunidades ordenadas por prioridad - {datetime.today().strftime('%d/%m/%Y')} {hora_actual}</h2>
                
                <div id="search-container">
                    <input type="text" id="searchInput" placeholder="Buscar por nombre de empresa...">
                </div>
                
                <div id="scroll-top" style="overflow-x: auto;">
                    <div style="min-width: 1400px;">&nbsp;</div>
                </div>
                
                <div class="table-container">
                    <table id="myTable">
                        <thead>
                            <tr>
                                <th>Empresa (Precio)</th>
                                <th>Tendencia Actual</th>
                                <th>EMA</th>
                                <th>Oportunidad</th>
                                <th>Compra si...</th>
                                <th>Vende si...</th>
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
                
                current_orden_grupo = data['ORDEN_PRIORIDAD']
                
                if previous_orden_grupo is None:
                     if current_orden_grupo in [1, 2]:
                         html_body += """
                            <tr class="category-header"><td colspan="6">OPORTUNIDADES DE COMPRA</td></tr>
                        """
                     elif current_orden_grupo in [3, 4, 5]:
                         html_body += """
                            <tr class="category-header"><td colspan="6">ATENTOS A VENDER</td></tr>
                        """
                     elif current_orden_grupo in [6, 7]:
                         html_body += """
                            <tr class="category-header"><td colspan="6">OTRAS EMPRESAS SIN MOVIMIENTOS</td></tr>
                        """
                
                elif current_orden_grupo != previous_orden_grupo:
                    if current_orden_grupo in [3, 4, 5] and previous_orden_grupo in [1, 2]:
                        html_body += """
                            <tr class="category-header"><td colspan="6">ATENTOS A VENDER</td></tr>
                        """
                    elif current_orden_grupo in [6, 7] and previous_orden_grupo in [1, 2, 3, 4, 5]:
                         html_body += """
                            <tr class="category-header"><td colspan="6">OTRAS EMPRESAS SIN MOVIMIENTOS</td></tr>
                        """
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

                clase_oportunidad = "compra" if "compra" in data['OPORTUNIDAD'].lower() else ("venta" if "venta" in data['OPORTUNIDAD'].lower() else ("vigilar" if "vigilar" in data['OPORTUNIDAD'].lower() else ""))
                
                celda_empresa_class = ""
                if "compra" in data['OPORTUNIDAD'].lower():
                    celda_empresa_class = "green-cell"
                elif "venta" in data['OPORTUNIDAD'].lower():
                    celda_empresa_class = "red-cell"
                
                observaciones = generar_observaciones(data)
                
                html_body += f"""
                            <tr>
                                <td class="{celda_empresa_class}">
                                    <input type="checkbox" id="toggle_{data['TICKER']}" class="toggle-input">
                                    <label for="toggle_{data['TICKER']}" class="toggle-label">
                                        {nombre_con_precio}
                                    </label>
                                </td>
                                <td>{data['TENDENCIA_ACTUAL']}</td>
                                <td>{formatear_numero(data['VALOR_EMA'])}€<br><b>({data['TIPO_EMA']} EMA)</b></td>
                                <td class="{clase_oportunidad}">{data['OPORTUNIDAD']}</td>
                                <td>{data['COMPRA_SI']}</td>
                                <td>{data['VENDE_SI']}</td>
                            </tr>
                            <tr class="collapsible-row">
                                <td colspan="6">
                                    <div style="text-align: left; padding: 10px;">
                                        <strong>Soportes:</strong> {formatear_numero(data['SOPORTE_1'])}€, {formatear_numero(data['SOPORTE_2'])}€<br>
                                        <strong>Resistencias:</strong> {formatear_numero(data['RESISTENCIA_1'])}€, {formatear_numero(data['RESISTENCIA_2'])}€<br><br>
                                        {observaciones}
                                    </div>
                                </td>
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
                // Función de filtrado
                function filterTable() {
                    var input, filter, table, tr, i, txtValue;
                    input = document.getElementById("searchInput");
                    filter = input.value.toUpperCase();
                    table = document.getElementById("myTable");
                    var tbody = table.querySelector('tbody');
                    tr = tbody.getElementsByTagName("tr");

                    for (i = 0; i < tr.length; i++) {
                        // Skip separator and category rows
                        if (tr[i].classList.contains("separator-row") || tr[i].classList.contains("category-header") || tr[i].classList.contains("collapsible-row")) {
                            continue;
                        }

                        // Check the company name row
                        var label = tr[i].querySelector("label");
                        if (label) {
                            txtValue = label.textContent || label.innerText;
                            var mainRow = tr[i];
                            var collapsibleRow = tr[i + 1];

                            if (txtValue.toUpperCase().indexOf(filter) > -1) {
                                mainRow.style.display = "";
                                // Check if the next row is a collapsible row and hide it by default on filter
                                if (collapsibleRow && collapsibleRow.classList.contains("collapsible-row")) {
                                    collapsibleRow.style.display = "none";
                                    var checkbox = mainRow.querySelector('.toggle-input');
                                    if(checkbox) checkbox.checked = false; // Uncheck it
                                }
                            } else {
                                mainRow.style.display = "none";
                                if (collapsibleRow && collapsibleRow.classList.contains("collapsible-row")) {
                                    collapsibleRow.style.display = "none";
                                }
                            }
                        }
                    }
                }
                
                // Asegurar que el script se ejecute cuando el DOM esté listo
                document.addEventListener('DOMContentLoaded', function() {
                    const searchInput = document.getElementById("searchInput");
                    if (searchInput) {
                        searchInput.addEventListener("keyup", filterTable);
                    }

                    const tableContainer = document.querySelector('.table-container');
                    const scrollTop = document.getElementById('scroll-top');
                    
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
        
        asunto = f"🔔 Alertas y Oportunidades IBEXIA: {len(datos_ordenados)} oportunidades detectadas hoy {datetime.today().strftime('%d/%m/%Y')}"
        enviar_email_con_adjunto(html_body, asunto)

    except Exception as e:
        print(f"❌ Error al ejecutar el script principal: {e}")

if __name__ == '__main__':
    generar_reporte()
