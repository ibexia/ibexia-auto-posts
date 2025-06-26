import os
import json
import smtplib
import yfinance as yf
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.oauth2 import service_account
from googleapiclient.discovery import build
import google.generativeai as genai
from datetime import datetime
import pandas as pd
import numpy as np
import time
import re

def leer_google_sheets():
    """
    Lee los tickers de Google Sheets desde la columna A.
    Requiere las variables de entorno GOOGLE_APPLICATION_CREDENTIALS y SPREADSHEET_ID.
    """
    credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not credentials_json:
        raise Exception("No se encontró la variable de entorno GOOGLE_APPLICATION_CREDENTIALS")

    creds_dict = json.loads(credentials_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
    )

    spreadsheet_id = os.getenv('SPREADSHEET_ID') # ¡CORREGIDO: antes SPREADSHEED_ID!
    if not spreadsheet_id:
        raise Exception("No se encontró la variable de entorno SPREADSHEET_ID")

    range_name = 'A:A'  # Se fuerza el rango a 'A:A' para leer toda la columna A

    service = build('sheets', 'v4', credentials=creds)
    # CORREGIDO: Usar service.spreadsheets().values() en lugar de service.sheets().values()
    sheet = service.spreadsheets().values()
    result = sheet.get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    values = result.get('values', [])

    if not values:
        print('No se encontraron datos en la hoja de cálculo.')
    else:
        print('Datos leídos de la hoja:')
        for row in values:
            print(row)

    # Retorna solo el primer elemento de cada fila que no esté vacía
    return [row[0] for row in values if row]


# Parámetros para el cálculo del SMI (Stochastic Momentum Index)
length_k = 10
length_d = 3
ema_signal_len = 10
smooth_period = 5

def calculate_smi_tv(df):
    """
    Calcula el Stochastic Momentum Index (SMI) Technical View para el DataFrame de precios.
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    hh = high.rolling(window=length_k).max()
    ll = low.rolling(window=length_k).min()
    diff = hh - ll
    rdiff = close - (hh + ll) / 2

    avgrel = rdiff.ewm(span=length_d, adjust=False).mean()
    avgdiff = diff.ewm(span=length_d, adjust=False).mean()

    # Manejo de división por cero para avgdiff: inicializa con ceros y calcula solo donde avgdiff no es cero
    smi_raw = pd.Series(0.0, index=df.index)
    non_zero_avgdiff_mask = avgdiff != 0
    smi_raw[non_zero_avgdiff_mask] = (avgrel[non_zero_avgdiff_mask] / (avgdiff[non_zero_avgdiff_mask] / 2)) * 100

    smi_smoothed = smi_raw.rolling(window=smooth_period).mean()
    smi_signal = smi_smoothed.ewm(span=ema_signal_len, adjust=False).mean()

    df = df.copy() # Trabaja en una copia para evitar SettingWithCopyWarning
    df['SMI'] = smi_smoothed
    df['SMI_signal'] = smi_signal
    
    return df

def find_significant_supports(df, current_price, window=40, tolerance_percent=0.01, max_deviation_percent=0.15):
    """
    Identifica los 3 soportes más significativos y cercanos al precio actual
    basándose en mínimos locales y agrupaciones de precios.
    Filtra valores no válidos (cero o negativos) para soportes.
    """
    recent_data = df.tail(window)
    lows = recent_data['Low']
    
    potential_supports = []
    
    # Encuentra mínimos locales que sean mayores a cero
    for i in range(1, len(lows) - 1):
        if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
            if lows.iloc[i] > 0:
                potential_supports.append(lows.iloc[i])

    # Si no hay mínimos locales significativos, usa los lows del periodo que sean mayores a cero
    if not potential_supports:
        potential_supports = [l for l in lows.tolist() if l > 0]
        
    support_zones = {}
    for support in potential_supports:
        found_zone = False
        for zone_level in support_zones.keys():
            # Agrupa soportes cercanos dentro de una tolerancia porcentual
            if zone_level != 0 and abs(support - zone_level) / zone_level <= tolerance_percent:
                support_zones[zone_level].append(support)
                found_zone = True
                break
        if not found_zone:
            support_zones[support] = [support]
            
    final_supports = []
    for zone_level, values in support_zones.items():
        avg_support = np.mean(values)
        # Filtra soportes por debajo del precio actual y dentro de una desviación máxima
        if avg_support < current_price:
            if current_price != 0 and abs(current_price - avg_support) / current_price <= max_deviation_percent:
                final_supports.append({'level': avg_support, 'frequency': len(values)})

    # Ordena los soportes por cercanía al precio actual y luego por frecuencia
    final_supports.sort(key=lambda x: (abs(x['level'] - current_price), -x['frequency']))
    
    # Selecciona los 3 soportes más relevantes y redondéalos
    top_3_supports = [round(s['level'], 2) for s in final_supports if s['level'] < current_price][:3]
    
    # Rellena con soportes adicionales si no hay 3 significativos, evitando duplicados y valores cero
    if len(top_3_supports) < 3:
        sorted_lows = sorted([l for l in lows.tolist() if l < current_price and l > 0], reverse=True)
        for low_val in sorted_lows:
            rounded_low_val = round(low_val, 2)
            if rounded_low_val not in top_3_supports:
                top_3_supports.append(rounded_low_val)
                if len(top_3_supports) == 3:
                    break
    
    # Si aún faltan soportes, genera valores mínimos para asegurar 3
    while len(top_3_supports) < 3:
        if len(top_3_supports) > 0:
            next_support = round(top_3_supports[-1] * 0.95, 2)
            top_3_supports.append(max(0.01, next_support)) # Asegura un mínimo de 0.01
        else:
            default_support = round(current_price * 0.90, 2)
            top_3_supports.append(max(0.01, default_support)) # Asegura un mínimo de 0.01
            
    return top_3_supports

def traducir_texto_con_gemini(text, max_retries=3, initial_delay=5):
    """
    Traduce texto al español utilizando la API de Gemini, con reintentos para manejar errores de cuota.
    """
    if not text or text.strip().lower() in ["n/a", "no disponibles", "no disponible"]:
        return text

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Advertencia: GEMINI_API_KEY no configurada. No se realizará la traducción.")
        return text

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")
    
    retries = 0
    delay = initial_delay
    while retries < max_retries:
        try:
            # Petición a Gemini para traducción
            response = model.generate_content(f"Traduce el siguiente texto al español de forma concisa y profesional: \"{text}\"")
            translated_text = response.text.strip().replace("**", "").replace("*", "")
            return translated_text
        except Exception as e:
            # Manejo específico de error de cuota (429) y otros errores
            if "429 You exceeded your current quota" in str(e):
                try:
                    match = re.search(r"retry_delay \{\s*seconds: (\d+)", str(e))
                    if match:
                        server_delay = int(match.group(1))
                        delay = max(delay, server_delay + 1)
                except:
                    pass # En caso de que no se pueda extraer el retraso del servidor
                
                print(f"❌ Cuota de Gemini excedida al traducir. Reintentando en {delay} segundos... (Intento {retries + 1}/{max_retries})")
                time.sleep(delay)
                retries += 1
                delay *= 2
            else:
                print(f"❌ Error al traducir texto con Gemini (no de cuota): {e}")
                return text # Retorna el texto original en caso de otros errores
    print(f"❌ Falló la traducción después de {max_retries} reintentos.")
    return text # Retorna el texto original si fallan todos los reintentos

def obtener_datos_yfinance(ticker):
    """
    Obtiene datos financieros y estadísticos de un ticker usando yfinance.
    Calcula el SMI y determina soportes clave.
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    
    hist = stock.history(period="90d", interval="1d")

    if hist.empty:
        print(f"❌ No se pudieron obtener datos históricos para {ticker}")
        return None

    try:
        # Asegúrate de que las columnas críticas existan antes de calcular SMI
        required_cols = ['High', 'Low', 'Close', 'Volume']
        if not all(col in hist.columns for col in required_cols):
            print(f"❌ Datos históricos incompletos para {ticker}. Faltan columnas: {set(required_cols) - set(hist.columns)}")
            return None

        hist = calculate_smi_tv(hist) # Calcula el SMI
        
        # Verifica la disponibilidad de la señal SMI
        if 'SMI_signal' not in hist.columns or hist['SMI_signal'].empty or len(hist['SMI_signal'].dropna()) < 2:
            print(f"❌ SMI_signal no disponible o insuficiente para {ticker}")
            return None

        smi_actual = round(hist['SMI_signal'].dropna().iloc[-1], 2)
        smi_anterior = round(hist['SMI_signal'].dropna().iloc[-2], 2)
        
        smi_tendencia = "subiendo" if smi_actual > smi_anterior else "bajando" if smi_actual < smi_anterior else "estable"

        current_price = round(info.get("regularMarketPrice", info.get("currentPrice", 0)), 2)

        # Calcula soportes significativos
        if current_price == 0:
            print(f"Advertencia: El precio actual para {ticker} es 0. Los soportes se establecerán en 0 o un valor mínimo.")
            soportes = [0.00, 0.00, 0.00]
        else:
            soportes = find_significant_supports(hist, current_price)
        
        soporte_1 = soportes[0] if len(soportes) > 0 else 0
        soporte_2 = soportes[1] if len(soportes) > 1 else 0
        soporte_3 = soportes[2] if len(soportes) > 2 else 0

        # Asegura que los soportes no sean 0 si el precio actual no lo es, para evitar divisiones por cero en cálculos posteriores
        soporte_1 = max(0.01, soporte_1) if soporte_1 == 0 and current_price != 0 else soporte_1
        soporte_2 = max(0.01, soporte_2) if soporte_2 == 0 and current_price != 0 else soporte_2
        soporte_3 = max(0.01, soporte_3) if soporte_3 == 0 and current_price != 0 else soporte_3

        # Obtener el volumen del día anterior (penúltimo día en el histórico)
        current_volume = 0
        if len(hist['Volume']) >= 2:
            current_volume = hist['Volume'].iloc[-2]

        # Calcula la nota de la empresa basada en el SMI
        nota_empresa = round((-(max(min(smi_actual, 60), -60)) + 60) * 10 / 120, 1)

        recomendacion = "Indefinido"
        condicion_rsi = "desconocido"
        
        # Lógica de recomendación y tendencia del SMI más matizada
        if nota_empresa <= 2: # SMI muy alto (sobrecompra fuerte)
            condicion_rsi = "muy sobrecomprado"
            smi_tendencia = "mostrando un agotamiento alcista."
            if smi_actual > smi_anterior:
                recomendacion = "Sobrecompra extrema. Riesgo inminente de corrección, considerar ventas."
            else:
                recomendacion = "Vender / Tomar ganancias. El impulso indica una corrección en curso."
        elif 2 < nota_empresa <= 4: # SMI alto (sobrecompra moderada)
            condicion_rsi = "algo sobrecomprado"
            smi_tendencia = "con un impulso alcista que podría estar agotándose."
            if smi_actual > smi_anterior:
                recomendacion = "Atentos a posible sobrecompra. El impulso alcista se está agotando."
            else:
                recomendacion = "Vigilar posible venta. El impulso muestra una disminución."
        elif 4 < nota_empresa <= 5: # SMI ligeramente sobrecomprado / entrando en zona neutra
            condicion_rsi = "muy poca sobrecompra"
            smi_tendencia = "manteniendo un impulso alcista sólido."
            if smi_actual > smi_anterior:
                recomendacion = "Impulso alcista fuerte. Cuidado con niveles de resistencia."
            else:
                recomendacion = "Tendencia de enfriamiento. Cuidado. Revisar soportes y resistencias."
        elif 5 < nota_empresa < 6: # Zona neutra
            condicion_rsi = "neutral"
            smi_tendencia = "en una fase de equilibrio."
            if smi_actual > smi_anterior:
                recomendacion = "Mantener (Neutro). El precio gana impulso."
            else:
                recomendacion = "Mantener (Neutro). El precio busca equilibrio."
        elif 6 <= nota_empresa < 7: # SMI ligeramente sobrevendido / entrando en zona neutra
            condicion_rsi = "muy poca sobreventa"
            smi_tendencia = "mostrando señales de recuperación."
            if smi_actual > smi_anterior:
                recomendacion = "Señal de recuperación. Posible compra con confirmación."
            else:
                recomendacion = "El impulso bajista persiste. Considerar cautela."
        elif 7 <= nota_empresa < 8: # SMI bajo (sobreventa moderada)
            condicion_rsi = "algo de sobreventa"
            smi_tendencia = "en una zona de sobreventa moderada, buscando un rebote."
            if smi_actual > smi_anterior:
                recomendacion = "Considerar posible compra. El impulso muestra un giro al alza."
            else:
                recomendacion = "Sobreventa moderada. Evaluar fortaleza de soportes, el precio podría caer más."
        elif 8 <= nota_empresa < 9: # SMI muy bajo (sobreventa fuerte)
            condicion_rsi = "sobreventa"
            smi_tendencia = "en una una sobreventa fuerte, con potencial de reversión."
            if smi_actual > smi_anterior:
                recomendacion = "Se acerca la hora de comprar. Fuerte señal de rebote."
            else:
                recomendacion = "Sobreventa significativa. Esperar confirmación de rebote antes de comprar."
        elif nota_empresa >= 9: # SMI extremadamente bajo (sobreventa extrema)
            condicion_rsi = "extremadamente sobrevendido"
            smi_tendencia = "en una una sobreventa extrema, lo que sugiere un rebote inminente."
            if smi_actual > smi_anterior:
                recomendacion = "Comprar. Excelente señal de reversión alcista."
            else:
                recomendacion = "Sobreventa extrema. El precio podría seguir cayendo a corto plazo, esperar confirmación de suelo."

        precio_objetivo_compra = 0.0
        base_precio_obj = soporte_1 if soporte_1 > 0 else current_price * 0.95

        # Lógica para definir el precio objetivo de compra
        if nota_empresa >= 7 and smi_actual > smi_anterior:
            precio_objetivo_compra = base_precio_obj
        elif nota_empresa >= 7 and smi_actual < smi_anterior:
            precio_objetivo_compra = base_precio_obj * 0.98
        else:
            drop_percentage_from_base = (7 - nota_empresa) / 7 * 0.15
            precio_objetivo_compra = base_precio_obj * (1 - drop_percentage_from_base)
            
        precio_objetivo_compra = max(0.01, round(precio_objetivo_compra, 2))
            
        # Lógica refinada para el mensaje de "Días para la Acción"
        dias_para_accion_str = "No estimado"
        smi_diff = hist['SMI_signal'].dropna().diff().iloc[-1] if len(hist['SMI_signal'].dropna()) > 1 else 0
        
        target_smi_venta_zona = 60
        target_smi_compra_zona = -60
        zona_umbral = 5  

        if smi_actual >= target_smi_venta_zona - zona_umbral and smi_actual > 0:
            dias_para_accion_str = "la empresa ya se encuentra en una **zona de potencial sobrecompra extrema**, indicando que la presión alcista podría estar agotándose y se anticipa una posible corrección o consolidación."
        elif smi_actual <= target_smi_compra_zona + zona_umbral and smi_actual < 0:
            dias_para_accion_str = "la empresa ya se encuentra en una **zona de potencial sobreventa extrema**, lo que sugiere que el precio podría estar cerca de un punto de inflexión al alza o un rebote técnico."
        elif smi_actual > smi_anterior and smi_actual < target_smi_venta_zona and smi_diff > 0.01:
            diferencia_a_target = target_smi_venta_zona - smi_actual
            dias_calculados = int(diferencia_a_target / smi_diff) if smi_diff != 0 else 0
            if dias_calculados >= 2:
                dias_para_accion_str = f"continuando su impulso alcista, podríamos estar aproximándonos a una potencial zona de toma de beneficios o venta en aproximadamente **{dias_calculados} días**."
            else:
                dias_para_accion_str = "el precio está consolidando una tendencia alcista y podría estar próximo a un punto de inflexión para una potencial toma de beneficios o venta."
        elif smi_actual < smi_anterior and smi_actual > target_smi_compra_zona and smi_diff < -0.01:
            diferencia_a_target = smi_actual - target_smi_compra_zona
            dias_calculados = int(diferencia_a_target / abs(smi_diff)) if smi_diff != 0 else 0
            if dias_calculados >= 2:
                dias_para_accion_str = f"continuando su impulso bajista, se estima una potencial zona de entrada o compra en aproximadamente **{dias_calculados} días**."
            else:
                dias_para_accion_str = "el precio está consolidando una tendencia bajista y podría estar próximo a un punto de inflexión para una potencial entrada o compra."
        else:
            dias_para_accion_str = "la empresa se encuentra en un periodo de consolidación, sin una dirección clara de impulso a corto plazo que anticipe un punto de acción inminente."

        # NO SE TRADUCE: Información de Yahoo Finance
        expansion_planes_raw = info.get("longBusinessSummary", "N/A")
        expansion_planes_translated = expansion_planes_raw # Se mantiene el texto original
        if expansion_planes_translated == "N/A" and expansion_planes_raw != "N/A":
            expansion_planes_translated = "Información de planes de expansión no disponible o no traducible en este momento."

        acuerdos_raw = info.get("agreements", "No disponibles")
        acuerdos_translated = acuerdos_raw # Se mantiene el texto original
        if acuerdos_translated == "No disponibles" and acuerdos_raw != "No disponibles":
            acuerdos_translated = "Información sobre acuerdos no disponible o no traducible en este momento."

        sentimiento_analistas_raw = info.get("recommendationKey", "N/A")
        sentimiento_analistas_translated = sentimiento_analistas_raw # Se mantiene el texto original
        if sentimiento_analistas_translated == "N/A" and sentimiento_analistas_raw != "N/A":
            sentimiento_analistas_translated = "Sentimiento de analistas no disponible o no traducible."
            
        # Obtener las últimas 7 notas de la empresa (calculadas a partir del SMI)
        # Asegúrate de que haya suficientes datos para al menos 7 notas
        recent_smi_signals = hist['SMI_signal'].dropna().tail(7)
        last_7_notes = [round((-(max(min(smi, 60), -60)) + 60) * 10 / 120, 1) for smi in recent_smi_signals]
        # Formatear las fechas para que sean legibles en el eje X del gráfico
        last_7_dates = [d.strftime('%m-%d') for d in recent_smi_signals.index] # Formato Mes-Día

        # Recopilación de todos los datos relevantes
        datos = {
            "TICKER": ticker,
            "NOMBRE_EMPRESA": info.get("longName", ticker),
            "PRECIO_ACTUAL": current_price,
            "VOLUMEN": current_volume,  
            "SOPORTE_1": soporte_1,
            "SOPORTE_2": soporte_2,
            "SOPORTE_3": soporte_3,
            "RESISTENCIA": round(hist["High"].max(), 2),
            "CONDICION_RSI": condicion_rsi,
            "RECOMENDACION": recomendacion,
            "SMI": smi_actual,
            "NOTA_EMPRESA": nota_empresa,
            "PRECIO_OBJETIVO_COMPRA": precio_objetivo_compra,
            "INGRESOS": info.get("totalRevenue", "N/A"),
            "EBITDA": info.get("ebitda", "N/A"),
            "BENEFICIOS": info.get("grossProfits", "N/A"),
            "DEUDA": info.get("totalDebt", "N/A"),
            "FLUJO_CAJA": info.get("freeCashflow", "N/A"),
            "EXPANSION_PLANES": expansion_planes_translated,
            "ACUERDOS": acuerdos_translated,
            "SENTIMIENTO_ANALISTAS": sentimiento_analistas_translated,
            "TENDENCIA_SOCIAL": "No disponible", # Placeholder, ya que no se extrae de yfinance
            "EMPRESAS_SIMILARES": ", ".join(info.get("category", "").split(",")) if info.get("category") else "No disponibles",
            "RIESGOS_OPORTUNIDADES": "No disponibles", # Placeholder
            "SMI_TENDENCIA": smi_tendencia,
            "DIAS_PARA_ACCION": dias_para_accion_str,
            "LAST_7_NOTES": last_7_notes,
            "LAST_7_DATES": last_7_dates
        }
    except Exception as e:
        print(f"❌ Error al obtener datos de {ticker}: {e}")
        return None

    return datos

def formatear_numero(valor):
    """Formatea un número a cadena con separador de miles y símbolo de euro."""
    try:
        numero = int(valor)
        return f"{numero:,} €".replace(",", ".") # Reemplaza coma por punto para formato español
    except (ValueError, TypeError):
        return "No disponible"
        
def construir_prompt_formateado(data, all_tickers, current_day_of_week):
    """
    Construye el prompt para Gemini con un formato HTML detallado,
    incorporando nuevos elementos de engagement y llamada a la acción,
    y ajustando el titular según la nota de la empresa.
    """
    # --- Lógica para adaptar el Titular a la Nota de la Empresa (rangos de punto a punto) ---
    nota = data['NOTA_EMPRESA']
    nombre_empresa = data['NOMBRE_EMPRESA']
    ticker = data['TICKER']
    
    if 0.0 <= nota < 1.0:
        titulo_post = f"Alerta Roja en <strong>{nombre_empresa} #{ticker} </strong>: ¿Colapso inminente? Análisis de su debilidad extrema"
    elif 1.0 <= nota < 2.0:
        titulo_post = f"Seria preocupación en <strong>{nombre_empresa} #{ticker} </strong>: Debilidad técnica extrema y riesgos latentes"
    elif 2.0 <= nota < 3.0:
        titulo_post = f"Cautela máxima con <strong>{nombre_empresa} #{ticker} </strong>: La tendencia bajista se intensifica"
    elif 3.0 <= nota < 4.0:
        titulo_post = f"Análisis de <strong>{nombre_empresa} #{ticker} </strong>: Señales de debilidad y la importancia de la paciencia"
    elif 4.0 <= nota < 5.0:
        titulo_post = f"<strong>{nombre_empresa} #{ticker} </strong>: En busca de dirección. ¿Consolidación o antesala de un movimiento?"
    elif 5.0 <= nota < 6.0:
        titulo_post = f"<strong>{nombre_empresa} #{ticker} </strong> en equilibrio: Un análisis de su fase neutral y oportunidades a largo plazo"
    elif 6.0 <= nota < 7.0:
        titulo_post = f"<strong>{nombre_empresa} #{ticker} </strong> despierta: Primeras señales de fortaleza y potencial de rebote"
    elif 7.0 <= nota < 8.0:
        titulo_post = f"Impulso en <strong>{nombre_empresa} #{ticker} </strong>: La oportunidad de compra se consolida"
    elif 8.0 <= nota < 9.0:
        titulo_post = f"¡Despegue inminente de <strong>{nombre_empresa} #{ticker} </strong>! Análisis de su explosivo potencial alcista"
    elif 9.0 <= nota <= 10.0:
        titulo_post = f"La oportunidad de la década en <strong>{nombre_empresa} #{ticker} </strong>: ¿Listo para multiplicar tu inversión?"
    else: # Fallback por si la nota está fuera de rango inesperadamente
        titulo_post = f"Análisis completo de <strong>{nombre_empresa} #{ticker} </strong>"


    # Lógica para soportes
    soportes_unicos = []
    temp_soportes = sorted([data['SOPORTE_1'], data['SOPORTE_2'], data['SOPORTE_3']], reverse=True)
    
    if len(temp_soportes) > 0 and temp_soportes[0] > 0.01:
        soportes_unicos.append(temp_soportes[0])
        for i in range(1, len(temp_soportes)):
            # Añade soportes si son suficientemente distintos del anterior y mayores que 0.01
            if temp_soportes[i] > 0.01 and soportes_unicos[-1] > 0 and abs(temp_soportes[i] - soportes_unicos[-1]) / soportes_unicos[-1] > 0.005:
                soportes_unicos.append(temp_soportes[i])
            
    soportes_texto = ""
    if len(soportes_unicos) == 1:
        soportes_texto = f"un soporte clave en <strong>{soportes_unicos[0]:,.2f} €</strong>."
    elif len(soportes_unicos) == 2:
        soportes_texto = f"dos soportes importantes en <strong>{soportes_unicos[0]:,.2f} €</strong> y <strong>{soportes_unicos[1]:,.2f} €</strong>."
    elif len(soportes_unicos) >= 3:
        soportes_texto = (f"tres soportes relevantes: el primero en <strong>{soportes_unicos[0]:,.2f} €</strong>, "
                          f"el segundo en <strong>{soportes_unicos[1]:,.2f} €</strong>, y el tercero en <strong>{soportes_unicos[2]:,.2f} €</strong>.")
    else:
        soportes_texto = "no presenta soportes claros en el análisis reciente, requiriendo un seguimiento cauteloso."

    # Cálculo del porcentaje de resistencia
    if float(data['PRECIO_ACTUAL']) > 0:
        resistencia_porcentaje = f"{((float(data['RESISTENCIA']) - float(data['PRECIO_ACTUAL'])) / float(data['PRECIO_ACTUAL']) * 100):.2f}%"
    else:
        resistencia_porcentaje = "no calculable debido a un precio actual no disponible o de 0€"

    # --- Lógica para el Call to Action del día siguiente (esto es lo que se ajusta a tu original) ---
    num_tickers_per_day = 10
    total_tickers_in_sheet = len(all_tickers)
    next_day_of_week = (current_day_of_week + 1) % 7

    start_index_next_day = (next_day_of_week * num_tickers_per_day) % total_tickers_in_sheet
    end_index_next_day = start_index_next_day + num_tickers_per_day
    
    tickers_for_tomorrow = []
    if end_index_next_day <= total_tickers_in_sheet:
        tickers_for_tomorrow = all_tickers[start_index_next_day:end_index_next_day]
    else:
        remaining_tickers = total_tickers_in_sheet - start_index_next_day
        tickers_for_tomorrow = all_tickers[start_index_next_day:] + all_tickers[:num_tickers_per_day - remaining_tickers]


    if tickers_for_tomorrow:
        tomorrow_companies_text = ", ".join([f"<strong>{t}</strong>" for t in tickers_for_tomorrow])
    else:
        tomorrow_companies_text = "otras empresas clave del mercado."

    # Datos para el gráfico Chart.js
    # Convertir las listas de Python a cadenas JSON para JavaScript
    labels_js = json.dumps(data['LAST_7_DATES'])
    notes_js = json.dumps(data['LAST_7_NOTES'])

    prompt = f"""
Actúa como un trader profesional con amplia experiencia en análisis técnico y mercados financieros. Genera un análisis completo en **formato HTML**, ideal para publicaciones web. Utiliza etiquetas `<h2>` para los títulos de sección y `<p>` para cada párrafo de texto. Redacta en primera persona, con total confianza en tu criterio y usando un lenguaje persuasivo y profesional.

Destaca los datos importantes como precios, notas de la empresa, cifras financieras y el nombre de la empresa utilizando la etiqueta `<strong>`. Asegúrate de que no haya asteriscos u otros símbolos de marcado en el texto final, solo HTML válido. Asegúrate de que todo esté escrito en español, independientemente del idioma de donde saques los datos, y que el texto fluya de manera natural y variada. No utilices el formato Markdown (doble asterisco, etc.) en el texto final, solo HTML puro.

**Asegúrate de que todo el post, desde el titular hasta la conclusión, sea coherente y refleje fielmente el sentimiento general indicado por la nota técnica de la empresa.**

Genera un análisis técnico y fundamental detallado de aproximadamente 1200 palabras sobre la empresa {data['NOMBRE_EMPRESA']}, utilizando los siguientes datos reales extraídos de Yahoo Finance. Presta especial atención a la **nota obtenida por la empresa**: {data['NOTA_EMPRESA']}. Amplía cada sección para ofrecer un análisis profundo y evitar repeticiones.

**Datos clave (utiliza estos datos para redactar el análisis):**
- Ticker: {data['TICKER']}
- Nombre de la empresa: {data['NOMBRE_EMPRESA']}
- Precio actual: {data['PRECIO_ACTUAL']}
- Volumen del último día completo: {data['VOLUMEN']}
- Soporte 1: {data['SOPORTE_1']}
- Soporte 2: {data['SOPORTE_2']}
- Soporte 3: {data['SOPORTE_3']}
- Resistencia clave: {data['RESISTENCIA']}
- Recomendación general: {data['RECOMENDACION']}
- Nota de la empresa (0-10): {data['NOTA_EMPRESA']} sobre 10
- Precio objetivo de compra: {data['PRECIO_OBJETIVO_COMPRA']}€
- Ingresos: {formatear_numero(data['INGRESOS'])}
- EBITDA: {formatear_numero(data['EBITDA'])}
- Beneficios: {formatear_numero(data['BENEFICIOS'])}
- Deuda: {formatear_numero(data['DEUDA'])}
- Flujo de caja: {formatear_numero(data['FLUJO_CAJA'])}
- Planes de expansión: {data['EXPANSION_PLANES']}
- Acuerdos: {data['ACUERDOS']}
- Sentimiento de analistas: {data['SENTIMIENTO_ANALISTAS']}
- Tendencia social: {data['TENDENCIA_SOCIAL']}
- Empresas similares: {data['EMPRESAS_SIMILARES']}
- Riesgos y oportunidades: {data['RIESGOS_OPORTUNIDADES']}
- Tendencia de impulso (SMI): {data['SMI_TENDENCIA']}
- Estimación para acción: {data['DIAS_PARA_ACCION']}
- Últimas 7 notas: {data['LAST_7_NOTES']}
- Últimas 7 fechas de notas: {data['LAST_7_DATES']}


Importante: si algún dato está marcado como "N/A", "No disponibles" o "No disponible", no lo menciones ni digas que falta. Integra la recomendación como una conclusión personal basada en tu experiencia y criterio profesional, sin atribuirla a un indicador específico. Asegura que el lenguaje sea dinámico y no repetitivo.

---
<h1>{titulo_post}</h1>



<h2>Análisis Inicial y Recomendación</h2>
<p>En el dinámico mercado actual, <strong>{data['NOMBRE_EMPRESA']} ({data['TICKER']})</strong> está enviando señales claras de un potencial giro alcista. ¿Es este el momento ideal para considerar una entrada? Mi análisis técnico apunta a que sí, con una oportunidad de compra inminente y un rebote en el horizonte.</p>
<p>La empresa cotiza actualmente a <strong>{data['PRECIO_ACTUAL']:,} €</strong>, un nivel que considero estratégico. Mi precio objetivo de compra se sitúa en <strong>{data['PRECIO_OBJETIVO_COMPRA']:,} €</strong>.
"""
    if data['PRECIO_ACTUAL'] > data['PRECIO_OBJETIVO_COMPRA']:
        prompt += f"""Este último representa el nivel más atractivo para una entrada conservadora, y aunque el precio actual está por encima, aún puede presentar una oportunidad si se evalúa cuidadosamente la relación riesgo/recompensa. Como analista, mi visión es que la convergencia hacia este objetivo podría ser el punto de partida para un movimiento significativo."""
    else:
        prompt += f"""Esto subraya una atractiva oportunidad de compra, al estar el precio actual por debajo de nuestro objetivo, sugiriendo un potencial de revalorización desde los niveles actuales."""
    prompt += f""" El volumen negociado recientemente, que alcanzó las <strong>{data['VOLUMEN']:,} acciones</strong>, es un factor clave que valida estos movimientos, y será crucial monitorearlo para confirmar la fuerza de cualquier tendencia emergente.</p>

<div style="width: 100%; max-width: 600px; margin: auto;">
    <canvas id="notesChart"></canvas>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
    document.addEventListener('DOMContentLoaded', function() {{
        const ctx = document.getElementById('notesChart').getContext('2d');
        const notesChart = new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {labels_js},
                datasets: [{{
                    label: 'Nota Técnica (0-10)',
                    data: {notes_js},
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.7)',
                        'rgba(255, 159, 64, 0.7)',
                        'rgba(255, 205, 86, 0.7)',
                        'rgba(75, 192, 192, 0.7)',
                        'rgba(54, 162, 235, 0.7)',
                        'rgba(153, 102, 255, 0.7)',
                        'rgba(201, 203, 207, 0.7)'
                    ],
                    borderColor: [
                        'rgb(255, 99, 132)',
                        'rgb(255, 159, 64)',
                        'rgb(255, 205, 86)',
                        'rgb(75, 192, 192)',
                        'rgb(54, 162, 235)',
                        'rgb(153, 102, 255)',
                        'rgb(201, 203, 207)'
                    ],
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Notas Técnicas de los Últimos 7 Días',
                        font: {{
                            size: 16
                        }}
                    }},
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 10,
                        title: {{
                            display: true,
                            text: 'Nota (0-10)'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Fecha'
                        }}
                    }}
                }}
            }}
        });
    }});
</script>

<p>La **nota técnica actual de la empresa es de {data['NOTA_EMPRESA']}/10**, lo que indica un fuerte potencial alcista. Esta calificación, basada en un análisis exhaustivo del Stochastic Momentum Index (SMI), sugiere que la acción se encuentra en una fase de acumulación o de inicio de un impulso positivo. Es un indicador clave que utilizo para identificar oportunidades de entrada cuando el mercado aún no ha descontado completamente el movimiento.</p>
<p>Mi recomendación es **{data['RECOMENDACION']}**. Esta postura se fundamenta en la confluencia de varios factores técnicos y fundamentales que analizaremos a continuación. Considero que el riesgo actual es favorable en relación con el potencial de beneficio.</p>

<h2>Análisis Técnico Detallado</h2>
<p>Desde una perspectiva técnica, la acción de <strong>{data['NOMBRE_EMPRESA']}</strong> muestra una estructura interesante. El precio actual de <strong>{data['PRECIO_ACTUAL']:,} €</strong> se encuentra en un punto crucial. Es vital observar cómo reacciona el precio en los próximos días, especialmente en relación con los niveles de soporte identificados.</p>
<p>Hemos identificado {soportes_texto} Estos niveles son críticos, ya que representan zonas donde la demanda históricamente ha superado a la oferta, deteniendo caídas y sirviendo como puntos de rebote. Un seguimiento cercano de la interacción del precio con estos soportes nos dará confirmación adicional de la fortaleza de la tendencia. La resistencia clave se sitúa en <strong>{data['RESISTENCIA']:,} €</strong>, lo que representa un potencial de revalorización del <strong>{resistencia_porcentaje}</strong> desde el precio actual. Superar este nivel con volumen confirmaría una continuación del movimiento alcista.</p>
<p>El Stochastic Momentum Index (SMI), un oscilador que valoro enormemente por su capacidad para medir el impulso y las condiciones de sobrecompra/sobreventa, nos muestra que la empresa se encuentra **{data['CONDICION_RSI']}**. La tendencia del SMI es **{data['SMI_TENDENCIA']}**, lo que sugiere que el impulso del precio está ganando fuerza. Esta divergencia o convergencia del SMI con el precio es una señal potente para anticipar movimientos. El valor actual del SMI es <strong>{data['SMI']:,}</strong>, lo que complementa mi análisis y refuerza la perspectiva de una oportunidad.</p>
<p>En cuanto al volumen, las <strong>{data['VOLUMEN']:,} acciones</strong> negociadas en el último día completo son un indicador vital. Un incremento de volumen en un movimiento alcista valida la fortaleza de la tendencia, mientras que un bajo volumen podría indicar una falta de convicción. Estaré atento a cualquier cambio significativo en el volumen que pueda confirmar o refutar la dirección esperada.</p>

<h2>Análisis Fundamental y Perspectivas</h2>
<p>Más allá de los gráficos, es fundamental entender la salud financiera de <strong>{data['NOMBRE_EMPRESA']}</strong>. La empresa ha reportado ingresos de <strong>{formatear_numero(data['INGRESOS'])}</strong>, un EBITDA de <strong>{formatear_numero(data['EBITDA'])}</strong>, y beneficios de <strong>{formatear_numero(data['BENEFICIOS'])}</strong>. Estas cifras nos dan una instantánea de su rendimiento operativo y su capacidad para generar ganancias. La deuda total de la empresa asciende a <strong>{formatear_numero(data['DEUDA'])}</strong>, y su flujo de caja libre es de <strong>{formatear_numero(data['FLUJO_CAJA'])}</strong>. Estos datos son cruciales para evaluar la solidez financiera y la capacidad de la empresa para afrontar sus obligaciones y financiar su crecimiento.</p>
<p>En cuanto a sus perspectivas de crecimiento, los planes de expansión de la empresa son: "{data['EXPANSION_PLANES']}". Estos planes, si se ejecutan con éxito, podrían ser un catalizador significativo para el precio de la acción. Además, la empresa ha estado involucrada en "{data['ACUERDOS']}", lo que podría indicar alianzas estratégicas o movimientos corporativos que impacten positivamente en su valoración.</p>
<p>El sentimiento general de los analistas es "{data['SENTIMIENTO_ANALISTAS']}". Aunque no es el único factor a considerar, el consenso de los expertos puede ofrecer una perspectiva adicional sobre las expectativas del mercado. La tendencia social actual de la empresa es "{data['TENDENCIA_SOCIAL']}". La percepción pública y la atención en redes sociales pueden influir en el interés de los inversores. Las empresas similares en el sector incluyen: "{data['EMPRESAS_SIMILARES']}". Comparar el rendimiento de <strong>{data['NOMBRE_EMPRESA']}</strong> con sus pares nos ayuda a contextualizar su posición en el mercado.</p>
<p>En mi evaluación, los principales riesgos y oportunidades para esta inversión son: "{data['RIESGOS_OPORTUNIDADES']}". Es vital que cada inversor evalúe estos factores en función de su propio perfil de riesgo y objetivos de inversión.</p>

<h2>Conclusión y Próximos Pasos</h2>
<p>En resumen, mi análisis de <strong>{data['NOMBRE_EMPRESA']} ({data['TICKER']})</strong> revela una oportunidad de inversión atractiva, respaldada por una sólida nota técnica y un impulso creciente. La estimación para una acción significativa es que {data['DIAS_PARA_ACCION']}.</p>
<p>Mi recomendación final es clara: **{data['RECOMENDACION']}**. Considero que este es un momento estratégico para aquellos inversores que buscan capitalizar un potencial movimiento alcista. Sin embargo, como siempre, la gestión del riesgo es primordial. Establezcan sus niveles de stop-loss y tomen ganancias de forma disciplinada.</p>
<p>Mañana, en nuestro próximo análisis, nos sumergiremos en las oportunidades y desafíos de **{tomorrow_companies_text}**. ¡Asegúrense de estar atentos para no perderse ninguna actualización clave del mercado!</p>
"""
    return prompt

def enviar_email(html_content, recipient_email, ticker_name):
    """
    Envía un email con el contenido HTML generado.
    Requiere las variables de entorno EMAIL_SENDER, EMAIL_PASSWORD y EMAIL_RECIPIENT.
    """
    sender_email = os.getenv('EMAIL_SENDER')
    sender_password = os.getenv('EMAIL_PASSWORD')

    if not sender_email or not sender_password or not recipient_email:
        print("❌ Error: Variables de entorno de email no configuradas correctamente.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Análisis de la Acción: {ticker_name}"
    msg["From"] = sender_email
    msg["To"] = recipient_email

    # Crea el cuerpo del email en HTML
    part_html = MIMEText(html_content, "html")
    msg.attach(part_html)

    try:
        # Conexión al servidor SMTP de Gmail
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        print(f"✅ Email enviado con éxito a {recipient_email} para {ticker_name}")
    except Exception as e:
        print(f"❌ Error al enviar el email para {ticker_name}: {e}")

def main():
    try:
        tickers = leer_google_sheets()
        if not tickers:
            print("No hay tickers para procesar. Saliendo.")
            return

        recipient_email = os.getenv('EMAIL_RECIPIENT')
        if not recipient_email:
            print("❌ Error: Variable de entorno EMAIL_RECIPIENT no configurada.")
            return
        
        # Obtener el día de la semana actual (0=Lunes, 6=Domingo)
        current_day_of_week = datetime.now().weekday() 

        for ticker in tickers:
            print(f"\n🚀 Procesando ticker: {ticker}")
            data = obtener_datos_yfinance(ticker)
            if data:
                print(f"✅ Datos obtenidos para {ticker}. Generando informe...")
                
                # Pasar all_tickers y current_day_of_week a la función que construye el prompt
                html_report = construir_prompt_formateado(data, tickers, current_day_of_week)
                
                if html_report:
                    # Enviar el email
                    enviar_email(html_report, recipient_email, data['NOMBRE_EMPRESA'])
                else:
                    print(f"❌ No se pudo generar el informe HTML para {ticker}.")
            else:
                print(f"❌ No se pudieron obtener o procesar datos para {ticker}. Saltando.")
            time.sleep(5) # Pequeña pausa entre tickers para evitar bloqueos por tasa

    except Exception as e:
        print(f"❌ Ha ocurrido un error crítico en la ejecución principal: {e}")

if __name__ == "__main__":
    main()
