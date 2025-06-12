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
    Lee los tickers de la columna A de una hoja de cálculo de Google Sheets.
    Requiere que las variables de entorno GOOGLE_APPLICATION_CREDENTIALS
    y SPREADSHEET_ID estén configuradas.
    """
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
    sheet = service.spreadsheets().values()
    result = sheet.get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    values = result.get('values', [])

    if not values:
        print('No se encontraron datos en la hoja de cálculo.')
        return []
    else:
        print('Datos leídos de la hoja:')
        # Filtra y limpia los tickers para asegurar que solo se devuelvan valores válidos
        tickers = [row[0].strip() for row in values if row and row[0].strip()]
        for ticker in tickers:
            print(ticker)
        return tickers


length_k = 10
length_d = 3
ema_signal_len = 10
smooth_period = 5

def calculate_smi_tv(df):
    """
    Calcula el indicador SMI (Stochastic Momentum Index) para un DataFrame dado.
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

    # Manejo de división por cero para avgdiff
    smi_raw = pd.Series(0.0, index=df.index) # Inicializa con ceros
    non_zero_avgdiff_mask = avgdiff != 0
    smi_raw[non_zero_avgdiff_mask] = (avgrel[non_zero_avgdiff_mask] / (avgdiff[non_zero_avgdiff_mask] / 2)) * 100

    smi_smoothed = smi_raw.rolling(window=smooth_period).mean()
    smi_signal = smi_smoothed.ewm(span=ema_signal_len, adjust=False).mean()

    df = df.copy()
    df['SMI'] = smi_smoothed
    df['SMI_signal'] = smi_signal
    
    return df

def find_significant_supports(df, current_price, window=40, tolerance_percent=0.01, max_deviation_percent=0.15):
    """
    Identifica los 3 soportes más significativos y cercanos al precio actual
    basándose en mínimos locales y agrupaciones de precios.
    """
    recent_data = df.tail(window)
    lows = recent_data['Low']
    
    potential_supports = []
    
    for i in range(1, len(lows) - 1):
        if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
            # Asegúrate de que el soporte no sea cero antes de añadirlo
            if lows.iloc[i] > 0:
                potential_supports.append(lows.iloc[i])

    if not potential_supports:
        # Filtrar lows que son cero o negativos si se usan como soportes por defecto
        potential_supports = [l for l in lows.tolist() if l > 0]
        
    support_zones = {}
    for support in potential_supports:
        found_zone = False
        for zone_level in support_zones.keys():
            # Evitar división por cero si zone_level es 0
            if zone_level != 0 and abs(support - zone_level) / zone_level <= tolerance_percent:
                support_zones[zone_level].append(support)
                found_zone = True
                break
        if not found_zone:
            support_zones[support] = [support]
            
    final_supports = []
    for zone_level, values in support_zones.items():
        avg_support = np.mean(values)
        if avg_support < current_price:
            # Evitar división por cero si current_price es 0
            if current_price != 0 and abs(current_price - avg_support) / current_price <= max_deviation_percent:
                final_supports.append({'level': avg_support, 'frequency': len(values)})

    final_supports.sort(key=lambda x: (abs(x['level'] - current_price), -x['frequency']))
    
    top_3_supports = [round(s['level'], 2) for s in final_supports if s['level'] < current_price][:3]
    
    if len(top_3_supports) < 3:
        # Filtrar lows que son cero o negativos
        sorted_lows = sorted([l for l in lows.tolist() if l < current_price and l > 0], reverse=True)
        for low_val in sorted_lows:
            rounded_low_val = round(low_val, 2)
            if rounded_low_val not in top_3_supports:
                top_3_supports.append(rounded_low_val)
                if len(top_3_supports) == 3:
                    break
    
    while len(top_3_supports) < 3:
        if len(top_3_supports) > 0:
            # Asegura que el cálculo no genere un soporte cero o negativo si el último es muy pequeño
            next_support = round(top_3_supports[-1] * 0.95, 2)
            top_3_supports.append(max(0.01, next_support)) # Asegura un mínimo de 0.01
        else:
            # Asegura que el cálculo no genere un soporte cero o negativo si current_price es muy pequeño
            default_support = round(current_price * 0.90, 2)
            top_3_supports.append(max(0.01, default_support)) # Asegura un mínimo de 0.01
            
    return top_3_supports

def traducir_texto_con_gemini(text, max_retries=3, initial_delay=5):
    """
    Traduce un texto al español usando la API de Gemini, con reintentos en caso de errores de cuota.
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
            response = model.generate_content(f"Traduce el siguiente texto al español de forma concisa y profesional: \"{text}\"")
            translated_text = response.text.strip().replace("**", "").replace("*", "")
            return translated_text
        except Exception as e:
            if "429 You exceeded your current quota" in str(e):
                try:
                    match = re.search(r"retry_delay \{\s*seconds: (\d+)", str(e))
                    if match:
                        server_delay = int(match.group(1))
                        delay = max(delay, server_delay + 1)
                except:
                    pass
                
                print(f"❌ Cuota de Gemini excedida al traducir. Reintentando en {delay} segundos... (Intento {retries + 1}/{max_retries})")
                time.sleep(delay)
                retries += 1
                delay *= 2
            else:
                print(f"❌ Error al traducir texto con Gemini (no de cuota): {e}")
                return text
    print(f"❌ Falló la traducción después de {max_retries} reintentos.")
    return text

def obtener_datos_yfinance(ticker):
    """
    Obtiene datos históricos y de información de un ticker usando yfinance.
    Calcula SMI y soportes, y traduce información relevante.
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    
    hist = stock.history(period="90d", interval="1d")

    if hist.empty:
        print(f"❌ No se pudieron obtener datos históricos para {ticker}")
        return None

    try:
        # Asegúrate de que las columnas críticas existan antes de calcular SMI
        required_cols = ['High', 'Low', 'Close', 'Volume'] # Añadido 'Volume' a las columnas requeridas
        if not all(col in hist.columns for col in required_cols):
            print(f"❌ Datos históricos incompletos para {ticker}. Faltan columnas: {set(required_cols) - set(hist.columns)}")
            return None

        hist = calculate_smi_tv(hist)
        
        if 'SMI_signal' not in hist.columns or hist['SMI_signal'].empty or len(hist['SMI_signal'].dropna()) < 2:
            print(f"❌ SMI_signal no disponible o insuficiente para {ticker}")
            return None

        smi_actual = round(hist['SMI_signal'].dropna().iloc[-1], 2)
        smi_anterior = round(hist['SMI_signal'].dropna().iloc[-2], 2)
        
        smi_tendencia = "subiendo" if smi_actual > smi_anterior else "bajando" if smi_actual < smi_anterior else "estable"

        current_price = round(info.get("regularMarketPrice", info.get("currentPrice", 0)), 2)

        # Si current_price es 0, no podemos calcular soportes significativos basados en porcentajes
        if current_price == 0:
            print(f"Advertencia: El precio actual para {ticker} es 0. Los soportes se establecerán en 0 o un valor mínimo.")
            soportes = [0.00, 0.00, 0.00] # O un valor mínimo para evitar errores
        else:
            soportes = find_significant_supports(hist, current_price)
        
        soporte_1 = soportes[0] if len(soportes) > 0 else 0
        soporte_2 = soportes[1] if len(soportes) > 1 else 0
        soporte_3 = soportes[2] if len(soportes) > 2 else 0

        # CORRECCIÓN: Asegúrate de que los soportes no sean 0 si se van a usar en comparaciones
        soporte_1 = max(0.01, soporte_1) if soporte_1 == 0 and current_price != 0 else soporte_1
        soporte_2 = max(0.01, soporte_2) if soporte_2 == 0 and current_price != 0 else soporte_2
        soporte_3 = max(0.01, soporte_3) if soporte_3 == 0 and current_price != 0 else soporte_3

        # Obtener el volumen del día anterior
        current_volume = 0
        if len(hist['Volume']) >= 2: # Asegúrate de que haya al menos 2 días de datos para el día anterior
            current_volume = hist['Volume'].iloc[-2] # El volumen del penúltimo día (día anterior)

        nota_empresa = round((-(max(min(smi_actual, 60), -60)) + 60) * 10 / 120, 1)

        recomendacion = "Indefinido"
        condicion_rsi = "desconocido"
        
        # --- LÓGICA DE RECOMENDACIÓN Y TENDENCIA DEL SMI MÁS MATIZADA ---
        if nota_empresa <= 2: # SMI muy alto (sobrecompra fuerte: SMI entre 60 y 100)
            condicion_rsi = "muy sobrecomprado"
            smi_tendencia = "mostrando un agotamiento alcista."
            if smi_actual > smi_anterior:
                recomendacion = "Sobrecompra extrema. Riesgo inminente de corrección, considerar ventas."
            else:
                recomendacion = "Vender / Tomar ganancias. El impulso indica una corrección en curso."
        elif 2 < nota_empresa <= 4: # SMI alto (sobrecompra moderada: SMI entre 30 y 60)
            condicion_rsi = "algo sobrecomprado"
            smi_tendencia = "con un impulso alcista que podría estar agotándose."
            if smi_actual > smi_anterior:
                recomendacion = "Atentos a posible sobrecompra. El impulso alcista se está agotando."
            else:
                recomendacion = "Vigilar posible venta. El impulso muestra una disminución."
        elif 4 < nota_empresa <= 5: # SMI ligeramente sobrecomprado / entrando en zona neutra (SMI entre 10 y 30)
            condicion_rsi = "muy poca sobrecompra"
            smi_tendencia = "manteniendo un impulso alcista sólido."
            if smi_actual > smi_anterior:
                recomendacion = "Impulso alcista fuerte. Cuidado con niveles de resistencia."
            else:
                recomendacion = "Tendencia de enfriamiento. Cuidado. Revisar soportes y resistencias."
        elif 5 < nota_empresa < 6: # Zona neutra (SMI entre -10 y 10)
            condicion_rsi = "neutral"
            smi_tendencia = "en una fase de equilibrio."
            if smi_actual > smi_anterior:
                recomendacion = "Mantener (Neutro). El precio gana impulso."
            else:
                recomendacion = "Mantener (Neutro). El precio busca equilibrio."
        elif 6 <= nota_empresa < 7: # SMI ligeramente sobrevendido / entrando en zona neutra (SMI entre -30 y -10)
            condicion_rsi = "muy poca sobreventa"
            smi_tendencia = "mostrando señales de recuperación."
            if smi_actual > smi_anterior:
                recomendacion = "Señal de recuperación. Posible compra con confirmación."
            else:
                recomendacion = "El impulso bajista persiste. Considerar cautela."
        elif 7 <= nota_empresa < 8: # SMI bajo (sobreventa moderada: SMI entre -60 y -30)
            condicion_rsi = "algo de sobreventa"
            smi_tendencia = "en una zona de sobreventa moderada, buscando un rebote."
            if smi_actual > smi_anterior:
                recomendacion = "Considerar posible compra. El impulso muestra un giro al alza."
            else:
                recomendacion = "Sobreventa moderada. Evaluar fortaleza de soportes, el precio podría caer más."
        elif 8 <= nota_empresa < 9: # SMI muy bajo (sobreventa fuerte: SMI entre -100 y -60)
            condicion_rsi = "sobreventa"
            smi_tendencia = "en una zona de sobreventa fuerte, con potencial de reversión."
            if smi_actual > smi_anterior:
                recomendacion = "Se acerca la hora de comprar. Fuerte señal de rebote."
            else:
                recomendacion = "Sobreventa significativa. Esperar confirmación de rebote antes de comprar."
        elif nota_empresa >= 9: # SMI extremadamente bajo (sobreventa extrema: SMI muy por debajo de -60)
            condicion_rsi = "extremadamente sobrevendido"
            smi_tendencia = "en una una sobreventa extrema, lo que sugiere un rebote inminente."
            if smi_actual > smi_anterior:
                recomendacion = "Comprar. Excelente señal de reversión alcista."
            else:
                recomendacion = "Sobreventa extrema. El precio podría seguir cayendo a corto plazo, esperar confirmación de suelo."

        precio_objetivo_compra = 0.0
        # Asegúrate de que base_precio_obj no sea 0 si se va a usar en una división, aunque aquí es multiplicando
        base_precio_obj = soporte_1 if soporte_1 > 0 else current_price * 0.95

        if nota_empresa >= 7 and smi_actual > smi_anterior:
            precio_objetivo_compra = base_precio_obj
        elif nota_empresa >= 7 and smi_actual < smi_anterior:
            precio_objetivo_compra = base_precio_obj * 0.98
        else:
            drop_percentage_from_base = (7 - nota_empresa) / 7 * 0.15
            precio_objetivo_compra = base_precio_obj * (1 - drop_percentage_from_base)
            
        precio_objetivo_compra = max(0.01, round(precio_objetivo_compra, 2))
            
        ### --- LÓGICA REFINADA PARA EL MENSAJE DE "DIAS PARA LA ACCION" ---
        dias_para_accion_str = "No estimado"
        smi_diff = hist['SMI_signal'].dropna().diff().iloc[-1] if len(hist['SMI_signal'].dropna()) > 1 else 0
        
        target_smi_venta_zona = 60
        target_smi_compra_zona = -60
        zona_umbral = 5  

        if smi_actual >= target_smi_venta_zona - zona_umbral and smi_actual > 0:
            dias_para_accion_str = "la empresa ya se encuentra en una **zona de potencial sobrecompra extrema**, indicando que la presión alcista podría estar agotándose y se anticipa una posible corrección o consolidación."
        elif smi_actual <= target_smi_compra_zona + zona_umbral and smi_actual < 0:
            dias_para_accion_str = "la empresa ya se encuentra en una **zona de potencial sobreventa extrema**, lo que sugiere que el precio podría estar cerca de un punto de inflexión al alza o un rebote técnico."
        # Asegúrate de smi_diff no sea 0 para estas divisiones
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
        # --------------------------------------------------------------------------------

        expansion_planes_raw = info.get("longBusinessSummary", "N/A")
        expansion_planes_translated = traducir_texto_con_gemini(expansion_planes_raw[:5000])
        if expansion_planes_translated == "N/A" and expansion_planes_raw != "N/A":
            expansion_planes_translated = "Información de planes de expansión no disponible o no traducible en este momento."

        acuerdos_raw = info.get("agreements", "No disponibles")
        acuerdos_translated = traducir_texto_con_gemini(acuerdos_raw)
        if acuerdos_translated == "No disponibles" and acuerdos_raw != "No disponibles":
            acuerdos_translated = "Información sobre acuerdos no disponible o no traducible en este momento."

        sentimiento_analistas_raw = info.get("recommendationKey", "N/A")
        sentimiento_analistas_translated = traducir_texto_con_gemini(sentimiento_analistas_raw)
        if sentimiento_analistas_translated == "N/A" and sentimiento_analistas_raw != "N/A":
            sentimiento_analistas_translated = "Sentimiento de analistas no disponible o no traducible."
            
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
            "TENDENCIA_SOCIAL": "No disponible",
            "EMPRESAS_SIMILARES": ", ".join(info.get("category", "").split(",")) if info.get("category") else "No disponibles",
            "RIESGOS_OPORTUNIDADES": "No disponibles",
            "SMI_TENDENCIA": smi_tendencia,
            "DIAS_PARA_ACCION": dias_para_accion_str
        }
    except Exception as e:
        print(f"❌ Error al obtener datos de {ticker}: {e}")
        return None

    return datos

def formatear_numero(valor):
    """
    Formatea un número grande como una cadena con comas y el símbolo de euro.
    """
    try:
        numero = int(valor)
        return f"{numero:,} €"
    except (ValueError, TypeError):
        return "No disponible"
        
def construir_prompt_formateado(data, next_tickers_list=None):
    """
    Construye el prompt HTML formateado para la API de Gemini
    a partir de los datos obtenidos y la lista de próximos tickers.
    """
    soportes_unicos = []
    temp_soportes = sorted([data['SOPORTE_1'], data['SOPORTE_2'], data['SOPORTE_3']], reverse=True)
    
    # Filtra los soportes que son 0 para no incluirlos en el texto a menos que sea necesario
    if len(temp_soportes) > 0 and temp_soportes[0] > 0.01: # Considera 0.01 como umbral mínimo para ser un soporte real
        soportes_unicos.append(temp_soportes[0])
        for i in range(1, len(temp_soportes)):
            if temp_soportes[i] > 0.01 and soportes_unicos[-1] > 0 and abs(temp_soportes[i] - soportes_unicos[-1]) / soportes_unicos[-1] > 0.005:
                soportes_unicos.append(temp_soportes[i])
            
    soportes_texto = ""
    if len(soportes_unicos) == 1:
        soportes_texto = f"un soporte clave en <strong>{soportes_unicos[0]:,} €</strong>."
    elif len(soportes_unicos) == 2:
        soportes_texto = f"dos soportes importantes en <strong>{soportes_unicos[0]:,} €</strong> y <strong>{soportes_unicos[1]:,} €</strong>."
    elif len(soportes_unicos) >= 3:
        soportes_texto = (f"tres soportes relevantes: el primero en <strong>{soportes_unicos[0]:,} €</strong>, "
                          f"el segundo en <strong>{soportes_unicos[1]:,} €</strong>, y el tercero en <strong>{soportes_unicos[2]:,} €</strong>.")
    else:
        soportes_texto = "no presenta soportes claros en el análisis reciente, requiriendo un seguimiento cauteloso."

    if float(data['PRECIO_ACTUAL']) > 0:
        resistencia_porcentaje = f"{((float(data['RESISTENCIA']) - float(data['PRECIO_ACTUAL'])) / float(data['PRECIO_ACTUAL']) * 100):.2f}%"
    else:
        resistencia_porcentaje = "no calculable debido a un precio actual no disponible o de 0€"

    # Derive values for the strategy section dynamically
    entry_point_1_val = max(0.01, data['SOPORTE_1'])
    # entry_point_2 is typically the second support, or a slightly lower point from the first for tiered entry
    entry_point_2_val = max(0.01, data['SOPORTE_2'] if data['SOPORTE_2'] > 0 else data['SOPORTE_1'] * 0.98) # Use S2 if available, else a bit below S1
    stop_loss_val = max(0.01, data['SOPORTE_3'] * 0.98 if data['SOPORTE_3'] > 0 else data['SOPORTE_2'] * 0.98 if data['SOPORTE_2'] > 0 else data['SOPORTE_1'] * 0.98) # Slightly below the lowest relevant support
    take_profit_val = data['RESISTENCIA'] # At the resistance

    # Ensure entry_point_1_val is greater than or equal to entry_point_2_val for logical consistency
    if entry_point_1_val < entry_point_2_val:
        entry_point_1_val, entry_point_2_val = entry_point_2_val, entry_point_1_val # Swap them

    # If the two entry points are too close, make the second one slightly lower
    if abs(entry_point_1_val - entry_point_2_val) / max(0.01, entry_point_1_val) < 0.005:
        if entry_point_2_val > 0.01: # Ensure not to make it zero or negative
            entry_point_2_val = max(0.01, entry_point_1_val * 0.99)
        else: # If entry_point_2_val is already very small, set it to a default offset from entry_point_1_val
            entry_point_2_val = max(0.01, entry_point_1_val - 0.1) # Arbitrary small offset for distinctness


    # Logic for next_day_company_placeholder - Now gets up to 10 next tickers
    next_day_company_dynamic = "otra empresa relevante o un sector en tendencia"
    if next_tickers_list and len(next_tickers_list) > 0:
        next_companies_to_show = next_tickers_list[:10] # Take up to the next 10
        if len(next_companies_to_show) == 1:
            next_day_company_dynamic = next_companies_to_show[0]
        elif len(next_companies_to_show) > 1:
            # Join with comma and 'y' for the last one
            next_day_company_dynamic = f"{', '.join(next_companies_to_show[:-1])} y {next_companies_to_show[-1]}"


    prompt = f"""
Actúa como un trader profesional con amplia experiencia en análisis técnico y mercados financieros. Genera el análisis completo en **formato HTML**, ideal para publicaciones web. Utiliza etiquetas `<h2>` para los títulos de sección y `<p>` para cada párrafo de texto. Redacta en primera persona, con total confianza en tu criterio. 

Destaca los datos importantes como precios, notas de la empresa, cifras financieras y el nombre de la empresa utilizando la etiqueta `<strong>`. Asegúrate de que no haya asteriscos u otros símbolos de marcado en el texto final, solo HTML válido. Asegurate que todo este escrito en español independientemente del idioma de donde saques los datos.

Genera un análisis técnico completo de aproximadamente 1200 palabras sobre la empresa {data['NOMBRE_EMPRESA']}, utilizando los siguientes datos reales extraídos de Yahoo Finance. Presta especial atención a la **nota obtenida por la empresa**: {data['NOTA_EMPRESA']}.

**Datos clave (para tu referencia interna, no para copiar verbatim):**
- Precio actual: {data['PRECIO_ACTUAL']}
- Volumen del último día completo: {data['VOLUMEN']}
- Soporte 1: {data['SOPORTE_1']}
- Soporte 2: {data['SOPORTE_2']}
- Soporte 3: {data['SOPORTE_3']}
- Resistencia clave: {data['RESISTENCIA']}
- Recomendación general: {data['RECOMENDACION']}
- Nota de la empresa (0-10): {data['NOTA_EMPRESA']} sobre 10
- Precio objetivo de compra: {data['PRECIO_OBJETIVO_COMPRA']}€
- Ingresos: {data['INGRESOS']}, EBITDA: {data['EBITDA']}, Beneficios: {data['BENEFICIOS']}
- Deuda: {data['DEUDA']}, Flujo de caja: {data['FLUJO_CAJA']}
- Planes de expansión: {data['EXPANSION_PLANES']}
- Acuerdos: {data['ACUERDOS']}
- Sentimiento analistas: {data['SENTIMIENTO_ANALISTAS']}
- Tendencia social: {data['TENDENCIA_SOCIAL']}
- Empresas similares: {data['EMPRESAS_SIMILARES']}
- Riesgos y oportunidades: {data['RIESGOS_OPORTUNIDADES']}
- Tendencia de impulso actual (SMI): {data['SMI_TENDENCIA']}.
- Estimación para acción: {data['DIAS_PARA_ACCION']}.

Importante: si algún dato no está disponible ("N/A", "No disponibles", "No disponible"), no lo menciones ni digas que falta. No expliques que la recomendación proviene de un indicador o dato específico. La recomendación debe presentarse como una conclusión personal basada en tu experiencia y criterio profesional como un analista.

---
<h1>[Crea un titular que capte la atención y sugiera una oportunidad o una pregunta que el lector querrá responder, basándote en la recomendación '{data['RECOMENDACION']}' y la tendencia de impulso '{data['SMI_TENDENCIA']}' para '{data['NOMBRE_EMPRESA']} ({data['TICKER']})'. Ejemplos: "¿Logista Integral (LOG.MC) a punto de despegar? Un giro alcista en el radar de los inversores" o "Oportunidad en Logista (LOG.MC): El análisis que te dice cuándo comprar para un rebote"]</h1>

<h2>Análisis Inicial y Recomendación</h2>
<p>[En el dinámico mercado actual, **{data['NOMBRE_EMPRESA']} ({data['TICKER']})** está enviando señales claras de un potencial giro. ¿Es este el momento ideal para considerar una entrada? Nuestro análisis técnico apunta a una **{data['RECOMENDACION']}** inminente. Desarrolla esta idea inicial como un gancho, expandiendo en la recomendación y la tendencia de impulso. Concluye este párrafo diciendo que mi recomendación principal es **{data['RECOMENDACION']}**].</p>

<p>La empresa cotiza actualmente a <strong>{data['PRECIO_ACTUAL']:,} €</strong>. Mi precio objetivo de compra se sitúa en <strong>{data['PRECIO_OBJETIVO_COMPRA']:,} €</strong>. [Si el precio actual es superior al precio objetivo de compra, explica que este último representa el nivel más atractivo para una entrada conservadora, y que el precio actual, aunque por encima, aún puede presentar una oportunidad si se evalúa el riesgo/recompensa. Si es inferior, recalca la oportunidad de compra al estar por debajo del objetivo]. El volumen negociado recientemente alcanza las <strong>{data['VOLUMEN']:,} acciones</strong>.</p>

<p>Asignamos una <strong>nota técnica de {data['NOTA_EMPRESA']} sobre 10</strong>. Esta puntuación refleja [explica concisamente qué significa esa puntuación en términos de riesgo, potencial de crecimiento, y la solidez *técnica* de la compañía para el corto plazo]. A continuación, detallo una visión más completa de mi evaluación profesional, desarrollada en base a una combinación de indicadores técnicos y fundamentos económicos.</p>

---
<h2>Datos Clave del Análisis de {data['NOMBRE_EMPRESA']}</h2>
<p>Para una visión rápida de los puntos más importantes de nuestra evaluación, aquí te presento un resumen en formato de lista para facilitar su lectura:</p>
<ul>
    <li><strong>Precio Actual:</strong> <strong>{data['PRECIO_ACTUAL']:,} €</strong></li>
    <li><strong>Volumen (último día):</strong> <strong>{data['VOLUMEN']:,} acciones</strong></li>
    <li><strong>Soporte Clave 1:</strong> <strong>{data['SOPORTE_1']:,} €</strong></li>
    <li><strong>Soporte Clave 2:</strong> <strong>{data['SOPORTE_2']:,} €</strong></li>
    <li><strong>Soporte Clave 3:</strong> <strong>{data['SOPORTE_3']:,} €</strong></li>
    <li><strong>Resistencia Importante:</strong> <strong>{data['RESISTENCIA']:,} €</strong> (a un {resistencia_porcentaje} del precio actual)</li>
    <li><strong>Nota Técnica (0-10):</strong> <strong>{data['NOTA_EMPRESA']} sobre 10</strong></li>
    <li><strong>Recomendación Principal:</strong> <strong>{data['RECOMENDACION']}</strong></li>
    <li><strong>Precio Objetivo de Compra:</strong> <strong>{data['PRECIO_OBJETIVO_COMPRA']:,} €</strong></li>
    <li><strong>Tendencia de Impulso (SMI):</strong> {data['SMI_TENDENCIA']}</li>
    <li><strong>Estimación de Acción:</strong> {data['DIAS_PARA_ACCION']}</li>
</ul>

---
<h2>Análisis a Corto Plazo: Soportes y Resistencias y Dinámica del Impulso</h2>
<p>Para entender los posibles movimientos a corto plazo en <strong>{data['NOMBRE_EMPRESA']}</strong>, es fundamental analizar el comportamiento reciente del volumen y las zonas clave de soporte y resistencia.</p>

<p>En este momento, observo {soportes_texto} La resistencia clave se encuentra en <strong>{data['RESISTENCIA']:,} €</strong>, situada a una distancia del <strong>{resistencia_porcentaje}</strong> desde el precio actual. Estas zonas técnicas pueden actuar como puntos de inflexión, y su cercanía o lejanía tiene implicaciones operativas claras.</p>

<p>Un aspecto crucial en el análisis de corto plazo es la dinámica de impulso de la empresa. Mi evaluación profesional indica que la tendencia actual se caracteriza por: <strong>{data['SMI_TENDENCIA']}</strong>. [PROFUNDIZA AQUÍ: Explica con detalle y de forma concisa lo que significa esta tendencia (por ejemplo, "sobreventa moderada, buscando un rebote" o "impulso alcista que podría estar agotándose") en términos de movimientos de precio esperados, psicología del mercado, y la probabilidad de un giro o continuación. Usa analogías sencillas si es útil. Extiende este párrafo para cubrir el significado práctico de la tendencia SMI y lo que implica para los próximos días/semanas. Por ejemplo, si la tendencia es 'sobreventa moderada, buscando un rebote', explica que esto significa que la acción ha sido 'castigada' en exceso y hay una alta probabilidad de que los compradores tomen el control, impulsando el precio al alza. Luego, analiza el volumen de <strong>{data['VOLUMEN']:,} acciones</strong> en relación con los movimientos recientes y soportes/resistencias. Comenta si el volumen actual es 'saludable', 'bajo', 'elevado' o 'anormal' para confirmar la validez de los movimientos de precio en los soportes y resistencias, y cómo esto valida o contradice la tendencia de impulso. La nota técnica de <strong>{data['NOTA_EMPRESA']} sobre 10</strong> debe ser un factor clave aquí para contextualizar el riesgo/recompensa de corto plazo. Redacta un análisis de mínimo 150 palabras, con lectura segmentada, mencionando cómo estos niveles influyen en la operativa a corto plazo. Concluye este párrafo diciendo: "En este contexto, {data['DIAS_PARA_ACCION']}. Esto indica un plazo claro para el posible movimiento, que debemos monitorear de cerca."]</p>

---
<h2>Visión a Largo Plazo y Fundamentales</h2>
<p>En un enfoque a largo plazo, el análisis se vuelve más robusto y se apoya en los fundamentos reales del negocio. Aquí, la evolución de <strong>{data['NOMBRE_EMPRESA']}</strong> dependerá en gran parte de sus cifras estructurales y sus perspectivas estratégicas. Para esta sección, la **nota técnica ({data['NOTA_EMPRESA']} sobre 10) NO debe influir en la valoración**. El análisis debe basarse **exclusivamente en los datos financieros y estratégicos** proporcionados y en una evaluación crítica de su solidez y potencial.</p>

<p>En el último ejercicio, los ingresos declarados fueron de <strong>{formatear_numero(data['INGRESOS'])}</strong>, el EBITDA alcanzó <strong>{formatear_numero(data['EBITDA'])}</strong>, y los beneficios netos se situaron en torno a <strong>{formatear_numero(data['BENEFICIOS'])}</strong>. 
En cuanto a su posición financiera, la deuda asciende a <strong>{formatear_numero(data['DEUDA'])}</strong>, y el flujo de caja operativo es de <strong>{formatear_numero(data['FLUJO_CAJA'])}</strong>.</p>

<p>[Si 'EXPANSION_PLANES' o 'ACUERDOS' contienen texto relevante y no genérico (que no sea "N/A", "No disponibles", "No disponible"), sintetízalo y comenta su posible impacto estratégico, evaluando si son oportunidades reales o riesgos. Si la información es demasiado breve o indica 'no disponible/no traducible', elabora sobre la importancia general de tales estrategias (expansión, acuerdos) para el sector de la empresa o para la empresa en sí, sin inventar detalles específicos. Redacta una proyección fundamentada (mínimo 150 palabras) con párrafos de máximo 3 líneas. Debe integrar estas cifras con una interpretación crítica de la solvencia, rentabilidad, crecimiento y las perspectivas estratégicas de la empresa, y **mojarse** con una valoración clara sobre su potencial a largo plazo basada *únicamente* en estos fundamentales. Por ejemplo, si los ingresos y beneficios crecen, comenta si este crecimiento es sostenible y qué implicaciones tiene para el futuro de la empresa.]</p>

---
<h2>Estrategia de Inversión y Gestión de Riesgos</h2>
<p>Basado en mi análisis profesional, propongo una estrategia de inversión y gestión de riesgos clara para <strong>{data['NOMBRE_EMPRESA']}</strong>. Mi recomendación sería considerar una posible entrada cerca del soporte de <strong>{entry_point_1_val:,.2f} €</strong> o, idealmente, en una zona aún más atractiva alrededor de los <strong>{entry_point_2_val:,.2f} €</strong>.</p>
<p>Para gestionar el riesgo de forma efectiva y proteger nuestro capital, es crucial establecer un stop loss ajustado. Sugiero situarlo justo por debajo del soporte más bajo relevante para nuestra estrategia, por ejemplo, en <strong>{stop_loss_val:,.2f} €</strong>. Esta medida limita las pérdidas potenciales si el mercado no se comporta como esperamos.</p>
<p>Mi objetivo de beneficio (Take Profit) a corto plazo se sitúa en la resistencia clave de <strong>{take_profit_val:,.2f} €</strong>. Este nivel representa un punto donde el precio históricamente ha encontrado dificultad para avanzar, y por tanto, es una excelente zona para asegurar ganancias.</p>
<p>Esta configuración de entrada, stop loss y objetivo de beneficio está diseñada para ofrecer una relación riesgo/recompensa favorable, optimizando el potencial de ganancias mientras se controlan las exposiciones negativas. La paciencia y la disciplina en la ejecución de esta estrategia son clave, ya que la relación riesgo/recompensa es un factor determinante en el éxito a largo plazo.</p>

---
<h2>Tu Opinión Importa: ¡Participa!</h2>
<p>¿Qué piensas sobre <strong>{data['NOMBRE_EMPRESA']} ({data['TICKER']})</strong> tras este análisis? Tu perspectiva es valiosa para nuestra comunidad.</p>
<p><strong>¿Considerarías comprar acciones de {data['NOMBRE_EMPRESA']} con este análisis?</strong></p>
<p>
    <input type="radio" id="vote1" name="opinion" value="si">
    <label for="vote1">Sí, la oportunidad es clara.</label><br>
    <input type="radio" id="vote2" name="opinion" value="no">
    <label for="vote2">No, prefiero esperar más datos.</label><br>
    <input type="radio" id="vote3" name="opinion" value="tengo">
    <label for="vote3">Ya las tengo en cartera.</label>
</p>
<p>¡Déjanos tu voto y tu comentario sobre tu visión de {data['NOMBRE_EMPRESA']}! Queremos saber qué piensas.</p>

---
<h2>¿Qué analizaremos mañana? ¡No te lo pierdas!</h2>
<p>Mañana, pondremos bajo la lupa a <strong>{next_day_company_dynamic}</strong>. ¿Será el próximo candidato para una oportunidad de compra o venta? ¡Vuelve mañana a la misma hora para descubrirlo y seguir ampliando tu conocimiento del mercado!</p>

---
<h2>Conclusión General y Descargo de Responsabilidad</h2>
<p>Para cerrar este análisis de <strong>{data['NOMBRE_EMPRESA']}</strong>, resumo mi visión actual basada en datos técnicos, financieros y estratégicos. [Aquí el modelo redactará un resumen fluido de unas 100 palabras, reforzando la opinión general y la coherencia entre recomendación, niveles técnicos y fundamentos].</p>

<p>Descargo de responsabilidad: Este contenido tiene una finalidad exclusivamente informativa. No constituye una recomendación de inversión. Se recomienda analizar cada decisión de forma individual, teniendo en cuenta el perfil de riesgo y los objetivos financieros personales.</p>

"""
    # The `asunto_email` for the email subject will remain static and descriptive
    asunto_email_subject = f"Análisis: {data['NOMBRE_EMPRESA']} ({data['TICKER']}) - {data['RECOMENDACION']}"

    return prompt, asunto_email_subject


def enviar_email(texto_generado, asunto_email):
    """
    Envía un correo electrónico con el contenido generado.
    Las credenciales de remitente y destinatario están codificadas aquí,
    lo que puede no ser ideal para un entorno de producción seguro.
    """
    remitente = "xumkox@gmail.com"
    destinatario = "xumkox@gmail.com"
    password = "kdgz lvdo wqvt vfkt"  

    msg = MIMEMultipart()
    msg['From'] = remitente
    msg['To'] = destinatario
    msg['Subject'] = asunto_email

    msg.attach(MIMEText(texto_generado, 'html'))  

    try:
        servidor = smtplib.SMTP('smtp.gmail.com', 587)
        servidor.starttls()
        servidor.login(remitente, password)
        servidor.sendmail(remitente, destinatario, msg.as_string())
        servidor.quit()
        print("✅ Correo enviado con éxito.")
    except Exception as e:
        print("❌ Error al enviar el correo:", e)


def generar_contenido_con_gemini(tickers_list):
    """
    Itera sobre una lista de tickers, obtiene sus datos, genera contenido
    con Gemini y envía el análisis por correo electrónico.
    """
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise Exception("No se encontró la variable de entorno GEMINI_API_KEY")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")  

    for i, ticker in enumerate(tickers_list):
        print(f"\n📊 Procesando ticker: {ticker}")
        try:  
            data = obtener_datos_yfinance(ticker)
            if not data:
                continue

            # Obtener los próximos 10 tickers de la lista
            next_tickers_for_prompt = []
            if i + 1 < len(tickers_list):
                next_tickers_for_prompt = tickers_list[i+1 : i+11] # Toma hasta 10 elementos siguientes

            prompt, asunto_email = construir_prompt_formateado(data, next_tickers_for_prompt)

            max_retries = 3
            initial_delay = 10  
            retries = 0
            delay = initial_delay

            while retries < max_retries:
                try:
                    response = model.generate_content(prompt)
                    print(f"\n🧠 Contenido generado para {ticker}:\n")
                    print(response.text)
                    enviar_email(response.text, asunto_email)
                    break  
                except Exception as e:
                    if "429 You exceeded your current quota" in str(e):
                        try:
                            match = re.search(r"retry_delay \{\s*seconds: (\d+)", str(e))
                            if match:
                                server_delay = int(match.group(1))
                                delay = max(delay, server_delay + 1)
                        except:
                            pass
                        
                        print(f"❌ Cuota de Gemini excedida al traducir. Reintentando en {delay} segundos... (Intento {retries + 1}/{max_retries})")
                        time.sleep(delay)
                        retries += 1
                        delay *= 2
                    else:
                        print(f"❌ Error al generar contenido con Gemini (no de cuota): {e}")
                        break
            else:
                print(f"❌ Falló la generación de contenido después de {max_retries} reintentos para {ticker}.")
        except Exception as e:
            print(f"❌ Error general al procesar {ticker}: {e}")


if __name__ == '__main__':
    # Descomenta la línea de abajo para leer tickers de Google Sheets en producción.
    # Asegúrate de que las variables de entorno GOOGLE_APPLICATION_CREDENTIALS y SPREADSHEET_ID estén configuradas.
    # tickers_from_sheet = leer_google_sheets() 
    
    # Para pruebas, usa una lista de tickers de ejemplo. Elimínala o coméntala para producción.
    # He añadido más tickers para que puedas ver el efecto de las 10 siguientes empresas.
    tickers_from_sheet = [
        "MSFT", "GOOGL", "AAPL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "ADBE", "PYPL",
        "CRM", "INTC", "CSCO", "CMCSA", "PEP", "KO", "DIS", "BA", "MCD", "NKE",
        "V", "MA", "JPM", "BAC", "WFC", "XOM", "CVX", "SPG", "PLD", "EQIX"
    ]
    
    if tickers_from_sheet:
        generar_contenido_con_gemini(tickers_from_sheet)
    else:
        print("No se encontraron tickers para procesar.")
