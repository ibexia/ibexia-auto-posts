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
        print('No se encontraron datos.')
    else:
        print('Datos leídos de la hoja:')
        for row in values:
            print(row)

    return [row[0] for row in values if row]


length_k = 10
length_d = 3
ema_signal_len = 10
smooth_period = 5

def calculate_smi_tv(df):
    high = pd.Series(df['High'])
    low = pd.Series(df['Low'])
    close = pd.Series(df['Close'])

    # Ensure enough data for rolling windows, plus 2 for diff and then EMA
    min_required_data = max(length_k, length_d, ema_signal_len, smooth_period) + 2 # Add buffer for initial NaNs from rolling/ewm
    if len(df) < min_required_data:
        print(f"Advertencia: No hay suficientes datos ({len(df)}) para calcular SMI para este período. Se requieren al menos {min_required_data}.")
        df['SMI'] = np.nan
        df['SMI_signal'] = np.nan
        return df

    hh = high.rolling(window=length_k).max()
    ll = low.rolling(window=length_k).min()
    
    diff = hh - ll
    rdiff = close - (hh + ll) / 2

    # Calculate EWMA.
    avgrel = rdiff.ewm(span=length_d, adjust=False).mean()
    avgdiff = diff.ewm(span=length_d, adjust=False).mean()

    smi_raw = pd.Series(np.nan, index=df.index)

    # --- ENHANCED DIVISION BY ZERO HANDLING ---
    # Create a boolean mask where avgdiff is problematic (NaN or very close to zero)
    problematic_avgdiff_mask = avgdiff.isna() | (avgdiff.abs() < 1e-9)

    # Use np.where to conditionally calculate smi_raw
    # If problematic_avgdiff_mask is True, smi_raw is NaN. Otherwise, perform the division.
    smi_raw = np.where(
        problematic_avgdiff_mask,
        np.nan,
        (avgrel / (avgdiff / 2)) * 100
    )
    
    # Ensure smi_raw is a pandas Series with the correct index
    smi_raw = pd.Series(smi_raw, index=df.index)

    # Fill remaining NaNs using interpolation for smoother transitions, then bfill/ffill for edges
    smi_raw = smi_raw.interpolate(method='linear', limit_direction='both').fillna(method='bfill').fillna(method='ffill')

    # If smi_raw is still all NaN or contains infs after filling, it means SMI could not be meaningfully calculated
    if smi_raw.isna().all() or np.isinf(smi_raw).all():
        print("Advertencia: SMI_raw es completamente NaN o inf después de la interpolación. No se puede calcular SMI.")
        df['SMI'] = np.nan
        df['SMI_signal'] = np.nan
        return df

    # Further calculations, ensuring they are not performed on all NaNs
    smi_smoothed = smi_raw.rolling(window=smooth_period).mean()
    smi_signal = smi_smoothed.ewm(span=ema_signal_len, adjust=False).mean()

    # Final check for SMI_signal after all calculations
    if smi_signal.isna().all():
        print("Advertencia: SMI_signal es completamente NaN después de suavizar. No se puede obtener un SMI válido.")
        df['SMI'] = np.nan
        df['SMI_signal'] = np.nan
        return df

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
            potential_supports.append(lows.iloc[i])

    if not potential_supports:
        potential_supports = lows.tolist()
        
    support_zones = {}
    for support in potential_supports:
        found_zone = False
        for zone_level in support_zones.keys():
            if support != 0 and zone_level != 0 and abs(support - zone_level) / support <= tolerance_percent:
                support_zones[zone_level].append(support)
                found_zone = True
                break
        if not found_zone:
            support_zones[support] = [support]
            
    final_supports = []
    for zone_level, values in support_zones.items():
        avg_support = np.mean(values)
        if avg_support < current_price:
            if current_price != 0 and abs(current_price - avg_support) / current_price <= max_deviation_percent:
                final_supports.append({'level': avg_support, 'frequency': len(values)})

    final_supports.sort(key=lambda x: (abs(x['level'] - current_price), -x['frequency']))
    
    top_3_supports = [round(s['level'], 2) for s in final_supports if s['level'] < current_price][:3]
    
    if len(top_3_supports) < 3:
        sorted_lows = sorted([l for l in lows.tolist() if l < current_price], reverse=True)
        for low_val in sorted_lows:
            rounded_low_val = round(low_val, 2)
            if rounded_low_val not in top_3_supports:
                top_3_supports.append(rounded_low_val)
                if len(top_3_supports) == 3:
                    break
    
    while len(top_3_supports) < 3:
        if len(top_3_supports) > 0:
            top_3_supports.append(round(top_3_supports[-1] * 0.95, 2))
        else:
            top_3_supports.append(round(current_price * 0.90, 2))
            
    return top_3_supports

def traducir_texto_con_gemini(text, max_retries=3, initial_delay=5):
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
    stock = yf.Ticker(ticker)
    info = stock.info
    
    hist = stock.history(period="2d", interval="1d")

    if hist.empty or len(hist) < 2:
        print(f"❌ No se pudieron obtener datos históricos suficientes para {ticker} para calcular el volumen del día anterior.")
        hist = stock.history(period="7d", interval="1d")
        if hist.empty or len(hist) < 2:
            print(f"❌ Aún no se pudieron obtener datos históricos suficientes para {ticker} tras reintento.")
            return None

    if len(hist) >= 2:
        current_volume = hist['Volume'].iloc[-2]
    else:
        current_volume = 0 

    hist_long = stock.history(period="90d", interval="1d")
    if hist_long.empty:
        print(f"❌ No se pudieron obtener datos históricos largos para {ticker}")
        return None
    
    if hist_long.empty or not all(col in hist_long.columns for col in ['High', 'Low', 'Close']):
        print(f"❌ Datos históricos incompletos o vacíos para {ticker} para el cálculo del SMI.")
        return None


    try:
        hist_long = calculate_smi_tv(hist_long)
        
        if 'SMI_signal' not in hist_long.columns or hist_long['SMI_signal'].empty or len(hist_long['SMI_signal'].dropna()) < 2:
            print(f"❌ SMI_signal no disponible o insuficiente para {ticker}")
            return None

        smi_actual = hist_long['SMI_signal'].dropna().iloc[-1]
        smi_anterior = hist_long['SMI_signal'].dropna().iloc[-2]
        
        if np.isfinite(smi_actual):
            smi_actual = round(smi_actual, 2)
        else:
            smi_actual = 0.0

        if np.isfinite(smi_anterior):
            smi_anterior = round(smi_anterior, 2)
        else:
            smi_anterior = 0.0

        smi_tendencia = "subiendo" if smi_actual > smi_anterior else "bajando" if smi_actual < smi_anterior else "estable"

        current_price = round(info.get("currentPrice", 0), 2)
        
        soportes = find_significant_supports(hist_long, current_price)
        soporte_1 = soportes[0] if len(soportes) > 0 else 0
        soporte_2 = soportes[1] if len(soportes) > 1 else 0
        soporte_3 = soportes[2] if len(soportes) > 2 else 0

        nota_empresa = round((-(max(min(smi_actual, 60), -60)) + 60) * 10 / 120, 1)

        recomendacion = "Indefinido"
        condicion_rsi = "desconocido"  
        
        if nota_empresa <= 2:
            condicion_rsi = "muy sobrecomprado"
            smi_tendencia = "mostrando un agotamiento alcista."
            if smi_actual > smi_anterior:
                recomendacion = "Sobrecompra extrema. Riesgo inminente de corrección, considerar ventas."
            else:
                recomendacion = "Vender / Tomar ganancias. El impulso indica una corrección en curso."
        elif 2 < nota_empresa <= 4:
            condicion_rsi = "algo sobrecomprado"
            smi_tendencia = "con un impulso alcista que podría estar agotándose."
            if smi_actual > smi_anterior:
                recomendacion = "Atentos a posible sobrecompra. El impulso alcista se está agotando."
            else:
                recomendacion = "Vigilar posible venta. El impulso muestra una disminución."
        elif 4 < nota_empresa <= 5:
            condicion_rsi = "muy poca sobrecompra"
            smi_tendencia = "manteniendo un impulso alcista sólido."
            if smi_actual > smi_anterior:
                recomendacion = "Impulso alcista fuerte. Cuidado con niveles de resistencia."
            else:
                recomendacion = "Tendencia de enfriamiento. Cuidado. Revisar soportes y resistencias."
        elif 5 < nota_empresa < 6:
            condicion_rsi = "neutral"
            smi_tendencia = "en una fase de equilibrio."
            if smi_actual > smi_anterior:
                recomendacion = "Mantener (Neutro). El precio gana impulso."
            else:
                recomendacion = "Mantener (Neutro). El precio busca equilibrio."
        elif 6 <= nota_empresa < 7:
            condicion_rsi = "muy poca sobreventa"
            smi_tendencia = "mostrando señales de recuperación."
            if smi_actual > smi_anterior:
                recomendacion = "Señal de recuperación. Posible compra con confirmación."
            else:
                recomendacion = "El impulso bajista persiste. Considerar cautela."
        elif 7 <= nota_empresa < 8:
            condicion_rsi = "algo de sobreventa"
            smi_tendencia = "en una zona de sobreventa moderada, buscando un rebote."
            if smi_actual > smi_anterior:
                recomendacion = "Considerar posible compra. El impulso muestra un giro al alza."
            else:
                recomendacion = "Sobreventa moderada. Evaluar fortaleza de soportes, el precio podría caer más."
        elif 8 <= nota_empresa < 9:
            condicion_rsi = "sobreventa"
            smi_tendencia = "en una zona de sobreventa fuerte, con potencial de reversión."
            if smi_actual > smi_anterior:
                recomendacion = "Se acerca la hora de comprar. Fuerte señal de rebote."
            else:
                recomendacion = "Sobreventa significativa. Esperar confirmación de rebote antes de comprar."
        elif nota_empresa >= 9:
            condicion_rsi = "extremadamente sobrevendido"
            smi_tendencia = "en una sobreventa extrema, lo que sugiere un rebote inminente."
            if smi_actual > smi_anterior:
                recomendacion = "Comprar. Excelente señal de reversión alcista."
            else:
                recomendacion = "Sobreventa extrema. El precio podría seguir cayendo a corto plazo, esperar confirmación de suelo."

        precio_objetivo_compra = 0.0
        base_precio_obj = soporte_1 if soporte_1 > 0 else current_price * 0.95

        if nota_empresa >= 7 and smi_actual > smi_anterior:
            precio_objetivo_compra = base_precio_obj
        elif nota_empresa >= 7 and smi_actual < smi_anterior:
            precio_objetivo_compra = base_precio_obj * 0.98
        else:
            drop_percentage_from_base = (7 - nota_empresa) / 7 * 0.15
            precio_objetivo_compra = base_precio_obj * (1 - drop_percentage_from_base)
            
        precio_objetivo_compra = max(0.01, round(precio_objetivo_compra, 2))
        
        dias_para_accion_str = "No estimado"
        smi_diff = hist_long['SMI_signal'].dropna().diff().iloc[-1] if len(hist_long['SMI_signal'].dropna()) > 1 else 0
        
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
            "RESISTENCIA": round(hist_long["High"].max(), 2),
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
    try:
        numero = int(valor)
        return f"{numero:,} €"
    except (ValueError, TypeError):
        return "No disponible"
        
def construir_prompt_formateado(data):
    titulo_post = f"{data['RECOMENDACION']} {data['NOMBRE_EMPRESA']} ({data['PRECIO_ACTUAL']}€) {data['TICKER']}"

    soportes_unicos = []
    temp_soportes = sorted([data['SOPORTE_1'], data['SOPORTE_2'], data['SOPORTE_3']], reverse=True)
    
    if len(temp_soportes) > 0 and temp_soportes[0] > 0:
        soportes_unicos.append(temp_soportes[0])
        for i in range(1, len(temp_soportes)):
            if temp_soportes[i] > 0 and soportes_unicos and abs(temp_soportes[i] - soportes_unicos[-1]) / soportes_unicos[-1] > 0.005:
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

    prompt = f"""
Actúa como un trader profesional con amplia experiencia en análisis técnico y mercados financieros. Genera el análisis completo en **formato HTML**, ideal para publicaciones web. Utiliza etiquetas `<h2>` para los títulos de sección y `<p>` para cada párrafo de texto. Redacta en primera persona, con total confianza en tu criterio. 

Destaca los datos importantes como precios, notas de la empresa, cifras financieras y el nombre de la empresa utilizando la etiqueta `<strong>`. Asegúrate de que no haya asteriscos u otros símbolos de marcado en el texto final, solo HTML válido. Asegurate que todo este escrito en español independientemente del idioma de donde saques los datos.

Genera un análisis técnico completo de aproximadamente 1200 palabras sobre la empresa {data['NOMBRE_EMPRESA']}, utilizando los siguientes datos reales extraídos de Yahoo Finance. Presta especial atención a la **nota obtenida por la empresa**: {data['NOTA_EMPRESA']}.

**Datos clave:**
- Precio actual: {data['PRECIO_ACTUAL']}
- Volumen del último día completo: {data['VOLUMEN']}
- Soporte 1: {data['SOPORTE_1']}
- Soporte 2: {data['SOPORTE_2']}
- Soporte 3: {data['SOPORTE_3']}
- Resistencia clave: {data['RESISTENCIA']}
- Recomendación general: {data['RECOMENDACION']}
- Nota de la empresa (0-10): {data['NOTA_EMPRESA']} sobre 10
- Precio objetivo de compra: {data['PRECIO_OBJETIVO_COMPRA']}€
- Resultados financieros recientes: {data['INGRESOS']}, {data['EBITDA']}, {data['BENEFICIOS']}
- Nivel de deuda y flujo de caja: {data['DEUDA']}, {data['FLUJO_CAJA']}
- Información estratégica: {data['EXPANSION_PLANES']}, {data['ACUERDOS']}
- Sentimiento del mercado: {data['SENTIMIENTO_ANALISTAS']}, {data['TENDENCIA_SOCIAL']}
- Comparativa sectorial: {data['EMPRESAS_SIMILARES']}
- Riesgos y oportunidades: {data['RIESGOS_OPORTUNIDADES']}
- La tendencia de impulso actual de la empresa se caracteriza por: {data['SMI_TENDENCIA']}.
- Mi estimación para una potencial zona de acción significativa (compra o venta) indica que: {data['DIAS_PARA_ACCION']}.

Importante: si algún dato no está disponible ("N/A", "No disponibles", "No disponible"), no lo menciones ni digas que falta. No expliques que la recomendación proviene de un indicador o dato específico. La recomendación debe presentarse como una conclusión personal basada en tu experiencia y criterio profesional como un analista.

---
<h1>{titulo_post}</h1>

<h2>Análisis Inicial y Recomendación</h2>
<p>Comienzo el análisis de <strong>{data['NOMBRE_EMPRESA']}</strong> destacando mi recomendación principal: <strong>{data['RECOMENDACION']}</strong>.</p>

<p>La empresa se encuentra en una situación clave. Cotiza actualmente a <strong>{data['PRECIO_ACTUAL']:,} €</strong>. Mi precio objetivo de compra se sitúa en <strong>{data['PRECIO_OBJETIVO_COMPRA']:,} €</strong>. [Si el precio actual es superior al precio objetivo de compra, explica que este último representa el nivel más atractivo para una entrada conservadora, y que el precio actual, aunque por encima, aún puede presentar una oportunidad si se evalúa el riesgo/recompensa. Si es inferior, recalca la oportunidad de compra al estar por debajo del objetivo]. El volumen negociado recientemente alcanza las <strong>{data['VOLUMEN']:,} acciones</strong>.</p>

<p>Asignamos una <strong>nota técnica de {data['NOTA_EMPRESA']} sobre 10</strong>. Esta puntuación refleja [explica concisamente qué significa esa puntuación en términos de riesgo, potencial de crecimiento, y la solidez *técnica* de la compañía para el corto plazo]. A continuación, detallo una visión más completa de mi evaluación profesional, desarrollada en base a una combinación de indicadores técnicos y fundamentos económicos.</p>

<h2>Análisis a Corto Plazo: Soportes y Resistencias y Dinámica del Impulso</h2>
<p>Para entender los posibles movimientos a corto plazo en <strong>{data['NOMBRE_EMPRESA']}</strong>, es fundamental analizar el comportamiento reciente del volumen y las zonas clave de soporte y resistencia.</p>

<p>En este momento, observo {soportes_texto} La resistencia clave se encuentra en <strong>{data['RESISTENCIA']:,} €</strong>, situada a una distancia del <strong>{((float(data['RESISTENCIA']) - float(data['PRECIO_ACTUAL'])) / float(data['PRECIO_ACTUAL']) * 100):.2f}%</strong> desde el precio actual. Estas zonas técnicas pueden actuar como puntos de inflexión, y su cercanía o lejanía tiene implicaciones operativas claras.</p>

<p>Un aspecto crucial en el análisis de corto plazo es la dinámica de impulso de la empresa. Mi evaluación profesional indica que la tendencia actual se caracteriza por: <strong>{data['SMI_TENDENCIA']}</strong>. En este contexto, {data['DIAS_PARA_ACCION']} Analizando el volumen de <strong>{data['VOLUMEN']:,} acciones</strong>, [compara el volumen actual con el volumen promedio reciente (si está disponible implícitamente en los datos que procesa el modelo) o con el volumen histórico en puntos de inflexión. Comenta si el volumen actual es 'saludable', 'bajo', 'elevado' o 'anormal' para confirmar la validez de los movimientos de precio en los soportes y resistencias]. Estos niveles técnicos y el patrón de volumen, junto con la nota técnica de <strong>{data['NOTA_EMPRESA']} sobre 10</strong>, nos proporcionan una guía para la operativa a corto plazo. [Aquí el modelo desarrollará un análisis de mínimo 150 palabras, con lectura segmentada, mencionando cómo estos niveles influyen en la operativa a corto plazo. La nota técnica debe ser un factor clave aquí].</p>

<h2>Visión a Largo Plazo y Fundamentales</h2>
<p>En un enfoque a largo plazo, el análisis se vuelve más robusto y se apoya en los fundamentos reales del negocio. Aquí, la evolución de <strong>{data['NOMBRE_EMPRESA']}</strong> dependerá en gran parte de sus cifras estructurales y sus perspectivas estratégicas. Para esta sección, la **nota técnica ({data['NOTA_EMPRESA']} sobre 10) NO debe influir en la valoración**. El análisis debe basarse **exclusivamente en los datos financieros y estratégicos** proporcionados y en una evaluación crítica de su solidez y potencial.</p>

<p>En el último ejercicio, los ingresos declarados fueron de <strong>{formatear_numero(data['INGRESOS'])}</strong>, el EBITDA alcanzó <strong>{formatear_numero(data['EBITDA'])}</strong>, y los beneficios netos se situaron en torno a <strong>{formatear_numero(data['BENEFICIOS'])}
