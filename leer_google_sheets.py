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
    sheet = service.values()
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
    high = df['High']
    low = df['Low']
    close = df['Close']

    hh = high.rolling(window=length_k).max()
    ll = low.rolling(window=length_k).min()
    diff = hh - ll
    rdiff = close - (hh + ll) / 2

    avgrel = rdiff.ewm(span=length_d, adjust=False).mean()
    avgdiff = diff.ewm(span=length_d, adjust=False).mean()

    smi_raw = (avgrel / (avgdiff / 2)) * 100
    smi_raw[avgdiff == 0] = 0.0

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
            potential_supports.append(lows.iloc[i])

    if not potential_supports:
        potential_supports = lows.tolist()
        
    support_zones = {}
    for support in potential_supports:
        found_zone = False
        for zone_level in support_zones.keys():
            if abs(support - zone_level) / support <= tolerance_percent:
                support_zones[zone_level].append(support)
                found_zone = True
                break
        if not found_zone:
            support_zones[support] = [support]
            
    final_supports = []
    for zone_level, values in support_zones.items():
        avg_support = np.mean(values)
        if avg_support < current_price:
            if abs(current_price - avg_support) / current_price <= max_deviation_percent:
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
    
    hist = stock.history(period="90d", interval="1d")  

    if hist.empty:
        print(f"❌ No se pudieron obtener datos históricos para {ticker}")
        return None

    try:
        hist = calculate_smi_tv(hist)
        
        if 'SMI_signal' not in hist.columns or hist['SMI_signal'].empty or len(hist['SMI_signal'].dropna()) < 2:
            print(f"❌ SMI_signal no disponible o insuficiente para {ticker}")
            return None

        smi_actual = round(hist['SMI_signal'].dropna().iloc[-1], 2)
        smi_anterior = round(hist['SMI_signal'].dropna().iloc[-2], 2) # SMI del día anterior
        
        smi_tendencia = "subiendo" if smi_actual > smi_anterior else "bajando" if smi_actual < smi_anterior else "estable"

        current_price = round(info.get("currentPrice", 0), 2)
        
        current_volume = hist['Volume'].iloc[-1] if not hist.empty else 0  

        soportes = find_significant_supports(hist, current_price)
        soporte_1 = soportes[0] if len(soportes) > 0 else 0
        soporte_2 = soportes[1] if len(soportes) > 1 else 0
        soporte_3 = soportes[2] if len(soportes) > 2 else 0

        nota_empresa = round((-(max(min(smi_actual, 60), -60)) + 60) * 10 / 120, 1)

        recomendacion = "Indefinido"
        condicion_rsi = "desconocido" 
        
        # --- LÓGICA DE RECOMENDACIÓN CORREGIDA ---
        if nota_empresa <= 2: # SMI muy alto (sobrecompra fuerte: SMI entre 60 y 100)
            condicion_rsi = "muy sobrecomprado"
            if smi_tendencia == "subiendo":
                recomendacion = "Sobrecompra extrema. Riesgo inminente de corrección, considerar ventas."
            else: # smi_tendencia == "bajando" o "estable"
                recomendacion = "Vender / Tomar ganancias. El impulso indica una corrección en curso."
        elif 2 < nota_empresa <= 4: # SMI alto (sobrecompra moderada: SMI entre 30 y 60)
            condicion_rsi = "algo sobrecomprado"
            if smi_tendencia == "subiendo":
                recomendacion = "Atentos a posible sobrecompra. El impulso alcista se está agotando."
            else: # smi_tendencia == "bajando" o "estable"
                recomendacion = "Vigilar posible venta. El impulso muestra una disminución."
        elif 4 < nota_empresa <= 5: # SMI ligeramente sobrecomprado / entrando en zona neutra (SMI entre 10 y 30)
            condicion_rsi = "muy poca sobrecompra"
            if smi_tendencia == "subiendo":
                recomendacion = "Impulso alcista fuerte. Cuidado con niveles de resistencia."
            else: # smi_tendencia == "bajando" o "estable"
                recomendacion = "Tendencia de enfriamiento. Cuidado. Revisar soportes y resistencias."
        elif 5 < nota_empresa < 6: # Zona neutra (SMI entre -10 y 10)
            condicion_rsi = "neutral"
            if smi_tendencia == "subiendo":
                recomendacion = "Mantener (Neutro). El precio gana impulso."
            else: # smi_tendencia == "bajando" o "estable"
                recomendacion = "Mantener (Neutro). El precio busca equilibrio."
        elif 6 <= nota_empresa < 7: # SMI ligeramente sobrevendido / entrando en zona neutra (SMI entre -30 y -10)
            condicion_rsi = "muy poca sobreventa"
            if smi_tendencia == "subiendo":
                recomendacion = "Señal de recuperación. Posible compra con confirmación."
            else: # smi_tendencia == "bajando" o "estable"
                recomendacion = "El impulso bajista persiste. Considerar cautela."
        elif 7 <= nota_empresa < 8: # SMI bajo (sobreventa moderada: SMI entre -60 y -30)
            condicion_rsi = "algo de sobreventa"
            if smi_tendencia == "subiendo":
                recomendacion = "Considerar posible compra. El impulso muestra un giro al alza."
            else: # smi_tendencia == "bajando" o "estable"
                recomendacion = "Sobreventa moderada. Evaluar fortaleza de soportes, el precio podría caer más."
        elif 8 <= nota_empresa < 9: # SMI muy bajo (sobreventa fuerte: SMI entre -100 y -60)
            condicion_rsi = "sobreventa"
            if smi_tendencia == "subiendo":
                recomendacion = "Se acerca la hora de comprar. Fuerte señal de rebote."
            else: # smi_tendencia == "bajando" o "estable"
                recomendacion = "Sobreventa significativa. Esperar confirmación de rebote antes de comprar."
        elif nota_empresa >= 9: # SMI extremadamente bajo (sobreventa extrema: SMI muy por debajo de -60)
            condicion_rsi = "extremadamente sobrevendido"
            if smi_tendencia == "subiendo":
                recomendacion = "Comprar. Excelente señal de reversión alcista."
            else: # smi_tendencia == "bajando" o "estable"
                recomendacion = "Sobreventa extrema. El precio podría seguir cayendo a corto plazo, esperar confirmación de suelo."

        precio_objetivo_compra = 0.0
        base_precio_obj = soporte_1 if soporte_1 > 0 else current_price * 0.95

        # Ajuste del precio objetivo de compra basado en la nota y la tendencia del SMI
        if nota_empresa >= 7 and smi_tendencia == "subiendo": # Cuando está sobrevendido Y el SMI sube -> buena señal para comprar
            precio_objetivo_compra = base_precio_obj
        elif nota_empresa >= 7 and smi_tendencia == "bajando": # Cuando está sobrevendido PERO el SMI sigue bajando -> más cautela
            precio_objetivo_compra = base_precio_obj * 0.98 # Un 2% más abajo
        else: # Para notas más bajas, ajustamos el porcentaje de caída desde la base
            drop_percentage_from_base = (7 - nota_empresa) / 7 * 0.15
            precio_objetivo_compra = base_precio_obj * (1 - drop_percentage_from_base)
            
        precio_objetivo_compra = max(0.01, round(precio_objetivo_compra, 2))
        
        ### --- INICIO DE LA MODIFICACIÓN PARA CÁLCULO DE DÍAS (SMI NO REVELADO) ---
        dias_para_accion = "No estimado"
        # Usamos SMI_signal para el cálculo interno, pero la salida al prompt no lo nombrará
        smi_diff = hist['SMI_signal'].dropna().diff().iloc[-1] if len(hist['SMI_signal'].dropna()) > 1 else 0

        # Definimos umbrales de "zona de acción" para el SMI de forma genérica
        # Estos son los niveles que el SMI_signal debería alcanzar para una acción clara
        target_smi_venta_zona = 60 
        target_smi_compra_zona = -60

        if smi_tendencia == "subiendo" and smi_actual < target_smi_venta_zona and smi_diff > 0.01: # El SMI sube y se acerca a sobrecompra
            diferencia_a_target = target_smi_venta_zona - smi_actual
            dias_para_accion = f"aproximadamente {int(diferencia_a_target / smi_diff)} días para una potencial zona de toma de beneficios o venta"
        elif smi_tendencia == "bajando" and smi_actual > target_smi_compra_zona and smi_diff < -0.01: # El SMI baja y se acerca a sobreventa
            diferencia_a_target = smi_actual - target_smi_compra_zona
            dias_para_accion = f"aproximadamente {int(diferencia_a_target / abs(smi_diff))} días para una potencial zona de entrada o compra"
        ### --- FIN DE LA MODIFICACIÓN PARA CÁLCULO DE DÍAS ---

        # --- Aplicar traducción a los campos relevantes aquí ---
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
        
        # --- Fin de la traducción ---


        datos = {
            "TICKER": ticker,
            "NOMBRE_EMPRESA": info.get("longName", ticker),
            "PRECIO_ACTUAL": current_price,
            "VOLUMEN": current_volume,  
            "SOPORTE_1": soporte_1,
            "SOPORTE_2": soporte_2,
            "SOPORTE_3": soporte_3,
            "RESISTENCIA": round(hist["High"].max(), 2),
            "CONDICION_RSI": condicion_rsi, # Este dato no se usará directamente en el prompt
            "RECOMENDACION": recomendacion,
            "SMI": smi_actual, # Este dato no se usará directamente en el prompt
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
            "SMI_TENDENCIA": smi_tendencia, # Se mantiene para el cálculo, pero no se menciona directamente en el prompt
            "DIAS_PARA_ACCION": dias_para_accion 
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

    # Pre-procesamiento de soportes para agruparlos si son muy cercanos
    soportes_unicos = []
    temp_soportes = sorted([data['SOPORTE_1'], data['SOPORTE_2'], data['SOPORTE_3']], reverse=True)
    
    if len(temp_soportes) > 0 and temp_soportes[0] > 0:
        soportes_unicos.append(temp_soportes[0])
        for i in range(1, len(temp_soportes)):
            if temp_soportes[i] > 0 and abs(temp_soportes[i] - soportes_unicos[-1]) / soportes_unicos[-1] > 0.005: # Tolerancia del 0.5%
                soportes_unicos.append(temp_soportes[i])
    
    # Construcción del texto de soportes
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

    ### --- INICIO DE LA MODIFICACIÓN DEL PROMPT (AJUSTE Y CORRECCIÓN DE SINTAXIS) ---
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
- Mi pronóstico de la tendencia de impulso actual de la empresa es: {data['SMI_TENDENCIA']}.
- Mi estimación de días para un punto de acción significativo (compra o venta) es: {data['DIAS_PARA_ACCION']}.

Importante: si algún dato no está disponible ("N/A", "No disponibles", "No disponible"), no lo menciones ni digas que falta. No expliques que la recomendación proviene de un indicador o dato específico. La recomendación debe presentarse como una conclusión personal basada en tu experiencia y criterio profesional como analista.

---
<h1>{titulo_post}</h1>

<h2>Análisis Inicial y Recomendación</h2>
<p>Comienzo el análisis de <strong>{data['NOMBRE_EMPRESA']}</strong> destacando mi recomendación principal: <strong>{data['RECOMENDACION']}</strong>.</p>

<p>La empresa se encuentra en una situación clave. Cotiza actualmente a <strong>{data['PRECIO_ACTUAL']:,} €</strong>. Mi precio objetivo de compra se sitúa en <strong>{data['PRECIO_OBJETIVO_COMPRA']:,} €</strong>. [Si el precio actual es superior al precio objetivo de compra, explica que este último representa el nivel más atractivo para una entrada conservadora, y que el precio actual, aunque por encima, aún puede presentar una oportunidad si se evalúa el riesgo/recompensa. Si es inferior, recalca la oportunidad de compra al estar por debajo del objetivo]. El volumen negociado recientemente alcanza las <strong>{data['VOLUMEN']:,} acciones</strong>.</p>

<p>Asignamos una <strong>nota técnica de {data['NOTA_EMPRESA']} sobre 10</strong>. Esta puntuación refleja [explica concisamente qué significa esa puntuación en términos de riesgo, potencial de crecimiento, y la solidez *técnica* de la compañía para el corto plazo]. A continuación, detallo una visión más completa de mi evaluación profesional, desarrollada en base a una combinación de indicadores técnicos y fundamentos económicos.</p>

<h2>Análisis a Corto Plazo: Soportes y Resistencias y Dinámica del Impulso</h2>
<p>Para entender los posibles movimientos a corto plazo en <strong>{data['NOMBRE_EMPRESA']}</strong>, es fundamental analizar el comportamiento reciente del volumen y las zonas clave de soporte y resistencia.</p>

<p>En este momento, observo {soportes_texto} La resistencia clave se encuentra en <strong>{data['RESISTENCIA']:,} €</strong>, situada a una distancia del <strong>{((float(data['RESISTENCIA']) - float(data['PRECIO_ACTUAL'])) / float(data['PRECIO_ACTUAL']) * 100):.2f}%</strong> desde el precio actual. Estas zonas técnicas pueden actuar como puntos de inflexión, y su cercanía o lejanía tiene implicaciones operativas claras.</p>

<p>Un aspecto crucial en el análisis de corto plazo es la dinámica de impulso de la empresa. Mi evaluación profesional indica que la tendencia actual es <strong>{data['SMI_TENDENCIA']}</strong>. {f"Manteniendo esta dirección, mi pronóstico sugiere que podríamos alcanzar una <strong>potencial zona de acción</strong> (sea de compra o venta) en <strong>{data['DIAS_PARA_ACCION']}</strong>. Esto indica que la presión {'alcista' if data['SMI_TENDENCIA'] == 'subiendo' else 'bajista'} podría continuar y es prudente monitorear los niveles clave muy de cerca en los próximos días." if data['DIAS_PARA_ACCION'] != 'No estimado' else ''} Analizando el volumen de <strong>{data['VOLUMEN']:,} acciones</strong>, [compara el volumen actual con el volumen promedio reciente (si está disponible implícitamente en los datos que procesa el modelo) o con el volumen histórico en puntos de inflexión. Comenta si el volumen actual es 'saludable', 'bajo', 'elevado' o 'anormal' para confirmar la validez de los movimientos de precio en los soportes y resistencias]. Estos niveles técnicos y el patrón de volumen, junto con la nota técnica de <strong>{data['NOTA_EMPRESA']} sobre 10</strong>, nos proporcionan una guía para la operativa a corto plazo. [Aquí el modelo desarrollará un análisis de mínimo 150 palabras, con lectura segmentada, mencionando cómo estos niveles influyen en la operativa a corto plazo. La nota técnica debe ser un factor clave aquí].</p>

<h2>Visión a Largo Plazo y Fundamentales</h2>
<p>En un enfoque a largo plazo, el análisis se vuelve más robusto y se apoya en los fundamentos reales del negocio. Aquí, la evolución de <strong>{data['NOMBRE_EMPRESA']}</strong> dependerá en gran parte de sus cifras estructurales y sus perspectivas estratégicas. Para esta sección, la **nota técnica ({data['NOTA_EMPRESA']} sobre 10) NO debe influir en la valoración**. El análisis debe basarse **exclusivamente en los datos financieros y estratégicos** proporcionados y en una evaluación crítica de su solidez y potencial.</p>

<p>En el último ejercicio, los ingresos declarados fueron de <strong>{formatear_numero(data['INGRESOS'])}</strong>, el EBITDA alcanzó <strong>{formatear_numero(data['EBITDA'])}</strong>, y los beneficios netos se situaron en torno a <strong>{formatear_numero(data['BENEFICIOS'])}</strong>. 
En cuanto a su posición financiera, la deuda asciende a <strong>{formatear_numero(data['DEUDA'])}</strong>, y el flujo de caja operativo es de <strong>{formatear_numero(data['FLUJO_CAJA'])}</strong>.</p>

<p>[Si 'EXPANSION_PLANES' o 'ACUERDOS' contienen texto relevante y no genérico, sintetízalo y comenta su posible impacto estratégico. Si la información es demasiado breve o indica 'no disponible/no traducible', elabora sobre la importancia general de tales estrategias para el sector de la empresa o para la empresa en sí, sin inventar detalles específicos]. [Aquí el modelo debe elaborar una proyección fundamentada (mínimo 150 palabras) con párrafos de máximo 3 líneas. Debe integrar estas cifras con una interpretación crítica de la solvencia, rentabilidad, crecimiento y las perspectivas estratégicas de la empresa, y **mojarse** con una valoración clara sobre su potencial a largo plazo basada *únicamente* en estos fundamentales].</p>

<h2>Conclusión General y Descargo de Responsabilidad</h2>
<p>Para cerrar este análisis de <strong>{data['NOMBRE_EMPRESA']}</strong>, resumo mi visión actual basada en datos técnicos, financieros y estratégicos. [Aquí el modelo redactará un resumen fluido de unas 100 palabras, reforzando la opinión general y la coherencia entre recomendación, niveles técnicos y fundamentos].</p>

<p>Descargo de responsabilidad: Este contenido tiene una finalidad exclusivamente informativa. No constituye una recomendación de inversión. Se recomienda analizar cada decisión de forma individual, teniendo en cuenta el perfil de riesgo y los objetivos financieros personales.</p>

"""
    ### --- FIN DE LA MODIFICACIÓN DEL PROMPT (AJUSTE Y CORRECCIÓN DE SINTAXIS) ---

    return prompt, titulo_post


def enviar_email(texto_generado, asunto_email):
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


def generar_contenido_con_gemini(tickers):
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise Exception("No se encontró la variable de entorno GEMINI_API_KEY")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")  

    for ticker in tickers:
        print(f"\n📊 Procesando ticker: {ticker}")
        try: 
            data = obtener_datos_yfinance(ticker)
            if not data:
                continue
            prompt, titulo_post = construir_prompt_formateado(data)

            max_retries = 3
            initial_delay = 10  
            retries = 0
            delay = initial_delay

            while retries < max_retries:
                try:
                    response = model.generate_content(prompt)
                    print(f"\n🧠 Contenido generado para {ticker}:\n")
                    print(response.text)
                    asunto_email = f"Análisis: {data['NOMBRE_EMPRESA']} ({data['TICKER']}) - {data['RECOMENDACION']}"
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
                        
                        print(f"❌ Cuota de Gemini excedida al generar contenido. Reintentando en {delay} segundos... (Intento {retries + 1}/{max_retries})")
                        time.sleep(delay)
                        retries += 1
                        delay *= 2
                    else:
                        print(f"❌ Error al generar contenido con Gemini (no de cuota): {e}")
                        break
            else:  
                print(f"❌ Falló la generación de contenido para {ticker} después de {max_retries} reintentos.")
                
        except Exception as e: 
            print(f"❌ Error crítico al procesar el ticker {ticker}: {e}. Saltando a la siguiente empresa.")
            continue 

        print(f"⏳ Esperando 60 segundos antes de procesar el siguiente ticker...")
        time.sleep(60) 

def main():
    all_tickers = leer_google_sheets()[1:]
    
    if not all_tickers:
        print("No hay tickers para procesar.")
        return

    day_of_week = datetime.today().weekday()
    
    num_tickers_per_day = 10  
    total_tickers_in_sheet = len(all_tickers)
    
    start_index = (day_of_week * num_tickers_per_day) % total_tickers_in_sheet
    
    end_index = start_index + num_tickers_per_day
    
    tickers_for_today = []
    if end_index <= total_tickers_in_sheet:
        tickers_for_today = all_tickers[start_index:end_index]
    else:
        tickers_for_today = all_tickers[start_index:] + all_tickers[:end_index - total_tickers_in_sheet]

    if tickers_for_today:
        print(f"Procesando tickers para el día {datetime.today().strftime('%A')}: {tickers_for_today}")
        generar_contenido_con_gemini(tickers_for_today)
    else:
        print(f"No hay tickers disponibles para el día {datetime.today().strftime('%A')} en el rango calculado. "
              f"start_index: {start_index}, end_index: {end_index}, total_tickers: {total_tickers_in_sheet}")


if __name__ == '__main__':
    main()
