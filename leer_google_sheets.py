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
    
    hist = stock.history(period="60d", interval="1d")  

    if hist.empty:
        print(f"❌ No se pudieron obtener datos históricos para {ticker}")
        return None

    try:
        hist = calculate_smi_tv(hist)
        if 'SMI_signal' not in hist.columns or hist['SMI_signal'].empty:
            print(f"❌ SMI_signal no disponible para {ticker}")
            return None

        smi_actual = round(hist['SMI_signal'].dropna().iloc[-1], 2)
        current_price = round(info.get("currentPrice", 0), 2)
        
        # --- MODIFICACIÓN: Obtener el volumen del último día completo del historial ---
        current_volume = hist['Volume'].iloc[-1] if not hist.empty else 0  
        # --- FIN MODIFICACIÓN ---

        soportes = find_significant_supports(hist, current_price)
        soporte_1 = soportes[0] if len(soportes) > 0 else 0
        soporte_2 = soportes[1] if len(soportes) > 1 else 0
        soporte_3 = soportes[2] if len(soportes) > 2 else 0

        nota_empresa = round((-(max(min(smi_actual, 60), -60)) + 60) * 10 / 120, 1)

        # La recomendación se basa en la nota técnica, que a su vez se basa en el SMI
        if nota_empresa <= 2:
            recomendacion = "Vender"
            condicion_rsi = "muy sobrecomprado" # Se mantiene el texto de "RSI" pero se interpreta como estado técnico
        elif 2 < nota_empresa <= 4:
            recomendacion = "Vigilar posible venta"
            condicion_rsi = "algo sobrecomprado"
        elif 4 < nota_empresa <= 5:
            recomendacion = "Cuidado. Revisar soportes y resistencias"
            condicion_rsi = "muy poca sobrecompra"
        elif 5 < nota_empresa < 6:
            recomendacion = "Mantener (Neutro)"
            condicion_rsi = "neutral"
        elif 6 <= nota_empresa < 7:
            recomendacion = "Posible compra. Revisar soportes y resistencias"
            condicion_rsi = "muy poca sobreventa"
        elif 7 <= nota_empresa < 8:
            recomendacion = "Considerar posible compra"
            condicion_rsi = "algo de sobreventa"
        elif 8 <= nota_empresa < 9:
            recomendacion = "Se acerca la hora de comprar"
            condicion_rsi = "sobreventa"
        elif nota_empresa >= 9:
            recomendacion = "Comprar"
            condicion_rsi = "extremadamente sobrevendido"
        else:
            recomendacion = "Indefinido"
            condicion_rsi = "desconocido"

        precio_objetivo_compra = 0.0
        
        base_precio_obj = soporte_1 if soporte_1 > 0 else current_price * 0.95

        if nota_empresa >= 7:
            precio_objetivo_compra = base_precio_obj
        else:
            drop_percentage_from_base = (7 - nota_empresa) / 7 * 0.15
            precio_objetivo_compra = base_precio_obj * (1 - drop_percentage_from_base)
            
        precio_objetivo_compra = max(0.01, round(precio_objetivo_compra, 2))

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

        # --- Lógica para la tendencia y días estimados ---
        smi_history_full = hist['SMI_signal'].dropna()
        smi_history_last_5 = smi_history_full.tail(5).tolist() # Últimos 5 valores de SMI_signal
        
        tendencia_smi = "No disponible"
        dias_estimados_accion = "No disponible"

        if len(smi_history_last_5) >= 2:
            # Calcular la tendencia
            notas_historicas_last_5 = [round((-(max(min(smi, 60), -60)) + 60) * 10 / 120, 1) for smi in smi_history_last_5]

            if len(notas_historicas_last_5) >= 2:
                # Usar una regresión lineal simple para una estimación más robusta de la tendencia
                x = np.arange(len(notas_historicas_last_5))
                y = np.array(notas_historicas_last_5)
                # Solo si hay suficiente variación para calcular una pendiente significativa
                if len(x) > 1 and np.std(y) > 0.01:
                    slope, intercept = np.polyfit(x, y, 1)
                    tendencia_promedio_diaria = slope
                else: # Si los valores son casi constantes
                    tendencia_promedio_diaria = 0.0
                
                if tendencia_promedio_diaria > 0.1: # umbral pequeño para considerar "mejorando"
                    tendencia_smi = "mejorando"
                elif tendencia_promedio_diaria < -0.1: # umbral pequeño para considerar "empeorando"
                    tendencia_smi = "empeorando"
                else:
                    tendencia_smi = "estable"

                # Estimar días para acción
                target_nota_vender = 2.0
                target_nota_comprar = 8.0

                if nota_empresa <= target_nota_vender:
                    dias_estimados_accion = "Ya en zona de posible venta"
                elif nota_empresa >= target_nota_comprar:
                    dias_estimados_accion = "Ya en zona de posible compra"
                elif tendencia_smi == "estable" or abs(tendencia_promedio_diaria) < 0.01:
                    dias_estimados_accion = "Tendencia estable, sin acción inmediata clara"
                elif tendencia_promedio_diaria < 0: # Nota está bajando, hacia venta
                    diferencia_necesaria = nota_empresa - target_nota_vender
                    if abs(tendencia_promedio_diaria) > 0.01: 
                        dias = diferencia_necesaria / abs(tendencia_promedio_diaria)
                        dias_estimados_accion = f"aprox. {int(max(1, dias))} días para alcanzar zona de venta"
                elif tendencia_promedio_diaria > 0: # Nota está subiendo, hacia compra (o recuperándose de sobreventa)
                    diferencia_necesaria = target_nota_comprar - nota_empresa
                    if abs(tendencia_promedio_diaria) > 0.01:
                        dias = diferencia_necesaria / abs(tendencia_promedio_diaria)
                        dias_estimados_accion = f"aprox. {int(max(1, dias))} días para alcanzar zona de compra"
        # --- Fin de la lógica para la tendencia y días estimados ---


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
            "TENDENCIA_NOTA": tendencia_smi, # Nuevo campo
            "DIAS_ESTIMADOS_ACCION": dias_estimados_accion # Nuevo campo
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
    
    # Asegurarse de que soportes_unicos tenga al menos un elemento para la tabla
    if not soportes_unicos:
        soportes_unicos.append(0) # Valor por defecto si no se encontraron soportes

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

    # Construcción de la tabla de resumen de puntos clave
    tabla_resumen = f"""
<h2>Resumen de Puntos Clave</h2>
<table border="1" style="width:100%; border-collapse: collapse;">
    <tr>
        <th style="padding: 8px; text-align: left; background-color: #f2f2f2;">Métrica</th>
        <th style="padding: 8px; text-align: left; background-color: #f2f2f2;">Valor</th>
    </tr>
    <tr>
        <td style="padding: 8px;">Precio Actual</td>
        <td style="padding: 8px;"><strong>{data['PRECIO_ACTUAL']:,} €</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Volumen</td>
        <td style="padding: 8px;"><strong>{data['VOLUMEN']:,} acciones</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Soporte Clave</td>
        <td style="padding: 8px;"><strong>{soportes_unicos[0]:,} €</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Resistencia Clave</td>
        <td style="padding: 8px;"><strong>{data['RESISTENCIA']:,} €</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Recomendación</td>
        <td style="padding: 8px;"><strong>{data['RECOMENDACION']}</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Nota Técnica (0-10)</td>
        <td style="padding: 8px;"><strong>{data['NOTA_EMPRESA']:,}</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Precio Objetivo de Compra</td>
        <td style="padding: 8px;"><strong>{data['PRECIO_OBJETIVO_COMPRA']:,} €</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Tendencia de la Nota</td>
        <td style="padding: 8px;"><strong>{data['TENDENCIA_NOTA']}</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Días Estimados para Acción</td>
        <td style="padding: 8px;"><strong>{data['DIAS_ESTIMADOS_ACCION']}</strong></td>
    </tr>
</table>
<br/>
"""

    # Dinámica del Impulso - Contenido generado dinámicamente
    dinamica_impulso_text = ""
    if data['TENDENCIA_NOTA'] == "mejorando":
        dinamica_impulso_text = f"La tendencia de nuestra nota técnica es actualmente **mejorando**, lo que sugiere un **impulso alcista** en el comportamiento técnico de la acción. Esto indica que los indicadores del gráfico están mostrando una fortaleza creciente. {f'Según esta dinámica, estimo que podríamos estar a {data["DIAS_ESTIMADOS_ACCION"]} para una posible acción de compra.' if 'compra' in data['DIAS_ESTIMADOS_ACCION'] else ''}"
    elif data['TENDENCIA_NOTA'] == "empeorando":
        dinamica_impulso_text = f"La tendencia de nuestra nota técnica es actualmente **empeorando**, lo que sugiere un **impulso bajista** en el comportamiento técnico de la acción. Esto indica que los indicadores del gráfico están mostrando una debilidad creciente. {f'Según esta dinámica, estimo que podríamos estar a {data["DIAS_ESTIMADOS_ACCION"]} para una posible acción de venta.' if 'venta' in data['DIAS_ESTIMADOS_ACCION'] else ''}"
    else: # Estable o "Ya en zona de posible venta/compra"
        if "Ya en zona" in data['DIAS_ESTIMADOS_ACCION']:
            dinamica_impulso_text = f"La nota técnica de la empresa ya se encuentra en una **zona de {('posible compra' if data['NOTA_EMPRESA'] >= 8 else 'posible venta')}**, lo que indica que el mercado ya ha descontado gran parte del movimiento en esa dirección. Esto podría ofrecer una oportunidad {('de entrada inmediata para compra' if data['NOTA_EMPRESA'] >= 8 else 'de salida inmediata para venta')} para el inversor que busque una acción rápida. Si bien la nota es **{data['NOTA_EMPRESA']}**, es crucial vigilar la volatilidad y los eventos externos que puedan alterar el impulso actual."
        else:
            dinamica_impulso_text = f"La tendencia de nuestra nota técnica es actualmente **estable**, lo que sugiere que el comportamiento técnico de la acción se mantiene sin cambios significativos. Esto implica que no se proyecta una acción inminente basada únicamente en este indicador, aunque siempre es importante estar atento a cualquier cambio en el volumen o los niveles de soporte y resistencia."


    # Volumen - Contenido generado dinámicamente
    volumen_analisis_text = ""
    if data['VOLUMEN'] > 0: # Asumiendo que 0 significa "No disponible" o error
        volumen_analisis_text = f"Analizando el volumen de **{data['VOLUMEN']:,} acciones**, este volumen [El modelo debe decidir si es alto/bajo/normal en relación al historial y la tendencia. Por ejemplo: 'es consistente con la fase de acumulación que observo en el gráfico, y refuerza la validez de los niveles de soporte detectados.' o 'es ligeramente inferior al promedio reciente, lo que podría indicar una falta de convicción en el movimiento actual.']. Un incremento del volumen en la ruptura de la resistencia, por ejemplo, sería una señal inequívoca de fuerza para la tendencia alcista que preveo. La consolidación actual en torno a los soportes identificados, combinada con el volumen, sugiere [interpreta la combinación de volumen y soportes, como acumulación de posiciones, debilidad de la venta, etc.]. El hecho de que no haya un volumen explosivo en este momento refuerza la idea de un movimiento gradual y menos arriesgado, en contraste con una rápida subida impulsada por especulación."
    else:
        volumen_analisis_text = "Actualmente, no dispongo de datos de volumen reciente para realizar un análisis en profundidad. Sin embargo, en cualquier estrategia de inversión, el volumen es un indicador crucial que valida los movimientos de precio y la fuerza de las tendencias. Un volumen significativo en rupturas de niveles clave o en cambios de tendencia es una señal potente a tener en cuenta."


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
- Nota de la empresa (0-10): {data['NOTA_EMPRESA']}
- Precio objetivo de compra: {data['PRECIO_OBJETIVO_COMPRA']}€
- Resultados financieros recientes: {data['INGRESOS']}, {data['EBITDA']}, {data['BENEFICIOS']}
- Nivel de deuda y flujo de caja: {data['DEUDA']}, {data['FLUJO_CAJA']}
- Información estratégica: {data['EXPANSION_PLANES']}, {data['ACUERDOS']}
- Sentimiento del mercado: {data['SENTIMIENTO_ANALISTAS']}, {data['TENDENCIA_SOCIAL']}
- Comparativa sectorial: {data['EMPRESAS_SIMILARES']}
- Riesgos y oportunidades: {data['RIESGOS_OPORTUNIDADES']}
- Tendencia de la nota: {data['TENDENCIA_NOTA']}
- Días estimados para acción: {data['DIAS_ESTIMADOS_ACCION']}

Importante: si algún dato no está disponible ("N/A", "No disponibles", "No disponible"), no lo menciones ni digas que falta. No expliques que la recomendación proviene de un indicador o dato específico. La recomendación debe presentarse como una conclusión personal basada en tu experiencia y criterio profesional como analista. Al redactar el análisis, haz referencia a la **nota obtenida por la empresa ({data['NOTA_EMPRESA']})** en al menos dos de los párrafos principales (Recomendación General, Análisis a Corto Plazo o Predicción a Largo Plazo) como un factor clave para tu valoración.

---
<h1>{titulo_post}</h1>


<h2>Análisis Inicial y Recomendación</h2>
<p>En el dinámico mercado actual, <strong>{data['NOMBRE_EMPRESA']} ({data['TICKER']})</strong> está enviando señales claras de un potencial giro. ¿Es este el momento ideal para considerar una entrada o salida? Mi análisis técnico apunta a que sí, con una oportunidad {('de compra inminente y un rebote en el horizonte' if data['NOTA_EMPRESA'] >= 7 else 'de venta potencial o de esperar una corrección')}.</p>

<p>La empresa cotiza actualmente a <strong>{data['PRECIO_ACTUAL']:,} €</strong>, un nivel que considero estratégico. Mi precio objetivo de compra se sitúa en <strong>{data['PRECIO_OBJETIVO_COMPRA']:,} €</strong>. Este último representa el nivel más atractivo para una entrada conservadora, y aunque el precio actual está {('por encima' if data['PRECIO_ACTUAL'] > data['PRECIO_OBJETIVO_COMPRA'] else 'por debajo')}, aún puede presentar una oportunidad si se evalúa cuidadosamente la relación riesgo/recompensa. Como analista, mi visión es que la convergencia hacia este objetivo podría ser el punto de partida para un movimiento significativo. El volumen negociado recientemente, que alcanzó las <strong>{data['VOLUMEN']:,} acciones</strong>, es un factor clave que valida estos movimientos, y será crucial monitorearlo para confirmar la fuerza de cualquier tendencia emergente.</p>

<p>Asignamos una <strong>nota técnica de {data['NOTA_EMPRESA']} sobre 10</strong>. Esta puntuación refleja [elige una de las siguientes opciones basadas en la nota, manteniendo el foco en el análisis técnico]:
    {"una excelente fortaleza técnica y baja volatilidad esperada a corto plazo. La sólida puntuación se basa en la evaluación de indicadores clave de impulso, soporte y resistencia, lo que indica un bajo riesgo técnico en relación con el potencial de crecimiento a corto plazo." if data['NOTA_EMPRESA'] >= 8 else ""}
    {"una fortaleza técnica moderada, con un equilibrio entre potencial y riesgo. Se basa en el comportamiento del gráfico, soportes, resistencias e impulso, sugiriendo una oportunidad que requiere seguimiento." if 6 <= data['NOTA_EMPRESA'] < 8 else ""}
    {"una situación técnica neutral, donde el gráfico no muestra un patrón direccional claro. La puntuación se deriva del análisis de los movimientos de precio y volumen, indicando que es un momento para la observación y no para la acción inmediata." if 5 <= data['NOTA_EMPRESA'] < 6 else ""}
    {"cierta debilidad técnica, con posibles señales de corrección o continuación bajista. La puntuación se basa en los indicadores del gráfico, que muestran una pérdida de impulso alcista y un aumento de la presión vendedora." if 3 <= data['NOTA_EMPRESA'] < 5 else ""}
    {"una debilidad técnica significativa y una posible sobrecompra en el gráfico, lo que sugiere un alto riesgo de corrección. La puntuación se basa en el análisis de los patrones de precio y volumen, indicando que es un momento para la cautela extrema." if data['NOTA_EMPRESA'] < 3 else ""}
Es importante recordar que esta nota es puramente un reflejo del **análisis del gráfico y sus indicadores técnicos**, y no obedece a la situación financiera o de otro tipo de la empresa. Como profesional, esta nota es mi valoración experta al interpretar el comportamiento del precio y los indicadores.</p>

<h2>Análisis a Corto Plazo: Soportes, Resistencias y Dinámica del Impulso</h2>
<p>Para entender los posibles movimientos a corto plazo en <strong>{data['NOMBRE_EMPRESA']}</strong>, es fundamental analizar el comportamiento reciente del volumen y las zonas clave de soporte y resistencia. Estos niveles no son meros puntos en un gráfico; son reflejos de la psicología del mercado y de puntos donde la oferta y la demanda han encontrado equilibrio o desequilibrio en el pasado, y pueden volver a hacerlo.</p>

<p>En este momento, observo {soportes_texto} La resistencia clave se encuentra en <strong>{data['RESISTENCIA']:,} €</strong>, situada a una distancia del <strong>{((float(data['RESISTENCIA']) - float(data['PRECIO_ACTUAL'])) / float(data['PRECIO_ACTUAL']) * 100):.2f}%</strong> desde el precio actual. Estas zonas técnicas pueden actuar como puntos de inflexión vitales, y su cercanía o lejanía tiene implicaciones operativas claras. Romper la resistencia implicaría un nuevo camino al alza, mientras que la pérdida de un soporte podría indicar una continuación de la caída. Estoy siguiendo de cerca cómo el precio interactúa con estos niveles.</p>

<h2>Estrategia de Inversión y Gestión de Riesgos</h2>
<p>Un aspecto crucial en el análisis de corto plazo es la dinámica del impulso de la empresa. Mi evaluación profesional indica que la tendencia actual de nuestra nota técnica es **{data['TENDENCIA_NOTA']}**. Esto sugiere {('un rebote inminente, dado que los indicadores muestran una sobreventa extrema, lo que significa que la acción ha sido \'castigada\' en exceso y hay una alta probabilidad de que los compradores tomen el control, impulsando el precio al alza. Esta situación de sobreventa, sumada al impulso alcista subyacente, nos sugiere que estamos ante el inicio de un rebote significativo.' if data['TENDENCIA_NOTA'] == 'mejorando' and data['NOTA_EMPRESA'] < 6 else '')}
{('una potencial continuación bajista, con los indicadores técnicos mostrando una sobrecompra significativa o una pérdida de impulso alcista. Esto sugiere que la acción podría experimentar una corrección. Es un momento para la cautela y la vigilancia de los niveles de soporte.' if data['TENDENCIA_NOTA'] == 'empeorando' and data['NOTA_EMPRESA'] > 4 else '')}
{('una fase de consolidación o lateralidad, donde los indicadores técnicos no muestran una dirección clara. Es un momento para esperar la confirmación de una nueva tendencia antes de tomar decisiones.' if data['TENDENCIA_NOTA'] == 'estable' else '')}
{f" Calculamos que este impulso podría llevarnos a una potencial zona de {('toma de beneficios o venta' if data['NOTA_EMPRESA'] >= 8 else 'entrada o compra')} en aproximadamente **{data['DIAS_ESTIMADOS_ACCION']}**." if "No disponible" not in data['DIAS_ESTIMADOS_ACCION'] and "Ya en zona" not in data['DIAS_ESTIMADOS_ACCION'] else ("La nota ya se encuentra en una zona de acción clara, lo que sugiere una oportunidad {('de compra' if data['NOTA_EMPRESA'] >= 8 else 'de venta')} inmediata, y por tanto, no se estima un plazo de días adicional." if "Ya en zona" in data['DIAS_ESTIMADOS_ACCION'] else "")}</p>

<p>{volumen_analisis_text}</p>

<p>Basado en nuestro análisis, una posible estrategia de entrada sería considerar una compra cerca {f"del soporte de <strong>{soportes_unicos[0]:,} €</strong>" if len(soportes_unicos) > 0 and soportes_unicos[0] > 0 else ""} o, idealmente, en {f"los <strong>{soportes_unicos[1]:,} €</strong>." if len(soportes_unicos) > 1 and soportes_unicos[1] > 0 else "."} Estos niveles ofrecen una relación riesgo/recompensa atractiva, permitiendo una entrada con mayor margen de seguridad. Para gestionar el riesgo de forma efectiva, se recomienda establecer un stop loss ajustado justo por debajo del soporte más bajo que hemos identificado, por ejemplo, en {f"<strong>{soportes_unicos[-1]:,} €</strong>." if len(soportes_unicos) > 0 and soportes_unicos[-1] > 0 else "un nivel apropiado de invalidación."} Este punto actuaría como un nivel de invalidez de nuestra tesis de inversión. Nuestro objetivo de beneficio (Take Profit) a corto plazo se sitúa en la resistencia clave de <strong>{data['RESISTENCIA']:,} €</strong>, lo que representa un potencial de revalorización significativo. Esta configuración de entrada, stop loss y objetivo permite una relación riesgo/recompensa favorable para el inversor, buscando maximizar el beneficio mientras se protege el capital.</p>


<h2>Visión a Largo Plazo y Fundamentales</h2>
<p>En un enfoque a largo plazo, el análisis se vuelve más robusto y se apoya en los fundamentos reales del negocio. Aquí, la evolución de <strong>{data['NOMBRE_EMPRESA']}</strong> dependerá en gran parte de sus cifras estructurales y sus perspectivas estratégicas.</p>

<p>En el último ejercicio, los ingresos declarados fueron de <strong>{formatear_numero(data['INGRESOS'])}</strong>, el EBITDA alcanzó <strong>{formatear_numero(data['EBITDA'])}</strong>, y los beneficios netos se situaron en torno a <strong>{formatear_numero(data['BENEFICIOS'])}</strong>. 
En cuanto a su posición financiera, la deuda asciende a <strong>{formatear_numero(data['DEUDA'])}</strong>, y el flujo de caja operativo es de <strong>{formatear_numero(data['FLUJO_CAJA'])}</strong>.</p>

<p>[Si 'EXPANSION_PLANES' o 'ACUERDOS' contienen texto relevante y no genérico, sintetízalo y comenta su posible impacto estratégico. Si la información es demasiado breve o indica 'no disponible/no traducible', elabora sobre la importancia general de tales estrategias para el sector de la empresa o para la empresa en sí, sin inventar detalles específicos]. La información disponible sugiere [integra estas cifras con una interpretación crítica. Evita conectar esto directamente con la nota técnica; en su lugar, enfócate en cómo estas cifras impactan la solvencia, crecimiento potencial y estabilidad a largo plazo. Por ejemplo: "una base financiera sólida que respalda su potencial de crecimiento a largo plazo." o "la necesidad de un seguimiento de su gestión de deuda a largo plazo."].</p>

<p>[Aquí el modelo debe elaborar una proyección fundamentada (mínimo 150 palabras) con párrafos de máximo 3 líneas. Debe integrar estas cifras con una interpretación crítica. Evita la nota técnica aquí; concéntrate en cómo los fundamentales impactan la valoración a largo plazo].</p>

<h2>Conclusión General y Descargo de Responsabilidad</h2>
<p>Para cerrar este análisis de <strong>{data['NOMBRE_EMPRESA']}</strong>, resumo mi visión actual basada en una integración de datos técnicos, financieros y estratégicos. Considero que las claras señales técnicas que apuntan a {('un rebote desde una zona de sobreventa extrema, configurando una oportunidad atractiva' if data['NOTA_EMPRESA'] >= 7 else 'una posible corrección, lo que exige cautela')}, junto con [menciona brevemente los aspectos positivos o neutrales de los fundamentales aquí, sin vincularlos a la nota técnica], hacen de esta empresa un activo para mantener bajo estricta vigilancia. La expectativa es que {f"en los próximos {data['DIAS_ESTIMADOS_ACCION']}" if "No disponible" not in data['DIAS_ESTIMADOS_ACCION'] and "Ya en zona" not in data['DIAS_ESTIMADOS_ACCION'] else "en el corto plazo"}, se presente una oportunidad {('de compra con una relación riesgo-recompensa favorable' if data['NOTA_EMPRESA'] >= 7 else 'de observación o de potencial venta, si los indicadores confirman la debilidad')}. Mantendremos una estrecha vigilancia sobre el comportamiento del precio y el volumen para confirmar esta hipótesis.</p>
{tabla_resumen}
<p>Descargo de responsabilidad: Este contenido tiene una finalidad exclusivamente informativa y educativa. No constituye ni debe interpretarse como una recomendación de inversión, asesoramiento financiero o una invitación a comprar o vender ningún activo. La inversión en mercados financieros conlleva riesgos, incluyendo la pérdida total del capital invertido. Se recomienda encarecidamente a cada inversor realizar su propia investigación exhaustiva (due diligence), consultar con un asesor financiero cualificado y analizar cada decisión de forma individual, teniendo en cuenta su perfil de riesgo personal, sus objetivos financieros y su situación económica antes de tomar cualquier decisión de inversión. El rendimiento pasado no es indicativo de resultados futuros.</p>

<h3>¿Qué analizaremos mañana? ¡No te lo pierdas!</h3>
<p>Mañana, pondremos bajo la lupa a otros 10 valores más. ¿Será el próximo candidato para una oportunidad de compra o venta? ¡Vuelve mañana a la misma hora para descubrirlo y seguir ampliando tu conocimiento de mercado!</p>

<h3>Tu Opinión Importa: ¡Participa!</h3>
<p>¿Considerarías comprar acciones de <strong>{data['NOMBRE_EMPRESA']} ({data['TICKER']})</strong> con este análisis?</p>
<ul>
    <li>Sí, la oportunidad es clara.</li>
    <li>No, prefiero esperar más datos.</li>
    <li>Ya las tengo en cartera.</li>
</ul>
<p>¡Déjanos tu voto y tu comentario sobre tu visión de <strong>{data['NOMBRE_EMPRESA']}</strong> en la sección de comentarios! Queremos saber qué piensas y fomentar una comunidad de inversores informada.</p>
"""

    return prompt, titulo_post


def enviar_email(texto_generado, asunto_email):
    remitente = "xumkox@gmail.com"
    destinatario = "xumkox@gmail.com"
    password = "kdgz lvdo wqvt vfkt"  # ¡RECORDATORIO! Considera usar variables de entorno para la contraseña por seguridad

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
            
        # --- PAUSA DE 1 MINUTO DESPUÉS DE CADA TICKER ---
        print(f"⏳ Esperando 60 segundos antes de procesar el siguiente ticker...")
        time.sleep(60) # Pausa de 60 segundos entre cada ticker

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
