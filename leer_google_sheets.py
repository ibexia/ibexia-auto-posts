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
    # CORRECCIÓN: Pasar los argumentos correctos al método get()
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
    recent_data = df.tail(window) # Últimas 'window' velas (ej: 20-50 días)
    lows = recent_data['Low']
    
    potential_supports = []
    
    # Identificar mínimos locales (puntos de rebote)
    # Un mínimo local es un punto más bajo que sus vecinos
    for i in range(1, len(lows) - 1):
        if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
            potential_supports.append(lows.iloc[i])

    if not potential_supports:
        # Si no hay mínimos locales claros, considera los mínimos de la ventana
        potential_supports = lows.tolist()
        
    # Agrupar soportes cercanos en "zonas"
    support_zones = {}
    for support in potential_supports:
        found_zone = False
        for zone_level in support_zones.keys():
            if abs(support - zone_level) / support <= tolerance_percent: # Dentro de la tolerancia
                support_zones[zone_level].append(support)
                found_zone = True
                break
        if not found_zone:
            support_zones[support] = [support]
            
    # Calcular el valor promedio de cada zona y su "frecuencia" (número de toques)
    # Filtra los soportes que están por encima del precio actual o demasiado lejos
    final_supports = []
    for zone_level, values in support_zones.items():
        avg_support = np.mean(values)
        if avg_support < current_price: # Solo soportes por debajo del precio actual
            if abs(current_price - avg_support) / current_price <= max_deviation_percent:
                final_supports.append({'level': avg_support, 'frequency': len(values)})

    # Ordenar por cercanía al precio actual y luego por frecuencia (más toques, más relevante)
    final_supports.sort(key=lambda x: (abs(x['level'] - current_price), -x['frequency']))
    
    # Tomar los 3 soportes más cercanos
    top_3_supports = [round(s['level'], 2) for s in final_supports if s['level'] < current_price][:3]
    
    # Asegurarse de que siempre haya al menos 3 soportes (rellenar con valores de la ventana si es necesario)
    if len(top_3_supports) < 3:
        sorted_lows = sorted([l for l in lows.tolist() if l < current_price], reverse=True)
        for low_val in sorted_lows:
            rounded_low_val = round(low_val, 2)
            if rounded_low_val not in top_3_supports:
                top_3_supports.append(rounded_low_val)
                if len(top_3_supports) == 3:
                    break
    
    # Si aún no hay 3 soportes, usar el mínimo de la ventana o incluso un valor inferior al actual.
    while len(top_3_supports) < 3:
        if len(top_3_supports) > 0:
            top_3_supports.append(round(top_3_supports[-1] * 0.95, 2)) # Un 5% por debajo del último
        else:
            top_3_supports.append(round(current_price * 0.90, 2)) # 10% por debajo del precio actual
            
    return top_3_supports


def obtener_datos_yfinance(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Limitar la ventana de observación para el análisis de soportes (aprox. 1-2 meses de datos diarios)
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
        
        # Calcular los soportes usando la nueva lógica
        soportes = find_significant_supports(hist, current_price)
        soporte_1 = soportes[0] if len(soportes) > 0 else 0
        soporte_2 = soportes[1] if len(soportes) > 1 else 0
        soporte_3 = soportes[2] if len(soportes) > 2 else 0

        # La nota_empresa se mantiene como antes
        nota_empresa = round((-(max(min(smi_actual, 60), -60)) + 60) * 10 / 120, 1)

        if nota_empresa <= 2:
            recomendacion = "Vender"
            condicion_rsi = "muy sobrecomprado"
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

        # Calcular el precio objetivo de compra
        precio_objetivo_compra = 0.0
        
        # Usar el soporte 1 como base, si no existe, usar un porcentaje del precio actual
        base_precio_obj = soporte_1 if soporte_1 > 0 else current_price * 0.95 

        if nota_empresa >= 7:
            # Si la nota es 7 o más, el objetivo de compra es el primer soporte
            precio_objetivo_compra = base_precio_obj
        else:
            # Si la nota es menor que 7, el objetivo de compra es por debajo del soporte,
            # escalando en función de qué tan lejos esté la nota de 7.
            # Se asume una caída máxima del 15% por debajo del soporte para una nota de 0
            drop_percentage_from_base = (7 - nota_empresa) / 7 * 0.15
            precio_objetivo_compra = base_precio_obj * (1 - drop_percentage_from_base)
            
        precio_objetivo_compra = max(0.01, round(precio_objetivo_compra, 2))


        datos = {
            "NOMBRE_EMPRESA": info.get("longName", ticker),
            "PRECIO_ACTUAL": current_price,
            "VOLUMEN": info.get("volume", 0),
            "SOPORTE_1": soporte_1,
            "SOPORTE_2": soporte_2,
            "SOPORTE_3": soporte_3,
            "RESISTENCIA": round(hist["High"].max(), 2), # Resistencia se mantiene como el máximo en la ventana
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
            "EXPANSION_PLANES": info.get("longBusinessSummary", "N/A"),
            "ACUERDOS": "No disponibles",
            "SENTIMIENTO_ANALISTAS": info.get("recommendationKey", "N/A"),
            "TENDENCIA_SOCIAL": "No disponible",
            "EMPRESAS_SIMILARES": ", ".join(info.get("category", "").split(",")) if info.get("category") else "No disponibles",
            "RIESGOS_OPORTUNIDADES": "No disponibles"
        }
    except Exception as e:
        print(f"❌ Error al obtener datos de {ticker}: {e}")
        return None

    return datos


def construir_prompt_formateado(data):
    titulo_post = f"{data['RECOMENDACION']} {data['NOMBRE_EMPRESA']} ({data['PRECIO_ACTUAL']}€)"

    prompt = f"""
Actúa como un trader profesional con amplia experiencia en análisis técnico y mercados financieros. Genera el análisis completo en **formato HTML**, ideal para publicaciones web. Utiliza etiquetas `<h2>` para los títulos de sección y `<p>` para cada párrafo de texto. Redacta en primera persona, con total confianza en tu criterio. 

Destaca los datos importantes como precios, notas de la empresa, cifras financieras y el nombre de la empresa utilizando la etiqueta `<strong>`. Asegúrate de que no haya asteriscos u otros símbolos de marcado en el texto final, solo HTML válido.

Genera un análisis técnico completo de aproximadamente 1000 palabras sobre la empresa {data['NOMBRE_EMPRESA']}, utilizando los siguientes datos reales extraídos de Yahoo Finance. Presta especial atención a la **nota obtenida por la empresa**: {data['NOTA_EMPRESA']}.

**Datos clave:**
- Precio actual: {data['PRECIO_ACTUAL']}
- Volumen: {data['VOLUMEN']}
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

Importante: si algún dato no está disponible, no lo menciones ni digas que falta. No expliques que la recomendación proviene de un indicador o dato específico. La recomendación debe presentarse como una conclusión personal basada en tu experiencia y criterio profesional como analista. Al redactar el análisis, haz referencia a la **nota obtenida por la empresa ({data['NOTA_EMPRESA']})** en al menos dos de los párrafos principales (Recomendación General, Análisis a Corto Plazo o Predicción a Largo Plazo) como un factor clave para tu valoración.

---
<h1>{titulo_post}</h1>

<h2>Análisis Inicial y Recomendación</h2>
<p>Para comenzar el análisis de <strong>{data['NOMBRE_EMPRESA']}</strong>, quiero dejar clara mi recomendación principal: <strong>{data['RECOMENDACION']}</strong>. Este juicio se fundamenta en un análisis exhaustivo de su situación actual, donde la <strong>nota de {data['NOTA_EMPRESA']}</strong> juega un papel crucial. La empresa se encuentra en un punto estratégico en el mercado, con un precio actual de <strong>{data['PRECIO_ACTUAL']}€</strong> y un <strong>precio objetivo de compra de {data['PRECIO_OBJETIVO_COMPRA']}€</strong>, con un volumen de <strong>{data['VOLUMEN']}</strong>.</p>
<p>Como recomendación general, mi opinión profesional sobre la situación actual de <strong>{data['NOMBRE_EMPRESA']}</strong> y sus perspectivas es la siguiente: [Aquí el modelo expandirá la recomendación, mínimo 150 palabras, usando un enfoque técnico y financiero combinado. Mencionará la nota de <strong>{data['NOTA_EMPRESA']}</strong> como factor determinante].</p>

<h2>Análisis a Corto Plazo: Soportes y Resistencias</h2>
<p>En el análisis a corto plazo, considero los posibles movimientos del precio en el horizonte inmediato. [Aquí el modelo describirá movimientos, volumen, los soportes clave son <strong>{data['SOPORTE_1']}€</strong>, <strong>{data['SOPORTE_2']}€</strong> y <strong>{data['SOPORTE_3']}€</strong>, y la resistencia en <strong>{data['RESISTENCIA']}€</strong>, mínimo 150 palabras. Hará referencia a la nota de <strong>{data['NOTA_EMPRESA']}</strong> si lo considera relevante para el corto plazo].</p>

<h2>Visión a Largo Plazo y Fundamentales</h2>
<p>Respecto a la predicción a largo plazo, mi visión para el futuro de la empresa incluye... [Aquí el modelo desarrollará la visión a futuro, análisis financiero (ingresos: <strong>{data['INGRESOS']}</strong>, EBITDA: <strong>{data['EBITDA']}</strong>, beneficios: <strong>{data['BENEFICIOS']}</strong>, deuda: <strong>{data['DEUDA']}</strong>, flujo de caja: <strong>{data['FLUJO_CAJA']}</strong>), posicionamiento estratégico (planes de expansión: <strong>{data['EXPANSION_PLANES']}</strong>, acuerdos: <strong>{data['ACUERDOS']}</strong>), y comportamiento esperado del precio, mínimo 150 palabras. Hará referencia a la nota de <strong>{data['NOTA_EMPRESA']}</strong> como influencia en la salud financiera a largo plazo].</p>

<h2>Conclusión General y Descargo de Responsabilidad</h2>
<p>En resumen, mi síntesis final de este análisis. [Aquí el modelo ofrecerá un resumen de aproximadamente 100 palabras, reiterando la opinión personal sobre la empresa y su proyección].</p>
<p>Descargo de responsabilidad: Este análisis es solo informativo y no constituye una recomendación de inversión. Cada persona debe evaluar sus decisiones de forma independiente.</p>
"""

    return prompt, titulo_post


def enviar_email(texto_generado, asunto_email):
    remitente = "xumkox@gmail.com"
    destinatario = "xumkox@gmail.com"
    password = "kdgz lvdo wqvt vfkt"

    msg = MIMEMultipart()
    msg['From'] = remitente
    msg['To'] = destinatario
    msg['Subject'] = asunto_email

    # Cambiado a 'html' para que el cliente de correo interprete el formato
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
    # Se ha actualizado el modelo a una versión más reciente
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest") 

    for ticker in tickers:
        print(f"\n📊 Procesando ticker: {ticker}")
        data = obtener_datos_yfinance(ticker)
        if not data:
            continue
        prompt, titulo_post = construir_prompt_formateado(data)

        try:
            response = model.generate_content(prompt)
            print(f"\n🧠 Contenido generado para {ticker}:\n")
            print(response.text)
            asunto_email = f"Análisis: {data['NOMBRE_EMPRESA']} - {data['RECOMENDACION']}"
            enviar_email(response.text, asunto_email)
        except Exception as e:
            print(f"❌ Error generando contenido con Gemini: {e}")


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
