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


def obtener_datos_yfinance(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="6mo")

    if hist.empty:
        print(f"❌ No se pudieron obtener datos históricos para {ticker}")
        return None

    try:
        hist = calculate_smi_tv(hist)
        if 'SMI_signal' not in hist.columns or hist['SMI_signal'].empty:
            print(f"❌ SMI_signal no disponible para {ticker}")
            return None

        smi_actual = round(hist['SMI_signal'].dropna().iloc[-1], 2)

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


        datos = {
            "NOMBRE_EMPRESA": info.get("longName", ticker),
            "PRECIO_ACTUAL": round(info.get("currentPrice", 0), 2),
            "VOLUMEN": info.get("volume", 0),
            "SOPORTE": round(hist["Low"].min(), 2),
            "RESISTENCIA": round(hist["High"].max(), 2),
            "CONDICION_RSI": condicion_rsi,
            "RECOMENDACION": recomendacion,
            "SMI": smi_actual,
            "NOTA_EMPRESA": nota_empresa,
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
Actúa como un trader profesional con amplia experiencia en análisis técnico y mercados financieros. Redacta en primera persona, con total confianza en tu criterio. 

Genera un análisis técnico completo de aproximadamente 1000 palabras sobre la empresa {data['NOMBRE_EMPRESA']}, utilizando los siguientes datos reales extraídos de Yahoo Finance. Presta especial atención a la **nota obtenida por la empresa**: {data['NOTA_EMPRESA']}.

**Título del Post:** {titulo_post}

- Precio actual: {data['PRECIO_ACTUAL']}
- Volumen: {data['VOLUMEN']}
- Soporte clave: {data['SOPORTE']}
- Resistencia clave: {data['RESISTENCIA']}
- Recomendación general: {data['RECOMENDACION']}
- Nota de la empresa (0-10): {data['NOTA_EMPRESA']}
- Resultados financieros recientes: {data['INGRESOS']}, {data['EBITDA']}, {data['BENEFICIOS']}
- Nivel de deuda y flujo de caja: {data['DEUDA']}, {data['FLUJO_CAJA']}
- Información estratégica: {data['EXPANSION_PLANES']}, {data['ACUERDOS']}
- Sentimiento del mercado: {data['SENTIMIENTO_ANALISTAS']}, {data['TENDENCIA_SOCIAL']}
- Comparativa sectorial: {data['EMPRESAS_SIMILARES']}
- Riesgos y oportunidades: {data['RIESGOS_OPORTUNIDADES']}

Importante: si algún dato no está disponible, no lo menciones ni digas que falta. No expliques que la recomendación proviene de un indicador o dato específico. La recomendación debe presentarse como una conclusión personal basada en tu experiencia y criterio profesional como analista. Al redactar el análisis, haz referencia a la **nota obtenida por la empresa ({data['NOTA_EMPRESA']})** en al menos dos de los párrafos principales (Recomendación General, Análisis a Corto Plazo o Predicción a Largo Plazo) como un factor clave para tu valoración.

Estructura el texto de la siguiente manera, sin usar títulos de sección explícitos, sino comenzando cada párrafo con una frase introductoria:

{titulo_post}

Para comenzar el análisis de **{data['NOMBRE_EMPRESA']}**, quiero dejar clara mi recomendación principal: **{data['RECOMENDACION']}**. Este juicio se fundamenta en un análisis exhaustivo de su situación actual, donde la **nota de {data['NOTA_EMPRESA']}** juega un papel crucial. La empresa se encuentra en un punto estratégico en el mercado, con un precio actual de {data['PRECIO_ACTUAL']}€ y un volumen de {data['VOLUMEN']}.

Como recomendación general, mi opinión profesional sobre la situación actual de **{data['NOMBRE_EMPRESA']}** y sus perspectivas es la siguiente: [Aquí el modelo expandirá la recomendación, mínimo 150 palabras, usando un enfoque técnico y financiero combinado. Mencionará la nota de {data['NOTA_EMPRESA']} como factor determinante].

En el análisis a corto plazo, considero los posibles movimientos del precio en el horizonte inmediato. [Aquí el modelo describirá movimientos, volumen, soportes ({data['SOPORTE']}) y resistencias ({data['RESISTENCIA']}), mínimo 150 palabras. Hará referencia a la nota de {data['NOTA_EMPRESA']} si lo considera relevante para el corto plazo].

Respecto a la predicción a largo plazo, mi visión para el futuro de la empresa incluye... [Aquí el modelo desarrollará la visión a futuro, análisis financiero (ingresos: {data['INGRESOS']}, EBITDA: {data['EBITDA']}, beneficios: {data['BENEFICIOS']}, deuda: {data['DEUDA']}, flujo de caja: {data['FLUJO_CAJA']}), posicionamiento estratégico (planes de expansión: {data['EXPANSION_PLANES']}, acuerdos: {data['ACUERDOS']}), y comportamiento esperado del precio, mínimo 150 palabras. Hará referencia a la nota de {data['NOTA_EMPRESA']} como influencia en la salud financiera a largo plazo].

En resumen, mi síntesis final de este análisis. [Aquí el modelo ofrecerá un resumen de aproximadamente 100 palabras, reiterando la opinión personal sobre la empresa y su proyección].

Descargo de responsabilidad: Este análisis es solo informativo y no constituye una recomendación de inversión. Cada persona debe evaluar sus decisiones de forma independiente.
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

    msg.attach(MIMEText(texto_generado, 'plain'))

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
    model = genai.GenerativeModel(model_name="models/gemini-2.0-flash-lite")

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
