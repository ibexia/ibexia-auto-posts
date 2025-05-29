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
import requests
from bs4 import BeautifulSoup


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

    range_name = os.getenv('RANGE_NAME', 'A1:A100')

    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    values = result.get('values', [])

    return [row[0] for row in values if row]


def obtener_datos_yfinance(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="30d")

    try:
        hist = calcular_smi_tv(hist)
        smi_actual = round(hist['SMI'].dropna().iloc[-1], 2)

        if smi_actual > 40:
            recomendacion = "Vender"
            condicion_rsi = "sobrecomprado"
        elif smi_actual < -40:
            recomendacion = "Comprar"
            condicion_rsi = "sobrevendido"
        else:
            recomendacion = "Mantener"
            condicion_rsi = "neutral"

        datos = {
            "NOMBRE_EMPRESA": info.get("longName", ticker),
            "PRECIO_ACTUAL": round(info.get("currentPrice", 0), 2),
            "VOLUMEN": info.get("volume", 0),
            "SOPORTE": round(hist["Low"].min(), 2),
            "RESISTENCIA": round(hist["High"].max(), 2),
            "CONDICION_RSI": condicion_rsi,
            "RECOMENDACION": recomendacion,
            "SMI": smi_actual,
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
            "RIESGOS_OPORTUNIDADES": "No disponibles",
            "NOTICIAS_RECIENTES": obtener_noticias_google(info.get("longName", ticker))
        }
    except Exception as e:
        print(f"❌ Error al obtener datos de {ticker}: {e}")
        return None

    return datos


def obtener_noticias_google(nombre_empresa):
    query = nombre_empresa.replace(" ", "+")
    url = f"https://news.google.com/search?q={query}&hl=es&gl=ES&ceid=ES%3Aes"
    noticias = []

    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        enlaces = soup.select('a.DY5T1d.RZIKme')

        for i, enlace in enumerate(enlaces[:3]):
            href = enlace.get('href')
            if href.startswith('./'):
                href = 'https://news.google.com' + href[1:]
            noticias.append(href)
    except Exception as e:
        print(f"❌ Error obteniendo noticias para {nombre_empresa}: {e}")

    return noticias


def construir_prompt_formateado(data):
    noticias = "\n".join(f"- {n}" for n in data['NOTICIAS_RECIENTES']) if data['NOTICIAS_RECIENTES'] else "No se encontraron noticias recientes."

    prompt = f"""
Actúa como un trader profesional con amplia experiencia en análisis técnico y mercados financieros. Redacta en primera persona, con total confianza en tu criterio. 
Vas a generar un análisis técnico COMPLETO de aproximadamente 1000 palabras sobre la empresa: {data['NOMBRE_EMPRESA']}, utilizando los siguientes datos reales extraídos de Yahoo Finance:

- Precio actual: {data['PRECIO_ACTUAL']}
- Volumen: {data['VOLUMEN']}
- Soporte clave: {data['SOPORTE']}
- Resistencia clave: {data['RESISTENCIA']}
- Recomendación general: {data['RECOMENDACION']}
- Resultados financieros recientes: {data['INGRESOS']}, {data['EBITDA']}, {data['BENEFICIOS']}
- Nivel de deuda y flujo de caja: {data['DEUDA']}, {data['FLUJO_CAJA']}
- Información estratégica: {data['EXPANSION_PLANES']}, {data['ACUERDOS']}
- Sentimiento del mercado: {data['SENTIMIENTO_ANALISTAS']}, {data['TENDENCIA_SOCIAL']}
- Comparativa sectorial: {data['EMPRESAS_SIMILARES']}
- Riesgos y oportunidades: {data['RIESGOS_OPORTUNIDADES']}

SECCIÓN 1 – TÍTULO Y INTRODUCCIÓN
{data['NOMBRE_EMPRESA']} – Recomendación: {data['RECOMENDACION']}

Análisis técnico de {data['NOMBRE_EMPRESA']}. Comentarios a corto y largo plazo, con información y avisos sobre los últimos movimientos del precio de sus acciones. Consulta datos relevantes de indicadores y medias móviles.

SECCIÓN 2 – RECOMENDACIÓN GENERAL (mínimo 150 palabras)

SECCIÓN 3 – RECOMENDACIÓN A CORTO PLAZO (mínimo 150 palabras)

SECCIÓN 4 – PREDICCIÓN A LARGO PLAZO (mínimo 150 palabras)

SECCIÓN 5 – INFORMACIÓN ADICIONAL (mínimo 150 palabras)
Últimas noticias relevantes sobre la empresa:
{noticias}

SECCIÓN 6 – RESUMEN (aproximadamente 100 palabras)

SECCIÓN 7 – DESCARGO DE RESPONSABILIDAD
Este análisis es solo informativo y no constituye una recomendación de inversión. Cada persona debe evaluar sus decisiones de forma independiente.
"""
    return prompt


def calcular_smi_tv(df):
    high = df['High']
    low = df['Low']
    close = df['Close']

    hh = high.rolling(window=14).max()
    ll = low.rolling(window=14).min()
    diff = hh - ll
    rdiff = close - (hh + ll) / 2

    avgrel = rdiff.ewm(span=3, adjust=False).mean()
    avgdiff = diff.ewm(span=3, adjust=False).mean()

    smi_raw = (avgrel / (avgdiff / 2)) * 100
    smi_raw[avgdiff == 0] = 0.0

    smi_smoothed = smi_raw.rolling(window=3).mean()
    df = df.copy()
    df['SMI'] = smi_smoothed
    return df


def enviar_email(texto_generado):
    remitente = "xumkox@gmail.com"
    destinatario = "xumkox@gmail.com"
    asunto = "Contenido generado por Gemini"
    password = "kdgz lvdo wqvt vfkt"

    msg = MIMEMultipart()
    msg['From'] = remitente
    msg['To'] = destinatario
    msg['Subject'] = asunto
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

    contenido_final = ""

    for ticker in tickers:
        print(f"\n📊 Procesando ticker: {ticker}")
        data = obtener_datos_yfinance(ticker)
        if not data:
            continue
        prompt = construir_prompt_formateado(data)

        try:
            response = model.generate_content(prompt)
            contenido_final += f"\n\n--- ANÁLISIS PARA {ticker} ---\n\n"
            contenido_final += response.text
        except Exception as e:
            print(f"❌ Error generando contenido con Gemini: {e}")

    if contenido_final:
        enviar_email(contenido_final)


def main():
    tickers = leer_google_sheets()

    if not tickers:
        print("❌ No se encontraron tickers en la hoja.")
        return

    total = len(tickers)
    day_index = datetime.now().weekday()  # 0 = lunes, ..., 6 = domingo

    # Desplazamiento cíclico por bloques de 10
    start = (day_index * 10) % total
    end = start + 10

    if end <= total:
        tickers_del_dia = tickers[start:end]
    else:
        # Si el rango se pasa del final, tomar del final y continuar desde el inicio
        tickers_del_dia = tickers[start:] + tickers[:end - total]

    print(f"✅ Tickers para hoy ({day_index}): {tickers_del_dia}")
    generar_contenido_con_gemini(tickers_del_dia)

if __name__ == '__main__':
    main()
