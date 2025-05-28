import os
import json
import smtplib
import yfinance as yf
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.oauth2 import service_account
from googleapiclient.discovery import build
import google.generativeai as genai


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

    range_name = os.getenv('RANGE_NAME', 'A1:A10')  # Solo tickers

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


def obtener_datos_yfinance(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="5d")

    try:
        datos = {
            "NOMBRE_EMPRESA": info.get("longName", ticker),
            "PRECIO_ACTUAL": round(info.get("currentPrice", 0), 2),
            "VOLUMEN": info.get("volume", 0),
            "SOPORTE": round(hist["Low"].min(), 2),
            "RESISTENCIA": round(hist["High"].max(), 2),
            "CONDICION_RSI": "sobrecomprado" if info.get("rsi", 50) > 70 else "sobrevendido" if info.get("rsi", 50) < 30 else "neutral",
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
    prompt = f"""
Actúa como un trader profesional con amplia experiencia en análisis técnico y mercados financieros. Redacta en primera persona, con total confianza en tu criterio. Usa un tono directo, persuasivo y magnético, transmitiendo certeza absoluta sobre la evolución del precio. Genera urgencia y emoción, haciendo ver que esta información es una oportunidad única para el lector.

Vas a generar un análisis técnico COMPLETO de 1000 palabras sobre la empresa: {data['NOMBRE_EMPRESA']}, utilizando los siguientes datos reales extraídos de Yahoo Finance (yfinance):
- Precio actual: {data['PRECIO_ACTUAL']}
- Volumen: {data['VOLUMEN']}
- Soporte clave: {data['SOPORTE']}
- Resistencia clave: {data['RESISTENCIA']}
- ¿Está sobrecomprado o sobrevendido?: {data['CONDICION_RSI']}
- Resultados financieros recientes: {data['INGRESOS']}, {data['EBITDA']}, {data['BENEFICIOS']}
- Nivel de deuda y flujo de caja: {data['DEUDA']}, {data['FLUJO_CAJA']}
- Información estratégica: {data['EXPANSION_PLANES']}, {data['ACUERDOS']}
- Sentimiento del mercado: {data['SENTIMIENTO_ANALISTAS']}, {data['TENDENCIA_SOCIAL']}
- Comparativa sectorial: {data['EMPRESAS_SIMILARES']}
- Riesgos y oportunidades: {data['RIESGOS_OPORTUNIDADES']}

🟨 SECCIÓN 1 – TÍTULO Y INTRODUCCIÓN
**{data['NOMBRE_EMPRESA']} – Recomendación de [Comprar/Vender/Mantener]**

**Análisis técnico de {data['NOMBRE_EMPRESA']}. Comentarios a corto y largo plazo e información y avisos sobre los últimos movimientos sobre el precio de sus acciones. Consulta los datos de medias móviles, RSI, MACD, Boolinger.**
www.ibexia.es

🟨 SECCIÓN 2 – RECOMENDACIÓN GENERAL (mínimo 150 palabras)
🟨 SECCIÓN 3 – RECOMENDACIÓN A CORTO PLAZO (mínimo 150 palabras)
🟨 SECCIÓN 4 – PREDICCIÓN A LARGO PLAZO (mínimo 150 palabras)
🟨 SECCIÓN 5 – INFORMACIÓN ADICIONAL (mínimo 150 palabras)
🟨 SECCIÓN 6 – RESUMEN (aprox. 100 palabras)
🟨 SECCIÓN 7 – DESCARGO DE RESPONSABILIDAD
✅ Usa palabras clave en **negrita** como: **análisis técnico**, **compra**, **venta**, **cómo invertir**, **brokers**, **plataformas de trading**, **acciones con potencial**.
✅ Repite el nombre de la empresa al menos 10 veces.
✅ Usa títulos H1 y H2 con emojis apropiados (📈, 📉, 💼, ⚠️, etc.).
✅ No incluyas enlaces externos, salvo www.ibexia.es.
    """
    return prompt


def enviar_email(texto_generado):
    remitente = "xumkox@gmail.com"
    destinatario = "xumkox@gmail.com"
    asunto = "Contenido generado por Gemini"
    password = "kdgz lvdo wqvt vfkt"  # Asegúrate de usar contraseña de aplicación segura

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

    for ticker in tickers:
        print(f"\n📊 Procesando ticker: {ticker}")
        data = obtener_datos_yfinance(ticker)
        if not data:
            continue
        prompt = construir_prompt_formateado(data)

        try:
            response = model.generate_content(prompt)
            print(f"\n🧠 Contenido generado para {ticker}:\n")
            print(response.text)
            enviar_email(response.text)
        except Exception as e:
            print(f"❌ Error generando contenido con Gemini: {e}")


def main():
    tickers = leer_google_sheets()[1:]  # Esto salta la primera fila (los encabezados)
    if tickers:
        generar_contenido_con_gemini(tickers)


if __name__ == '__main__':
    main()
