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

    # Modificado: Se eliminó el límite superior para leer todas las filas en la columna A
    # Esto asegura que se lean todas las 70 filas (o más) de tickers.
    range_name = os.getenv('RANGE_NAME', 'A:A')

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
    hist = stock.history(period="30d")  # 30 días para mejor cálculo del SMI

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
            "RIESGOS_OPORTUNIDADES": "No disponibles"
        }
    except Exception as e:
        print(f"❌ Error al obtener datos de {ticker}: {e}")
        return None

    return datos


def construir_prompt_formateado(data):
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
Incluye aquí información reciente y relevante como noticias del mercado, futuros contratos, movimientos destacados o cualquier dato externo de interés para entender mejor la situación actual de la empresa.

SECCIÓN 6 – RESUMEN (aproximadamente 100 palabras)

SECCIÓN 7 – DESCARGO DE RESPONSABILIDAD
Este análisis es solo informativo y no constituye una recomendación de inversión. Cada persona debe evaluar sus decisiones de forma independiente.

"""
    return prompt

length_k = 14
length_d = 3
smooth_period = 3
ema_signal_len = 3

def calcular_smi_tv(df):
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
    
    df = df.copy()
    df['SMI'] = smi_smoothed
    
    return df

    
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
    all_tickers = leer_google_sheets()[1:]  # Esto salta la primera fila (los encabezados)
    
    if not all_tickers:
        print("No hay tickers para procesar.")
        return

    day_of_week = datetime.today().weekday()  # Lunes es 0, Martes 1, ..., Domingo 6
    
    # Solo procesamos de lunes a viernes (0 a 4)
    if 0 <= day_of_week <= 4:
        num_tickers_per_day = 10
        total_tickers_in_sheet = len(all_tickers)
        
        # Calcular el índice de inicio directamente basado en el día de la semana.
        # El operador módulo garantiza que el índice se "envuelva" si el número de tickers
        # es menor que el total de tickers procesados en una semana (50).
        start_index = (day_of_week * num_tickers_per_day) % total_tickers_in_sheet
        
        end_index = start_index + num_tickers_per_day
        
        tickers_for_today = []
        if end_index <= total_tickers_in_sheet:
            tickers_for_today = all_tickers[start_index:end_index]
        else:
            # Si el final del bloque excede el total de tickers,
            # tomamos lo que queda hasta el final y luego volvemos al principio.
            tickers_for_today = all_tickers[start_index:] + all_tickers[:end_index - total_tickers_in_sheet]

        if tickers_for_today:
            print(f"Procesando tickers para el día {datetime.today().strftime('%A')}: {tickers_for_today}")
            generar_contenido_con_gemini(tickers_for_today)
        else:
            print(f"No hay tickers disponibles para el día {datetime.today().strftime('%A')} en el rango calculado. "
                  f"start_index: {start_index}, end_index: {end_index}, total_tickers: {total_tickers_in_sheet}")
    else:
        print("Hoy es fin de semana. No se procesarán tickers.")


if __name__ == '__main__':
    main()
