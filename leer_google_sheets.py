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
    result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    values = result.get('values', [])

    if not values:
        print('No se encontraron datos.')
    else:
        print('Datos leídos de la hoja:')
        for row in values:
            print(row)

    return [row[0] for row in values if row]


length_k = 10  # Ajustado a los valores del código (1)
length_d = 3   # Ajustado a los valores del código (1)
ema_signal_len = 10 # Ajustado a los valores del código (1)
smooth_period = 5 # Ajustado a los valores del código (1)

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
    smi_signal = smi_smoothed.ewm(span=ema_signal_len, adjust=False).mean() # Aquí se usa ema_signal_len para calcular SMI_signal

    df = df.copy()
    df['SMI'] = smi_smoothed
    df['SMI_signal'] = smi_signal # Se añade SMI_signal al DataFrame
    
    return df


def obtener_datos_yfinance(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    # Asegúrate de descargar suficientes datos para los cálculos del SMI
    hist = stock.history(period="6mo") # Se cambió a 6 meses como en el código (1)

    if hist.empty:
        print(f"❌ No se pudieron obtener datos históricos para {ticker}")
        return None

    try:
        hist = calculate_smi_tv(hist)
        # Se verifica que 'SMI_signal' existe antes de intentar acceder a él
        if 'SMI_signal' not in hist.columns or hist['SMI_signal'].empty:
            print(f"❌ SMI_signal no disponible para {ticker}")
            return None

        smi_actual = round(hist['SMI_signal'].dropna().iloc[-1], 2) # Usar SMI_signal para la recomendación

        # Calcular la nota de la empresa como en el código (1)
        nota_empresa = round((-(max(min(smi_actual, 60), -60)) + 60) * 10 / 120, 1)

        # Lógica de recomendaciones basada en la NOTA DE LA EMPRESA (0-10)
        if nota_empresa <= 2:
            recomendacion = "Comprar con Urgencia"
            condicion_rsi = "extremadamente sobrevendido"
        elif 2 < nota_empresa <= 4:
            recomendacion = "Se acerca la hora de comprar"
            condicion_rsi = "fuertemente sobrevendido"
        elif 4 < nota_empresa <= 5:
            recomendacion = "Considerar compra parcial"
            condicion_rsi = "sobrevendido"
        elif 5 < nota_empresa < 6: # Nota entre 5 y 6 (exclusivos)
            recomendacion = "Mantener (Neutro)"
            condicion_rsi = "neutral"
        elif 6 <= nota_empresa < 7:
            recomendacion = "Vigilancia de retroceso"
            condicion_rsi = "ligeramente sobrecomprado"
        elif 7 <= nota_empresa < 8:
            recomendacion = "Considerar venta parcial"
            condicion_rsi = "sobrecomprado"
        elif 8 <= nota_empresa < 9:
            recomendacion = "Se acerca la hora de vender"
            condicion_rsi = "fuertemente sobrecomprado"
        elif nota_empresa >= 9:
            recomendacion = "Vender con Urgencia"
            condicion_rsi = "extremadamente sobrecomprado"
        else:
            recomendacion = "Indefinido" # Por si acaso algún valor no cae en los rangos anteriores
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
            "NOTA_EMPRESA": nota_empresa, # Se añade la nota de la empresa
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

Vas a generar un análisis técnico completo de aproximadamente 1000 palabras sobre la empresa {data['NOMBRE_EMPRESA']}, utilizando los siguientes datos reales extraídos de Yahoo Finance. Presta especial atención a la **nota obtenida por la empresa**: {data['NOTA_EMPRESA']}.

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

Importante: si algún dato no está disponible, no lo menciones ni digas que falta. No expliques que la recomendación proviene de un indicador o dato específico. La recomendación debe presentarse como una conclusión personal basada en tu experiencia y criterio profesional como analista. Al redactar el análisis, haz referencia a la **nota obtenida por la empresa ({data['NOTA_EMPRESA']})** en al menos dos de las secciones principales (Recomendación General, Análisis a Corto Plazo o Predicción a Largo Plazo) como un factor clave para tu valoración.

Estructura el texto de la siguiente manera:

SECCIÓN 1 – TÍTULO E INTRODUCCIÓN
Presentación general de la empresa y de la situación actual del mercado en torno a ella. Describe brevemente el contexto técnico, financiero y estratégico, mencionando cómo la **nota de {data['NOMBRE_EMPRESA']} de {data['NOTA_EMPRESA']}** sitúa a la empresa en el panorama actual.

SECCIÓN 2 – RECOMENDACIÓN GENERAL 
Expón tu opinión profesional sobre la situación actual de la empresa y sus perspectivas (mínimo 150 palabras). Usa un enfoque técnico y financiero combinado, sin justificar con fuentes externas. Solo tu criterio como analista. La **nota de {data['NOTA_EMPRESA']}** es un factor determinante en mi visión de la empresa.

SECCIÓN 3 – ANÁLISIS A CORTO PLAZO 
Describe los posibles movimientos del precio en el corto plazo. (mínimo 150 palabras)Incluye consideraciones sobre volumen, soportes y resistencias, y cualquier otro elemento técnico que consideres relevante. Dada la **nota de {data['NOTA_EMPRESA']}**, anticipo ciertos comportamientos en el precio a corto plazo.

SECCIÓN 4 – PREDICCIÓN A LARGO PLAZO 
Desarrolla tu visión a futuro para la empresa, (mínimo 150 palabras) incluyendo análisis financiero, posicionamiento estratégico y comportamiento esperado del precio. Mi predicción a largo plazo está fuertemente influenciada por la **nota de {data['NOTA_EMPRESA']}** y su implicación en la salud financiera de la empresa.

SECCIÓN 5 – RESUMEN proximadamente 100 palabras)
Síntesis final de tu análisis. Reitera tu opinión personal sobre la empresa y su proyección. (a

SECCIÓN 6 – DESCARGO DE RESPONSABILIDAD
Este análisis es solo informativo y no constituye una recomendación de inversión. Cada persona debe evaluar sus decisiones de forma independiente.

"""

    return prompt

def enviar_email(texto_generado):
    remitente = "xumkox@gmail.com"
    destinatario = "xumkox@gmail.com"
    asunto = "Analisis empresas"
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
    
    # Ya no hay restricción por día de la semana, se procesa todos los días (0 a 6)
    num_tickers_per_day = 10
    total_tickers_in_sheet = len(all_tickers)
    
    # Calcular el índice de inicio para el día actual.
    # El operador módulo garantiza que el ciclo de tickers se repita cada 7 días.
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


if __name__ == '__main__':
    main()
