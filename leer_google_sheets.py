import os
import json
import smtplib
import yfinance as yf
import google.generativeai as genai
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
import re
import random

# IMPORTACIÓN NECESARIA PARA GENERAR EL GRÁFICO CANDLESTICK PROFESIONAL
import mplfinance as mpf 

# Configuración de entorno (debes asegurarte de tener estas variables definidas)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
SENDER_EMAIL = os.getenv('SENDER_EMAIL')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL')
SHEET_ID = os.getenv('SHEET_ID')
RANGE_NAME = os.getenv('RANGE_NAME', 'Hoja1!A:A')
# Mantenemos las variables originales.
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS') 

# NUEVA FUNCIÓN AÑADIDA PARA GARANTIZAR LA SERIALIZACIÓN A JSON/NULL
def safe_json_dump(data_list):
    """
    Serializa una lista de Python a una cadena JSON, asegurando que los valores None
    se conviertan a la palabra clave 'null' de JavaScript.
    """
    # json.dumps convierte None a 'null' y los floats a formato JavaScript (con punto decimal)
    # Se asegura de que la lista solo contenga valores o None, para que json.dumps funcione.
    return json.dumps([val if val is not None else None for val in data_list])

def leer_google_sheets():
    """Lee la lista de tickers desde Google Sheets."""
    # NO TOCAR NADA EN ESTA FUNCIÓN para que siga funcionando con la configuración de credenciales existente
    if not GOOGLE_APPLICATION_CREDENTIALS:
        # Se asume que si no está definida, es porque el entorno de ejecución la maneja de otra forma
        # o que el error se gestionará externamente.
        print("⚠️ Variable GOOGLE_APPLICATION_CREDENTIALS no definida. Intentando continuar...")
        return [] # Retorna vacío para evitar fallos si el error no es crítico

    try:
        # La lógica original asume que GOOGLE_APPLICATION_CREDENTIALS contiene la ruta o el JSON.
        # Aquí cargamos desde la variable de entorno, como en el intento anterior, pero SIN la comprobación extra de 'raise Exception'
        creds_dict = json.loads(GOOGLE_APPLICATION_CREDENTIALS)
        creds = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
        )
        
        service = build('sheets', 'v4', credentials=creds)
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=SHEET_ID, range=RANGE_NAME).execute()
        values = result.get('values', [])
        
        if not values:
            return []

        # Aplana la lista de listas para obtener una lista simple de tickers
        tickers = [item[0].strip() for item in values if item and item[0].strip()]
        return tickers

    except Exception as e:
        # Importante: El error original era 'Missing required parameter "spreadsheetId"'.
        # Esto indica que SHEET_ID está vacío, no las credenciales.
        # Mantener la gestión de errores original para no cambiar la lógica.
        print(f"❌ Error al acceder a Google Sheets: {e}")
        return []

def enviar_email(subject, body, attachment_path=None):
    """Envía un correo electrónico con el contenido generado y un gráfico adjunto."""
    if not SENDER_EMAIL or not SENDER_PASSWORD or not RECIPIENT_EMAIL:
        print("❌ Faltan credenciales de correo electrónico. No se puede enviar el email.")
        return

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'html'))

    if attachment_path and os.path.exists(attachment_path):
        try:
            with open(attachment_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f"attachment; filename= {os.path.basename(attachment_path)}",
            )
            msg.attach(part)
        except Exception as e:
            print(f"❌ Error al adjuntar archivo {attachment_path}: {e}")

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        print("✅ Email enviado correctamente.")
        # Limpiar el archivo adjunto después de enviarlo
        if attachment_path and os.path.exists(attachment_path):
            os.remove(attachment_path)
    except Exception as e:
        print(f"❌ Error al enviar el email: {e}")

def generar_grafico_candlestick(df, ticker, projected_col_name='Precio_Proyectado'):
    """
    Genera un gráfico de velas (candlestick) profesional con las correcciones solicitadas:
    1. Reducción del grosor (más profesional).
    2. Eje X continuo (sin huecos) usando x_axis_type='index'.
    3. Eliminación de la línea de "cierre real" (solo velas).
    4. Línea de precio proyectado a trazos (--).
    """
    plot_filepath = f"{ticker}_analisis.png"
    
    # 1. & 4. Definición de estilo profesional y más delgado
    # Usamos inherit para un look más limpio y bordes menos pronunciados.
    mc = mpf.make_marketcolors(
        up='g', down='r', 
        edge='inherit',      # Bordes más delgados
        wick='inherit',      # Mechas más delgadas
        volume='inherit'
    )
    
    # Usar un estilo base simple y aplicar personalizaciones
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
    
    # Ajustes finos para hacer las velas y mechas más delgadas (Grosor menor)
    config = {
        # CRÍTICO 1: Reducir el grosor de velas y mechas (más profesional)
        'widths': {'candle_width': 0.6, 'wick_width': 0.3}, 
        'marketcolors': mc,
        'mavcolors': ['#F08080', '#ADD8E6'], # Colores de medias móviles si se usan
        'linecolor': '#2A2A2A', 
    }
    s['y_on_right'] = False
    # CRÍTICO 1: Afecta a líneas genéricas de matplotlib (ej. líneas de rejilla/fondo)
    s['rc'] = {'lines.linewidth': 0.5} 

    # Aseguramos que la columna de proyección exista, si no, la creamos como PLACEHOLDER
    # Mantenemos esta lógica porque el código original no mostraba cómo se calcula esta columna.
    if projected_col_name not in df.columns:
        df[projected_col_name] = df['Close'].rolling(window=10).mean()

    # CRÍTICO 3: Solo añadimos la línea proyectada. La línea de "cierre real" se quita
    # simplemente no añadiéndola aquí. Las velas siguen visibles por 'type='candle''.
    apds = []
    if projected_col_name in df.columns and not df[projected_col_name].isnull().all():
        addplot_projection = mpf.make_addplot(
            df[projected_col_name], 
            color='blue', 
            linestyle='--',     # CRÍTICO 4: Línea a trazos
            linewidth=1.5,
            panel=0, 
            type='line', 
            secondary_y=False,
            label=projected_col_name
        )
        apds.append(addplot_projection)
    
    # CRÍTICO 2: Plotear usando index axis type para eliminar los huecos
    mpf.plot(
        df,
        type='candle', # Mantiene las velas visibles
        addplot=apds,
        title=f'\nAnálisis Candlestick (Sin Huecos) para {ticker}',
        ylabel='Precio (€)',
        style=s,
        config=config, 
        savefig=dict(fname=plot_filepath, dpi=300, bbox_inches='tight'),
        x_axis_type='index', # CRÍTICO 2: Asegura el eje continuo (sin huecos de no-operación)
        show_nontrading=False 
    )

    return plot_filepath


def generar_contenido_con_gemini(tickers):
    """
    Procesa cada ticker, obtiene datos, genera el gráfico y llama a Gemini.
    """
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY no está configurada. Saliendo.")
        return

    genai.configure(api_key=GEMINI_API_KEY)
    
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction=(
            "Eres un analista financiero experto. Genera un informe conciso y objetivo "
            "basado en los datos de precios, el gráfico adjunto y la proyección. "
            "El informe debe incluir: 1. Un breve resumen de la tendencia actual. "
            "2. El significado de la 'Precio_Proyectado' (que es una media móvil simple). "
            "3. Una conclusión sobre el activo. Responde siempre en español. No uses negritas."
        )
    )

    # Definir el rango de fechas: 60 días hasta ayer
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90) # Obtener unos 3 meses de datos

    for ticker in tickers:
        print(f"⚙️ Procesando ticker: {ticker}...")
        
        # 1. Obtener datos históricos
        try:
            df_yf = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1d')
        except Exception as e:
            print(f"❌ Error al descargar datos de {ticker}: {e}")
            continue

        if df_yf.empty:
            print(f"⚠️ No se encontraron datos para {ticker}. Saltando.")
            continue

        # 2. Generar el gráfico con las correcciones
        try:
            plot_filepath = generar_grafico_candlestick(df_yf, ticker)
            print(f"✅ Gráfico generado y guardado en {plot_filepath}")
        except Exception as e:
            print(f"❌ Error al generar el gráfico para {ticker}: {e}")
            continue

        # 3. Preparar los datos para Gemini (ejemplo: últimos 5 cierres)
        last_closes = df_yf['Close'].tail(5).tolist()
        last_dates = df_yf.index.strftime('%Y-%m-%d').tail(5).tolist()
        
        data_for_gemini = (
            f"Ticker: {ticker}\n"
            f"Últimas 5 fechas y cierres: {safe_json_dump(list(zip(last_dates, last_closes)))}\n"
            f"Datos históricos (Open, High, Low, Close, Volume) adjuntos en el gráfico."
        )

        # 4. Llamar a la API de Gemini
        try:
            response = model.generate_content(
                [
                    data_for_gemini,
                    f"Genera el informe para el ticker {ticker}",
                    plot_filepath  # Pasar la ruta de la imagen
                ]
            )
            reporte_html = response.text.replace('\n', '<br>')
            print(f"✅ Reporte de Gemini generado para {ticker}.")
        except Exception as e:
            reporte_html = f"❌ Error al generar contenido con Gemini: {e}"
            print(reporte_html)

        # 5. Enviar el email
        email_subject = f"Análisis de {ticker} | Informe y Gráfico Actualizado"
        email_body = (
            f"Hola,<br><br>"
            f"Adjunto el informe generado por el analista (Gemini) y el gráfico de velas actualizado para el ticker <b>{ticker}</b>.<br><br>"
            f"---<br>"
            f"<b>Informe de Gemini:</b><br>"
            f"{reporte_html}<br>"
            f"---<br><br>"
            f"Saludos."
        )
        enviar_email(email_subject, email_body, plot_filepath)

        # 6. Pausa para evitar límites de tasa
        print(f"⏳ Esperando 180 segundos antes de procesar el siguiente ticker...")
        time.sleep(180)


def main():
    """Función principal para coordinar la lectura de hojas y el procesamiento diario."""
    try:
        all_tickers = leer_google_sheets()
        # El código original usaba [1:] para saltar la cabecera. Lo mantengo.
        all_tickers = all_tickers[1:] 
    except Exception as e:
        print(f"❌ Error al leer Google Sheets: {e}. Asegúrate de que las variables de entorno están configuradas correctamente y el archivo JSON de credenciales es válido.")
        return
    
    if not all_tickers:
        print("No hay tickers para procesar.")
        return

    # Lógica de rotación de tickers (Mantenida exactamente como estaba)
    day_of_week = datetime.today().weekday()
    
    num_tickers_per_day = 12  
    total_tickers_in_sheet = len(all_tickers)
    
    # Calcular el índice inicial para el día de hoy
    start_index = (day_of_week * num_tickers_per_day) % total_tickers_in_sheet
    
    end_index = start_index + num_tickers_per_day
    
    tickers_for_today = []
    if end_index <= total_tickers_in_sheet:
        tickers_for_today = all_tickers[start_index:end_index]
    else:
        # Manejar el caso de desbordamiento (lista circular)
        tickers_for_today = all_tickers[start_index:] + all_tickers[:end_index - total_tickers_in_sheet]

    if tickers_for_today:
        print(f"Procesando tickers para el día {datetime.today().strftime('%A')}: {tickers_for_today}")
        generar_contenido_con_gemini(tickers_for_today)
    else:
        print(f"No hay tickers para procesar hoy después de la lógica de rotación.")


if __name__ == '__main__':
    main()
