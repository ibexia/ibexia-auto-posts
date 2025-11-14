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

# IMPORTANTE: Se añade mplfinance para generar los gráficos
import mplfinance as mpf
from io import BytesIO

# Configuración de Logging
# import logging
# logging.basicConfig(level=logging.INFO)

# --- FUNCIONES DE UTILIDAD (SIN CAMBIOS) ---

def safe_json_dump(data_list):
    """
    Serializa una lista de Python a una cadena JSON, asegurando que los valores None
    se conviertan a la palabra clave 'null' de JavaScript.
    """
    return json.dumps([val if val is not None else None for val in data_list])


def leer_google_sheets():
    """Lee tickers de Google Sheets usando las credenciales de entorno."""
    credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not credentials_json:
        raise Exception("No se encontró la variable de entorno GOOGLE_APPLICATION_CREDENTIALS")

    creds_dict = json.loads(credentials_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
    )
    
    # ID de la hoja de cálculo y rango de la variable de entorno
    try:
        spreadsheet_id = os.environ['SPREADSHEET_ID']
        range_name = os.environ['RANGE_NAME'] # Ej: 'Hoja1!A1:A'
    except KeyError as e:
        raise Exception(f"Falta la variable de entorno: {e}")

    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    
    result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    values = result.get('values', [])
    
    if not values:
        return []
    
    # Aplanar la lista de listas (cada fila de A1:A es una lista de 1 elemento)
    tickers = [item[0] for item in values if item]
    return tickers

def obtener_datos_yfinance(ticker, start_date, end_date):
    """
    Descarga datos históricos de Yahoo Finance para un ticker.
    """
    try:
        # Descargar los datos, incluyendo Open, High, Low, Close
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        
        if df.empty:
            print(f"⚠️ No se encontraron datos para {ticker} en el rango especificado.")
            return None
            
        # Asegurar que el DataFrame tiene las columnas necesarias
        df = df[['Open', 'High', 'Low', 'Close']].dropna()
        
        # Eliminar filas con valores NaN que pueden aparecer en las primeras filas
        return df.dropna()
    except Exception as e:
        print(f"❌ Error al descargar datos de {ticker}: {e}")
        return None

def calcular_algoritmo(df):
    """
    Calcula un valor de algoritmo ficticio y un precio proyectado para el ejemplo.
    (Mantenido como ejemplo de estructura, NO tocar la lógica de los datos)
    """
    if df.empty:
        return None, None

    # Simulación de un indicador que oscila entre 0 y 100
    df['Algoritmo'] = (df['Close'] - df['Close'].rolling(window=20).min()) / \
                     (df['Close'].rolling(window=20).max() - df['Close'].rolling(window=20).min()) * 100
    df['Algoritmo'] = df['Algoritmo'].fillna(50.0).clip(0, 100) # Llenar NaN y limitar

    # Simulación de un precio proyectado (una media móvil simple + un offset)
    df['Proyectado'] = df['Close'].rolling(window=5).mean() * (1 + random.uniform(-0.01, 0.01))
    df['Proyectado'] = df['Proyectado'].shift(1) # Para que no use el cierre de hoy
    df['Proyectado'] = df['Proyectado'].fillna(method='bfill') # Rellenar primeros NaN

    # El DataFrame ahora contiene Open, High, Low, Close, Algoritmo, Proyectado
    return df

def enviar_email_con_adjunto(to_email, subject, body, attachment_content, attachment_filename):
    """Envía un correo electrónico con un archivo adjunto."""
    try:
        # Credenciales de Email (variables de entorno)
        smtp_server = os.environ['SMTP_SERVER']
        smtp_port = int(os.environ['SMTP_PORT'])
        smtp_username = os.environ['SMTP_USERNAME']
        smtp_password = os.environ['SMTP_PASSWORD']

        msg = MIMEMultipart()
        msg['From'] = smtp_username
        msg['To'] = to_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'html'))

        # Adjuntar la imagen (el gráfico)
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment_content)
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename= {attachment_filename}")
        msg.attach(part)

        # Conexión y envío del email
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(smtp_username, smtp_password)
            server.sendmail(smtp_username, to_email, msg.as_string())
        
        print(f"📧 Email enviado a {to_email} con el gráfico de {attachment_filename}.")

    except Exception as e:
        print(f"❌ Error al enviar el email: {e}")

# --- FUNCIÓN PRINCIPAL DE INTERACCIÓN CON GEMINI (MODIFICADA SOLO EN LA GENERACIÓN DEL GRÁFICO) ---

def generar_contenido_con_gemini(tickers):
    """
    Procesa una lista de tickers, genera el contenido de Gemini y lo envía por email.
    """
    try:
        # Configuración de la API de Gemini
        gemini_api_key = os.environ['GEMINI_API_KEY']
        genai.configure(api_key=gemini_api_key)
        client = genai.Client()
        
        # El email de destino se toma de las variables de entorno
        to_email = os.environ['TO_EMAIL']
        
    except KeyError as e:
        print(f"❌ Falta la variable de entorno de configuración: {e}")
        return

    # Definir el rango de fechas (últimos 90 días)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=90)

    for ticker in tickers:
        print(f"⚙️ Procesando ticker: {ticker}...")
        
        df = obtener_datos_yfinance(ticker, start_date, end_date)
        if df is None or df.empty:
            continue

        df = calcular_algoritmo(df)
        if df is None:
            continue

        # Últimos datos para el prompt
        ultimo_cierre = df['Close'].iloc[-1]
        algoritmo_hoy = df['Algoritmo'].iloc[-1]
        proyectado_manana = df['Proyectado'].iloc[-1]

        # --- PREPARACIÓN DEL GRÁFICO (IMPLEMENTACIÓN DE CORRECCIONES) ---

        # 1. Preparar el DataFrame para la trama sin huecos (Sequential Index)
        # 2. Calcular límites para el escalado (Combine Plots & Scaling)
        min_price = df['Close'].min()
        max_price = df['Close'].max()
        
        # 3. Escalar el algoritmo al rango de precios
        min_algo = df['Algoritmo'].min()
        max_algo = df['Algoritmo'].max()
        algo_range = max_algo - min_algo
        price_range = max_price - min_price

        # Manejar el caso de datos planos (para evitar división por cero o resultados absurdos)
        if algo_range > 0 and price_range > 0:
            df['Algoritmo_Scaled'] = min_price + (df['Algoritmo'] - min_algo) / algo_range * price_range
        else:
            # Si los datos son planos, simplemente se utiliza el cierre
            df['Algoritmo_Scaled'] = df['Close']
            
        # 4. Líneas de Sobreventa/Sobrecompra escaladas al precio min/max (Min/Max Price)
        # La sobreventa va al precio mínimo y la sobrecompra al máximo, como se solicitó.
        df['Sobreventa_Scaled'] = min_price
        df['Sobrecompra_Scaled'] = max_price

        # 5. DataFrame para la trama: eliminar índice de fecha para ploteo secuencial (No Gaps)
        df_plot = df[['Open', 'High', 'Low', 'Close', 'Algoritmo_Scaled', 'Sobreventa_Scaled', 'Sobrecompra_Scaled', 'Proyectado']].reset_index(drop=True)


        # 6. Definir estilo para Grosor Mínimo (Minimum Thickness)
        mc = mpf.make_marketcolors(
            up='green', down='red',
            edge='inherit',      # Borde del cuerpo igual al color del cuerpo
            wick={'up':'green','down':'red'}, # Mecha del mismo color
            volume='in',
        )
        s = mpf.make_mpf_style(
            base_mpf_style='yahoo', 
            marketcolors=mc,
            # Reducir el grosor general para velas y líneas de addplot
            rc={'axes.linewidth': 0.5,      # Grosor del borde del gráfico
                'lines.linewidth': 1.0,     # Grosor por defecto de las líneas (las addplots usan este si no se especifica)
                'patch.linewidth': 0.5,     # Grosor del borde de los cuerpos de las velas (al mínimo)
                'axes.grid': True,
               },
        )
        
        # 7. Definir Addplots (Combine Plots, Remove Cierre Real, Dashed Projected)
        apds = [
            # Algoritmo Scalado (Línea azul del algoritmo)
            mpf.make_addplot(df_plot['Algoritmo_Scaled'], panel=0, color='blue', linewidth=1, label='Algoritmo'),
            
            # Sobreventa Scalado (Línea roja de sobreventa en el precio mínimo)
            mpf.make_addplot(df_plot['Sobreventa_Scaled'], panel=0, color='red', linestyle='-', linewidth=1, label='Sobreventa'),
            
            # Sobrecompra Scalado (Línea verde de sobrecompra en el precio máximo)
            mpf.make_addplot(df_plot['Sobrecompra_Scaled'], panel=0, color='green', linestyle='-', linewidth=1, label='Sobrecompra'),
            
            # Proyectado (Línea naranja A TRAZOS para el precio proyectado)
            mpf.make_addplot(df_plot['Proyectado'], panel=0, color='orange', linestyle='--', linewidth=1, label='Proyectado'),
        ]

        # 8. Generar el gráfico en memoria (Candlesticks only, Cierre Real removed)
        try:
            fig, axlist = mpf.plot(df_plot, 
                                   type='candle', # Tipo de gráfico de velas (Candlestick only)
                                   style=s, 
                                   addplot=apds, 
                                   ylabel=f'Precio {ticker} (EUR)', 
                                   title=f'Análisis Técnico de {ticker}', 
                                   volume=False, 
                                   x_axis_date=False, # Sin huecos (No Gaps)
                                   figratio=(10, 6), 
                                   returnfig=True)

            # Guardar el gráfico en un buffer de memoria
            buffer = BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            
            # Convertir la imagen a base64 para la API de Gemini
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
        except Exception as e:
            print(f"❌ Error al generar el gráfico de {ticker} con mplfinance: {e}")
            continue

        # --- LLAMADA A GEMINI Y ENVÍO DE EMAIL (SIN CAMBIOS) ---
        
        # Preparación de la imagen para Gemini
        image_part = {
            "inline_data": {
                "data": img_base64,
                "mime_type": 'image/png'
            }
        }
        
        # Texto del prompt
        prompt_text = f"""
        Analiza este gráfico de precios de {ticker} (últimos 90 días).
        
        El gráfico principal muestra velas japonesas y líneas escaladas al precio.
        
        - La línea **Azul** es el indicador del algoritmo escalado al rango de precios.
        - La línea **Roja** es la zona de sobreventa (escalada al precio mínimo).
        - La línea **Verde** es la zona de sobrecompra (escalada al precio máximo).
        - La línea **Naranja a trazos** es el precio proyectado.
        
        **Datos de hoy:**
        - Último Cierre: {ultimo_cierre:.2f}
        - Valor del Algoritmo: {algoritmo_hoy:.2f}
        - Precio Proyectado: {proyectado_manana:.2f}
        
        Genera un informe conciso y objetivo con la siguiente estructura y tono de experto:
        1. **Resumen de la Situación:** Descripción de la tendencia principal (alcista, bajista, lateral) basándose en las velas y la posición del precio respecto a las líneas de sobrecompra/sobreventa.
        2. **Análisis del Algoritmo:** Indica si el valor del algoritmo está en zona de sobrecompra (>75) o sobreventa (<25).
        3. **Conclusión y Proyección:** Resume la situación, mencionando la proyección de precio (línea naranja).
        
        El informe debe ser en español.
        """
        
        print(f"🧠 Enviando solicitud a Gemini para {ticker}...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[prompt_text, image_part],
                )
                
                # Cuerpo del email
                email_body_html = f"""
                <html>
                    <body style="font-family: Arial, sans-serif; line-height: 1.6;">
                        <h2>Análisis Técnico para {ticker}</h2>
                        <p>A continuación se presenta el informe generado por el modelo de IA y el gráfico de precios:</p>
                        <div style="border-left: 3px solid #007BFF; padding-left: 15px; background-color: #f7f7f7; padding: 10px; border-radius: 5px;">
                            {response.text.replace('\\n', '<br>')}
                        </div>
                        <p>El gráfico con el detalle de las velas y el algoritmo se adjunta a este correo.</p>
                        <p><i>Nota: Las líneas de sobreventa (roja) y sobrecompra (verde) están escaladas al precio mínimo y máximo del periodo, y la línea azul es el algoritmo escalado al mismo rango.</i></p>
                    </body>
                </html>
                """

                # Enviar el email
                enviar_email_con_adjunto(
                    to_email,
                    f"Informe Diario de Análisis Técnico: {ticker}",
                    email_body_html,
                    buffer.getvalue(),
                    f"analisis_tecnico_{ticker}_{datetime.now().strftime('%Y%m%d')}.png"
                )
                break  # Salir del bucle si el envío es exitoso
                
            except Exception as e:
                print(f"❌ Error en Gemini o al enviar email para {ticker}: {e}. Reintentando ({attempt + 1}/{max_retries}).")
                if attempt == max_retries - 1:
                    print(f"❌ Error persistente para {ticker} después de {max_retries} reintentos.")
            
        print(f"⏳ Esperando 180 segundos antes de procesar el siguiente ticker...")
        time.sleep(180)


# --- FUNCIÓN MAIN (SIN CAMBIOS) ---

def main():
    try:
        all_tickers = leer_google_sheets()[1:]
    except Exception as e:
        print(f"❌ Error al leer Google Sheets: {e}. Asegúrate de que las variables de entorno están configuradas correctamente y el archivo JSON de credenciales es válido.")
        return
    
    if not all_tickers:
        print("No hay tickers para procesar.")
        return

    day_of_week = datetime.today().weekday()
    
    # Número de tickers a procesar por día (Ej: 12)
    num_tickers_per_day = 12  
    total_tickers_in_sheet = len(all_tickers)
    
    start_index = (day_of_week * num_tickers_per_day) % total_tickers_in_sheet
    
    end_index = start_index + num_tickers_per_day
    
    tickers_for_today = []
    if end_index <= total_tickers_in_sheet:
        tickers_for_today = all_tickers[start_index:end_index]
    else:
        # Envolver al inicio de la lista si se excede el final
        tickers_for_today = all_tickers[start_index:] + all_tickers[:end_index - total_tickers_in_sheet]

    if tickers_for_today:
        print(f"Procesando tickers para el día {datetime.today().strftime('%A')}: {tickers_for_today}")
        generar_contenido_con_gemini(tickers_for_today)
    else:
        print(f"No hay tickers asignados para el día {datetime.today().strftime('%A')}.")

if __name__ == '__main__':
    # Se añade la importación de base64 y la llamada a main
    import base64
    main()
