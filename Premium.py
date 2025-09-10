import os
import json
import smtplib
import yfinance as yf
import google.generativeai as genai
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
import re
import random
from email.mime.base import MIMEBase
from email import encoders

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

    range_name = 'A:A'
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    values = result.get('values', [])

    if not values:
        print('No se encontraron datos.')
        return []
    else:
        print('Datos leídos de la hoja:')
        for row in values:
            print(row)
        # Limpiar y aplanar la lista
        tickers = [item.strip() for sublist in values for item in sublist if item.strip()]
        return tickers

def obtener_datos_yfinance(ticker):
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="1mo")
        precio_actual = hist['Close'].iloc[-1]
        
        # Simulación de las tendencias y precios de aplanamiento
        tendencia = random.choice(["Bajando", "Subiendo"])
        sobre = random.choice(["Sobrecompra", "Sobreventa", "Intermedio"])
        
        if tendencia == "Bajando" and sobre == "Sobrecompra":
            oportunidad = "Riesgo de Venta Activada"
            precio_aplana = precio_actual * 1.1 # Ejemplo de precio de aplanamiento
        elif tendencia == "Bajando":
            oportunidad = "Posibilidad de Compra"
            precio_aplana = precio_actual * 0.9 # Ejemplo de precio de aplanamiento
        elif tendencia == "Subiendo" and sobre == "Sobreventa":
            oportunidad = "Posibilidad de Compra Activada"
            precio_aplana = precio_actual * 0.95 # Ejemplo de precio de aplanamiento
        elif tendencia == "Subiendo" and sobre == "Sobrecompra":
            oportunidad = "Riesgo de Venta"
            precio_aplana = precio_actual * 1.05 # Ejemplo de precio de aplanamiento
        elif tendencia == "Subiendo":
            oportunidad = "Seguirá Subiendo"
            precio_aplana = precio_actual * 1.02 # Ejemplo de precio de aplanamiento
        else:
            oportunidad = "Sin Categoría"
            precio_aplana = "N/A"

        return {
            'TICKER': ticker,
            'PRECIO_ACTUAL': precio_actual,
            'PRECIO_APLANAMIENTO': precio_aplana,
            'TENDENCIA': tendencia,
            'SOBRE': sobre,
            'OPORTUNIDAD': oportunidad
        }
    except Exception as e:
        print(f"❌ Error al obtener datos para {ticker}: {e}")
        return None

def enviar_email_con_adjunto(cuerpo_html, asunto):
    remitente_email = os.getenv('REMITENTE_EMAIL')
    remitente_password = os.getenv('REMITENTE_PASSWORD')
    destinatario_email = os.getenv('DESTINATARIO_EMAIL')
    
    if not all([remitente_email, remitente_password, destinatario_email]):
        print("❌ Faltan credenciales de correo electrónico en las variables de entorno.")
        return

    mensaje = MIMEMultipart("alternative")
    mensaje["Subject"] = asunto
    mensaje["From"] = remitente_email
    mensaje["To"] = destinatario_email
    
    part_html = MIMEText(cuerpo_html, "html")
    mensaje.attach(part_html)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as servidor:
            servidor.login(remitente_email, remitente_password)
            servidor.sendmail(remitente_email, destinatario_email, mensaje.as_string())
        print("✅ Correo electrónico enviado correctamente.")
    except Exception as e:
        print(f"❌ Error al enviar el correo electrónico: {e}")

def generar_reporte():
    try:
        tickers = leer_google_sheets()
        if not tickers:
            return

        datos_completos = []
        for ticker in tickers:
            print(f"Buscando datos de {ticker}...")
            data = obtener_datos_yfinance(ticker)
            if data:
                datos_completos.append(data)
        
        # --- Lógica de clasificación y ordenación ---
        def obtener_clave_ordenacion(empresa):
            categoria = empresa['OPORTUNIDAD']
            
            # Asignar un valor numérico a cada categoría para el orden principal
            orden_grupo = 0
            if categoria == "Posibilidad de Compra":
                orden_grupo = 1
                # Para este grupo, la ordenación interna es por el menor % de subida
                if empresa['PRECIO_APLANAMIENTO'] != "N/A":
                    # Calcular el porcentaje de subida
                    precio_actual = empresa['PRECIO_ACTUAL']
                    precio_compra_si = empresa['PRECIO_APLANAMIENTO']
                    porcentaje = ((precio_compra_si - precio_actual) / precio_actual) * 100
                    return (orden_grupo, porcentaje)
                else:
                    return (orden_grupo, float('inf'))
            elif categoria == "Posibilidad de Compra Activada":
                orden_grupo = 2
                # Para este grupo, la ordenación interna es por el mayor % de bajada
                if empresa['PRECIO_APLANAMIENTO'] != "N/A":
                    # Calcular el porcentaje de bajada
                    precio_actual = empresa['PRECIO_ACTUAL']
                    precio_vende_si = empresa['PRECIO_APLANAMIENTO']
                    porcentaje = ((precio_vende_si - precio_actual) / precio_actual) * 100
                    return (orden_grupo, -porcentaje)
                else:
                    return (orden_grupo, float('-inf'))
            elif categoria == "Seguirá Subiendo":
                orden_grupo = 3
                # Ordenar como el grupo 2 (mayor % de bajada)
                if empresa['PRECIO_APLANAMIENTO'] != "N/A":
                    precio_actual = empresa['PRECIO_ACTUAL']
                    precio_vende_si = empresa['PRECIO_APLANAMIENTO']
                    porcentaje = ((precio_vende_si - precio_actual) / precio_actual) * 100
                    return (orden_grupo, -porcentaje)
                else:
                    return (orden_grupo, float('-inf'))
            elif categoria == "Riesgo de Venta":
                orden_grupo = 4
                # Ordenar como el grupo 2 (mayor % de bajada)
                if empresa['PRECIO_APLANAMIENTO'] != "N/A":
                    precio_actual = empresa['PRECIO_ACTUAL']
                    precio_vende_si = empresa['PRECIO_APLANAMIENTO']
                    porcentaje = ((precio_vende_si - precio_actual) / precio_actual) * 100
                    return (orden_grupo, -porcentaje)
                else:
                    return (orden_grupo, float('-inf'))
            elif categoria == "Riesgo de Venta Activada":
                orden_grupo = 5
                # Ordenar como el grupo 1 (menor % de subida)
                if empresa['PRECIO_APLANAMIENTO'] != "N/A":
                    precio_actual = empresa['PRECIO_ACTUAL']
                    precio_compra_si = empresa['PRECIO_APLANAMIENTO']
                    porcentaje = ((precio_compra_si - precio_actual) / precio_actual) * 100
                    return (orden_grupo, porcentaje)
                else:
                    return (orden_grupo, float('inf'))
            else:
                return (99, empresa['TICKER']) # Cualquier otro caso al final

        datos_ordenados = sorted(datos_completos, key=obtener_clave_ordenacion)
        # --- Fin de la lógica de ordenación ---

        # Código HTML para el reporte
        html_body = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Alerta de Oportunidades IBEXIA</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; color: #333; margin: 0; padding: 20px; }}
                h1 {{ color: #004085; text-align: center; margin-bottom: 30px; }}
                .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); max-width: 90%; margin: auto; overflow-x: auto; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; }}
                th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #0056b3; color: #fff; text-transform: uppercase; font-weight: bold; }}
                tr:hover {{ background-color: #f1f1f1; }}
                .tag {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
                .buy-opportunity {{ background-color: #d4edda; color: #155724; }}
                .active-buy {{ background-color: #cce5ff; color: #004085; }}
                .stable-up {{ background-color: #fff3cd; color: #856404; }}
                .sell-risk {{ background-color: #f8d7da; color: #721c24; }}
                .active-sell {{ background-color: #d1ecf1; color: #0c5460; }}
                .no-category {{ background-color: #e2e3e5; color: #6c757d; }}
            </style>
        </head>
        <body>
            <h1>Alerta de Oportunidades IBEXIA</h1>
            <div class="container">
                <table>
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th>Oportunidad</th>
                            <th>Tendencia</th>
                            <th>Estado</th>
                            <th>Precio Actual</th>
                            <th>Precio Aplanamiento</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(f"""
                        <tr>
                            <td>{empresa['TICKER']}</td>
                            <td><span class="tag {'buy-opportunity' if empresa['OPORTUNIDAD'] == 'Posibilidad de Compra' else 'active-buy' if empresa['OPORTUNIDAD'] == 'Posibilidad de Compra Activada' else 'stable-up' if empresa['OPORTUNIDAD'] == 'Seguirá Subiendo' else 'sell-risk' if empresa['OPORTUNIDAD'] == 'Riesgo de Venta' else 'active-sell' if empresa['OPORTUNIDAD'] == 'Riesgo de Venta Activada' else 'no-category'}">{empresa['OPORTUNIDAD']}</span></td>
                            <td>{empresa['TENDENCIA']}</td>
                            <td>{empresa['SOBRE']}</td>
                            <td>{empresa['PRECIO_ACTUAL']:.2f}</td>
                            <td>{empresa['PRECIO_APLANAMIENTO']:.2f}</td>
                        </tr>
                        """ for empresa in datos_ordenados if empresa['OPORTUNIDAD'] != "Sin Categoría")}
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """
        
        asunto = f"🔔 Alertas y Oportunidades IBEXIA: {len(datos_completos)} oportunidades detectadas hoy {datetime.today().strftime('%d/%m/%Y')}"
        enviar_email_con_adjunto(html_body, asunto)

    except Exception as e:
        print(f"❌ Error al ejecutar el script principal: {e}")

if __name__ == '__main__':
    generar_reporte()
