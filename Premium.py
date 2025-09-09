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
    else:
        print('Datos leídos de la hoja:')
        for row in values:
            print(row)
    return [row[0] for row in values if row]

def formatear_numero(numero):
    if pd.isna(numero) or numero is None:
        return "N/A"
    try:
        num = float(numero)
        return f"{num:,.3f}"
    except (ValueError, TypeError):
        return "N/A"
        
def calculate_smi_tv(df):
    high = df['High']
    low = df['Low']
    close = df['Close']
    length_k = 10
    length_d = 3
    ema_signal_len = 10
    smooth_period = 5
    hh = high.rolling(window=length_k).max()
    ll = low.rolling(window=length_k).min()
    diff = hh - ll
    rdiff = close - (hh + ll) / 2
    avgrel = rdiff.ewm(span=length_d, adjust=False).mean()
    avgdiff = diff.ewm(span=length_d, adjust=False).mean()
    epsilon = 1e-9
    smi_raw = np.where(
        (avgdiff / 2 + epsilon) != 0,
        (avgrel / (avgdiff / 2 + epsilon)) * 100,
        0.0
    )
    smi_raw = np.clip(smi_raw, -100, 100)
    smi_smoothed = pd.Series(smi_raw, index=df.index).rolling(window=smooth_period).mean()
    smi_signal = smi_smoothed.ewm(span=ema_signal_len, adjust=False).mean()
    df['SMI'] = smi_smoothed
    return df

def calcular_precio_aplanamiento(df):
    try:
        if len(df) < 3:
            return "N/A"

        length_d = 3
        smooth_period = 5

        df_prev = df.iloc[:-1].copy()
        df_prev = calculate_smi_tv(df_prev)

        smi_smoothed_prev = df['SMI'].iloc[-2]

        avgrel_prev_last = (df_prev['Close'] - (df_prev['High'].rolling(window=10).max() + df_prev['Low'].rolling(window=10).min()) / 2).ewm(span=length_d, adjust=False).mean().iloc[-1]
        avgdiff_prev_last = (df_prev['High'].rolling(window=10).max() - df_prev['Low'].rolling(window=10).min()).ewm(span=length_d, adjust=False).mean().iloc[-1]

        alpha_ema = 2 / (length_d + 1)
        
        df_temp = df.copy()
        df_temp['SMI'] = pd.Series(df_temp['SMI'], index=df_temp.index).rolling(window=smooth_period).mean()
        smi_raw_yesterday = df_temp['SMI'].iloc[-2]

        hh_today = df['High'].rolling(window=10).max().iloc[-1]
        ll_today = df['Low'].rolling(window=10).min().iloc[-1]
        diff_today = hh_today - ll_today
        
        avgdiff_today = (1 - alpha_ema) * avgdiff_prev_last + alpha_ema * diff_today
        
        avgrel_today_target = (smi_raw_yesterday / 100) * (avgdiff_today / 2)
        
        rdiff_today_target = (avgrel_today_target - (1 - alpha_ema) * avgrel_prev_last) / alpha_ema
        
        close_target = rdiff_today_target + (hh_today + ll_today) / 2
        
        return close_target

    except Exception as e:
        print(f"❌ Error en el cálculo de precio de aplanamiento: {e}")
        return "N/A"

def obtener_datos_yfinance(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Filtro para empresas sin precio actual
        current_price = info.get("currentPrice")
        if not current_price:
            print(f"⚠️ Advertencia: No se encontró precio actual para {ticker}. Saltando...")
            return None

        hist_extended = stock.history(period="60d", interval="1d")
        if hist_extended.empty:
            print(f"⚠️ Advertencia: No se encontraron datos históricos para {ticker}. Saltando...")
            return None
        hist_extended = calculate_smi_tv(hist_extended)

        smi_series = hist_extended['SMI'].dropna()
        if len(smi_series) < 2:
            print(f"⚠️ Advertencia: No hay suficientes datos de SMI para {ticker}. Saltando...")
            return None
        
        smi_yesterday = smi_series.iloc[-2]
        smi_today = smi_series.iloc[-1]
        
        pendiente_hoy = smi_today - smi_yesterday
        
        tendencia_hoy = "Subiendo" if pendiente_hoy > 0.1 else ("Bajando" if pendiente_hoy < -0.1 else "Plano")
        
        estado_smi = "Sobrecompra" if smi_today > 40 else ("Sobreventa" if smi_today < -40 else "Intermedio")
        
        precio_aplanamiento = calcular_precio_aplanamiento(hist_extended)
        
        # --- Lógica para determinar la última acción de compra/venta y su fecha
        comprado_status = "N/A"
        precio_compra = "N/A"
        fecha_compra = "N/A"
        
        # Copiamos la serie de SMI para evitar modificar el original
        smi_series_copy = hist_extended['SMI'].copy()
        pendientes_smi = smi_series_copy.diff()
        
        # Recorremos el historial desde el final para encontrar la última acción
        for i in range(len(hist_extended) - 1, 0, -1):
            smi_prev = hist_extended['SMI'].iloc[i - 1]
            pendiente_prev = pendientes_smi.iloc[i - 1]
            pendiente_curr = pendientes_smi.iloc[i]
            
            # Señal de venta: cambio de tendencia de positiva a negativa
            if pendiente_curr < 0 and pendiente_prev >= 0:
                comprado_status = "NO"
                precio_compra = hist_extended['Close'].iloc[i-1]
                fecha_compra = hist_extended.index[i-1].strftime('%d/%m/%Y')
                break # Encontrada la última acción, salimos del bucle
            
            # Señal de compra: cambio de tendencia de negativa a positiva en zona de sobreventa o intermedia
            elif pendiente_curr > 0 and pendiente_prev <= 0 and smi_prev < 40:
                comprado_status = "SI"
                precio_compra = hist_extended['Close'].iloc[i-1]
                fecha_compra = hist_extended.index[i-1].strftime('%d/%m/%Y')
                break # Encontrada la última acción, salimos del bucle

        return {
            "TICKER": ticker,
            "NOMBRE_EMPRESA": info.get("longName", ticker),
            "PRECIO_ACTUAL": current_price,
            "SMI_AYER": smi_yesterday,
            "SMI_HOY": smi_today,
            "TENDENCIA_ACTUAL": tendencia_hoy,
            "ESTADO_SMI": estado_smi,
            "PRECIO_APLANAMIENTO": precio_aplanamiento,
            "PENDIENTE": pendiente_hoy,
            "COMPRADO": comprado_status,
            "PRECIO_COMPRA": formatear_numero(precio_compra),
            "FECHA_COMPRA": fecha_compra,
        }

    except Exception as e:
        print(f"❌ Error al obtener datos de {ticker}: {e}. Saltando a la siguiente empresa...")
        return None

def clasificar_empresa(data):
    estado_smi = data['ESTADO_SMI']
    tendencia = data['TENDENCIA_ACTUAL']
    precio_aplanamiento = data['PRECIO_APLANAMIENTO']

    if estado_smi == "Sobreventa":
        if tendencia == "Subiendo":
            data['OPORTUNIDAD'] = "Posibilidad de Compra Activada"
            data['COMPRA_SI'] = "COMPRA AHORA"
            data['VENDE_SI'] = "NO VENDAS"
            data['ORDEN_PRIORIDAD'] = 1
        elif tendencia == "Bajando":
            data['OPORTUNIDAD'] = "Posibilidad de Compra"
            data['COMPRA_SI'] = f"COMPRA si supera {formatear_numero(precio_aplanamiento)}€ ⬆️"
            data['VENDE_SI'] = "NO VENDAS"
            data['ORDEN_PRIORIDAD'] = 2
        else: # Tendencia Plana
            data['OPORTUNIDAD'] = "Intermedio"
            data['COMPRA_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['VENDE_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['ORDEN_PRIORIDAD'] = 99 # Baja prioridad
    
    elif estado_smi == "Intermedio":
        if tendencia == "Bajando":
            data['OPORTUNIDAD'] = "Seguirá bajando"
            data['COMPRA_SI'] = f"COMPRA si supera {formatear_numero(precio_aplanamiento)}€ ⬆️"
            data['VENDE_SI'] = "YA ES TARDE PARA VENDER"
            data['ORDEN_PRIORIDAD'] = 4
        elif tendencia == "Subiendo":
            data['OPORTUNIDAD'] = "Seguirá subiendo"
            data['COMPRA_SI'] = "YA ES TARDE PARA COMPRAR"
            data['VENDE_SI'] = f"VENDE si baja de {formatear_numero(precio_aplanamiento)}€ ⬇️"
            data['ORDEN_PRIORIDAD'] = 3
        else: # Tendencia Plana
            data['OPORTUNIDAD'] = "Intermedio"
            data['COMPRA_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['VENDE_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['ORDEN_PRIORIDAD'] = 99 # Baja prioridad
            
    elif estado_smi == "Sobrecompra":
        if tendencia == "Subiendo":
            data['OPORTUNIDAD'] = "Riesgo de Venta"
            data['COMPRA_SI'] = "NO COMPRES"
            data['VENDE_SI'] = f"VENDE si baja de {formatear_numero(precio_aplanamiento)}€ ⬇️"
            data['ORDEN_PRIORIDAD'] = 5
        elif tendencia == "Bajando":
            data['OPORTUNIDAD'] = "Riesgo de Venta Activada"
            data['COMPRA_SI'] = "NO COMPRES"
            data['VENDE_SI'] = "VENDE AHORA"
            data['ORDEN_PRIORIDAD'] = 6
        else: # Tendencia Plana
            data['OPORTUNIDAD'] = "Intermedio"
            data['COMPRA_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['VENDE_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['ORDEN_PRIORIDAD'] = 99 # Baja prioridad

    return data

def enviar_email_con_adjunto(html_body, asunto_email):
    remitente = "xumkox@gmail.com"
    destinatario = "xumkox@gmail.com"
    password = "kdgz lvdo wqvt vfkt"
    msg = MIMEMultipart()
    msg['From'] = remitente
    msg['To'] = destinatario
    msg['Subject'] = asunto_email

    # Crea el archivo HTML
    html_filename = "analisis-empresas.html"
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(html_body)

    # Adjunta el archivo HTML al correo
    with open(html_filename, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f"attachment; filename={html_filename}",
    )
    msg.attach(part)

    try:
        servidor = smtplib.SMTP('smtp.gmail.com', 587)
        servidor.starttls()
        servidor.login(remitente, password)
        servidor.sendmail(remitente, destinatario, msg.as_string())
        servidor.quit()
        print(f"✅ Correo enviado con el asunto: {asunto_email}")
        # Elimina el archivo después de enviarlo
        os.remove(html_filename)
    except Exception as e:
        print("❌ Error al enviar el correo:", e)

def generar_reporte():
    try:
        all_tickers = leer_google_sheets()[1:]
        if not all_tickers:
            print("No hay tickers para procesar.")
            return

        datos_completos = []
        for ticker in all_tickers:
            print(f"🔎 Analizando {ticker}...")
            data = obtener_datos_yfinance(ticker)
            if data:
                datos_completos.append(clasificar_empresa(data))
            time.sleep(1)

        # Ordenar la lista alfabéticamente por el nombre de la empresa
        datos_completos.sort(key=lambda x: x.get('NOMBRE_EMPRESA', ''))
        
        # Filtrar para no mostrar las que tienen orden 99 (no relevantes)
        datos_completos = [d for d in datos_completos if d.get('ORDEN_PRIORIDAD') != 99]

        html_body = f"""
        <html>
        <head>
            <title>Resumen Diario de Oportunidades - {datetime.today().strftime('%d/%m/%Y')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                h2 {{ color: #2c3e50; }}
                p {{ color: #7f8c8d; }}
                #search-container {{ margin-bottom: 20px; }}
                #searchInput {{
                    width: 100%;
                    padding: 10px;
                    font-size: 16px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    box-sizing: border-box;
                }}
                .table-container {{
                    overflow-x: auto;
                    overflow-y: auto;
                }}
                table {{ 
                    width: 800px;
                    min-width: 800px;
                    border-collapse: collapse; 
                    margin-top: 20px; 
                }}
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 8px; 
                    text-align: left;
                    vertical-align: top;
                    white-space: normal;
                }}
                th {{ background-color: #f2f2f2; }}
                .compra {{ color: #1abc9c; font-weight: bold; }}
                .venta {{ color: #e74c3c; font-weight: bold; }}
                .comprado-si {{ background-color: #2ecc71; color: white; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h2>Resumen Diario de Oportunidades - {datetime.today().strftime('%d/%m/%Y')}</h2>
            
            <p>Se ha generado un resumen de las empresas según su estado y tendencia del SMI. La tabla está ordenada alfabéticamente. Usa el buscador para encontrar una empresa rápidamente.</p>
            
            <div id="search-container">
                <input type="text" id="searchInput" onkeyup="filterTable()" placeholder="Buscar por nombre de empresa...">
            </div>
            
            <div id="scroll-top" style="overflow-x: auto;">
                <div style="width: 800px;">&nbsp;</div>
            </div>
            
            <div class="table-container">
                <table id="myTable">
                    <thead>
                        <tr>
                            <th>Empresa (Precio)</th>
                            <th>¿Estamos comprados?</th>
                            <th>Precio de compra</th>
                            <th>Fecha de compra</th>
                            <th>Tendencia Actual</th>
                            <th>Oportunidad</th>
                            <th>Compra si...</th>
                            <th>Vende si...</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        if not datos_completos:
            html_body += """
                        <tr><td colspan="8">No se encontraron empresas con oportunidades claras hoy.</td></tr>
            """
        else:
            for data in datos_completos:
                nombre_con_precio = f"<b>{data['NOMBRE_EMPRESA']}</b> ({formatear_numero(data['PRECIO_ACTUAL'])}€)"
                
                oportunidad = data['OPORTUNIDAD']
                clase_oportunidad = "compra" if "compra" in oportunidad.lower() else ("venta" if "venta" in oportunidad.lower() else "")

                precio_compra_display = f"{data['PRECIO_COMPRA']}€" if data['COMPRADO'] == 'SI' and data['PRECIO_COMPRA'] != 'N/A' else ''
                fecha_compra_display = data['FECHA_COMPRA'] if data['COMPRADO'] == 'SI' else ''
                
                comprado_class = "comprado-si" if data['COMPRADO'] == 'SI' else ''

                html_body += f"""
                        <tr>
                            <td>{nombre_con_precio}</td>
                            <td class="{comprado_class}">{data['COMPRADO']}</td>
                            <td>{precio_compra_display}</td>
                            <td>{fecha_compra_display}</td>
                            <td>{data['TENDENCIA_ACTUAL']}</td>
                            <td class="{clase_oportunidad}">{oportunidad}</td>
                            <td>{data['COMPRA_SI']}</td>
                            <td>{data['VENDE_SI']}</td>
                        </tr>
                """
        
        html_body += """
                    </tbody>
                </table>
            </div>
            
            <br>
            <p><strong>Aviso:</strong> El algoritmo de trading se basa en indicadores técnicos y no garantiza la rentabilidad. Utiliza esta información con tu propio análisis y criterio. ¡Feliz trading!</p>

            <script>
                function filterTable() {
                    var input, filter, table, tr, td, i, txtValue;
                    input = document.getElementById("searchInput");
                    filter = input.value.toUpperCase();
                    table = document.getElementById("myTable");
                    tr = table.getElementsByTagName("tr");
                    for (i = 0; i < tr.length; i++) {
                        td = tr[i].getElementsByTagName("td")[0];
                        if (td) {
                            txtValue = td.textContent || td.innerText;
                            if (txtValue.toUpperCase().indexOf(filter) > -1) {
                                tr[i].style.display = "";
                            } else {
                                tr[i].style.display = "none";
                            }
                        }
                    }
                }
                
                // Sincronizar el scroll de las dos barras
                const tableContainer = document.querySelector('.table-container');
                const scrollTop = document.getElementById('scroll-top');
                
                scrollTop.addEventListener('scroll', () => {
                    tableContainer.scrollLeft = scrollTop.scrollLeft;
                });
                
                tableContainer.addEventListener('scroll', () => {
                    scrollTop.scrollLeft = tableContainer.scrollLeft;
                });
            </script>
        </body>
        </html>
        """
        
        asunto = f"🔔 Alertas y Oportunidades IBEXIA: {len(datos_completos)} oportunidades detectadas hoy {datetime.today().strftime('%d/%m/%Y')}"
        enviar_email_con_adjunto(html_body, asunto)

    except Exception as e:
        print(f"❌ Error al ejecutar el script principal: {e}")

if __name__ == '__main__':
    generar_reporte()
