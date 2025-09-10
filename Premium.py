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

        avgrel_prev_last = (df_prev['Close'] - (df_prev['High'].rolling(window=10).max() + df_prev['Low'].rolling(window=10).min()) / 2).ewm(span=length_d, adjust=False).mean().iloc[-1]
        avgdiff_prev_last = (df_prev['High'].rolling(window=10).max() - df_prev['Low'].rolling(window=10).min()).ewm(span=length_d, adjust=False).mean().iloc[-1]
        smi_raw_yesterday = df['SMI'].iloc[-2]

        alpha_ema = 2 / (length_d + 1)
        
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
        
def calcular_beneficio_perdida(precio_compra, precio_actual, inversion=10000):
    try:
        precio_compra = float(precio_compra)
        precio_actual = float(precio_actual)
        
        if precio_compra <= 0 or precio_actual <= 0:
            return "N/A"

        acciones = inversion / precio_compra
        beneficio_perdida = (precio_actual - precio_compra) * acciones
        return f"{beneficio_perdida:,.2f}"
    except (ValueError, TypeError):
        return "N/A"

def obtener_datos_yfinance(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
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
        
        comprado_status = "NO"
        precio_compra = "N/A"
        fecha_compra = "N/A"
        
        smi_series_copy = hist_extended['SMI'].copy()
        pendientes_smi = smi_series_copy.diff()
        
        for i in range(len(hist_extended) - 1, 0, -1):
            smi_prev = hist_extended['SMI'].iloc[i - 1]
            pendiente_prev = pendientes_smi.iloc[i - 1]
            pendiente_curr = pendientes_smi.iloc[i]
            
            if pendiente_curr < 0 and pendiente_prev >= 0:
                comprado_status = "NO"
                precio_compra = hist_extended['Close'].iloc[i-1]
                fecha_compra = hist_extended.index[i-1].strftime('%d/%m/%Y')
                break
            
            elif pendiente_curr > 0 and pendiente_prev <= 0 and smi_prev < 40:
                comprado_status = "SI"
                precio_compra = hist_extended['Close'].iloc[i-1]
                fecha_compra = hist_extended.index[i-1].strftime('%d/%m/%Y')
                break

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
            "PRECIO_COMPRA": precio_compra,
            "FECHA_COMPRA": fecha_compra,
        }

    except Exception as e:
        print(f"❌ Error al obtener datos de {ticker}: {e}. Saltando a la siguiente empresa...")
        return None

def clasificar_empresa(data, hist_df):
    estado_smi = data['ESTADO_SMI']
    tendencia = data['TENDENCIA_ACTUAL']
    precio_aplanamiento = data['PRECIO_APLANAMIENTO']
    smi_actual = data['SMI_HOY']

    prioridad = {
        "Posibilidad de Compra Activada": 1,
        "Posibilidad de Compra": 2,
        "Seguirá subiendo": 3,
        "Seguirá bajando": 4,
        "Riesgo de Venta": 5,
        "Riesgo de Venta Activada": 6,
        "Intermedio": 99
    }

    if estado_smi == "Sobreventa":
        if tendencia == "Subiendo":
            data['OPORTUNIDAD'] = "Posibilidad de Compra Activada"
            data['COMPRA_SI'] = "COMPRA AHORA"
            data['VENDE_SI'] = "NO VENDAS"
            data['ORDEN_PRIORIDAD'] = prioridad["Posibilidad de Compra Activada"]
        elif tendencia == "Bajando":
            data['OPORTUNIDAD'] = "Posibilidad de Compra"
            data['COMPRA_SI'] = f"COMPRA si supera {formatear_numero(precio_aplanamiento)}€ ⬆️"
            data['VENDE_SI'] = "NO VENDAS"
            data['ORDEN_PRIORIDAD'] = prioridad["Posibilidad de Compra"]
        else:
            data['OPORTUNIDAD'] = "Intermedio"
            data['COMPRA_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['VENDE_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['ORDEN_PRIORIDAD'] = prioridad["Intermedio"]
    
    elif estado_smi == "Intermedio":
        if tendencia == "Bajando":
            data['OPORTUNIDAD'] = "Seguirá bajando"
            data['COMPRA_SI'] = f"COMPRA si supera {formatear_numero(precio_aplanamiento)}€ ⬆️"
            data['VENDE_SI'] = "YA ES TARDE PARA VENDER"
            data['ORDEN_PRIORIDAD'] = prioridad["Seguirá bajando"]
        elif tendencia == "Subiendo":
            data['OPORTUNIDAD'] = "Seguirá subiendo"
            data['COMPRA_SI'] = "YA ES TARDE PARA COMPRAR"
            data['VENDE_SI'] = f"VENDE si baja de {formatear_numero(precio_aplanamiento)}€ ⬇️"
            data['ORDEN_PRIORIDAD'] = prioridad["Seguirá subiendo"]
        else:
            data['OPORTUNIDAD'] = "Intermedio"
            data['COMPRA_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['VENDE_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['ORDEN_PRIORIDAD'] = prioridad["Intermedio"]
            
    elif estado_smi == "Sobrecompra":
        if tendencia == "Subiendo":
            data['OPORTUNIDAD'] = "Riesgo de Venta"
            data['COMPRA_SI'] = "NO COMPRES"
            data['VENDE_SI'] = f"VENDE si baja de {formatear_numero(precio_aplanamiento)}€ ⬇️"
            data['ORDEN_PRIORIDAD'] = prioridad["Riesgo de Venta"]
        elif tendencia == "Bajando":
            data['OPORTUNIDAD'] = "Riesgo de Venta Activada"
            data['COMPRA_SI'] = "NO COMPRES"
            data['VENDE_SI'] = "VENDE AHORA"
            data['ORDEN_PRIORIDAD'] = prioridad["Riesgo de Venta Activada"]
        else:
            data['OPORTUNIDAD'] = "Intermedio"
            data['COMPRA_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['VENDE_SI'] = "NO PREVEEMOS GIRO EN ESTOS MOMENTOS"
            data['ORDEN_PRIORIDAD'] = prioridad["Intermedio"]
    
    return data

def enviar_email_con_adjunto(html_body, asunto_email):
    remitente = "xumkox@gmail.com"
    destinatario = "xumkox@gmail.com"
    password = "kdgz lvdo wqvt vfkt"
    msg = MIMEMultipart()
    msg['From'] = remitente
    msg['To'] = destinatario
    msg['Subject'] = asunto_email

    html_filename = "analisis-empresas.html"
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(html_body)

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
            
            try:
                stock = yf.Ticker(ticker)
                hist_df = stock.history(period="60d", interval="1d")
                if hist_df.empty:
                    print(f"⚠️ Advertencia: No se encontraron datos históricos para {ticker}. Saltando...")
                    continue
                hist_df = calculate_smi_tv(hist_df)
                
                data = obtener_datos_yfinance(ticker)
                if data:
                    datos_completos.append(clasificar_empresa(data, hist_df))
            except Exception as e:
                print(f"❌ Error al procesar {ticker}: {e}. Saltando a la siguiente empresa...")
                continue
                
            time.sleep(1)

        # --- Lógica de ordenación integrada ---
        def obtener_clave_ordenacion(empresa):
            categoria = empresa['OPORTUNIDAD']
            
            orden_grupo = 99
            orden_interna = float('inf')
            
            if categoria == "Posibilidad de Compra" and empresa['TENDENCIA_ACTUAL'] == "Bajando":
                orden_grupo = 1
                if empresa['PRECIO_APLANAMIENTO'] != "N/A" and empresa['PRECIO_ACTUAL'] is not None:
                    try:
                        precio_compra = float(empresa['PRECIO_APLANAMIENTO'])
                        precio_actual = float(empresa['PRECIO_ACTUAL'])
                        porcentaje = ((precio_compra - precio_actual) / precio_actual) * 100
                        orden_interna = porcentaje
                    except (ValueError, TypeError):
                        pass

            elif categoria == "Posibilidad de Compra Activada" and empresa['TENDENCIA_ACTUAL'] == "Subiendo" and empresa['ESTADO_SMI'] == "Sobreventa":
                orden_grupo = 2
                if empresa['PRECIO_APLANAMIENTO'] != "N/A" and empresa['PRECIO_ACTUAL'] is not None:
                    try:
                        precio_vende = float(empresa['PRECIO_APLANAMIENTO'])
                        precio_actual = float(empresa['PRECIO_ACTUAL'])
                        porcentaje = ((precio_vende - precio_actual) / precio_actual) * 100
                        orden_interna = -porcentaje
                    except (ValueError, TypeError):
                        pass
            
            elif categoria == "Seguirá subiendo" and empresa['TENDENCIA_ACTUAL'] == "Subiendo" and empresa['ESTADO_SMI'] == "Intermedio":
                orden_grupo = 3
                if empresa['PRECIO_APLANAMIENTO'] != "N/A" and empresa['PRECIO_ACTUAL'] is not None:
                    try:
                        precio_vende = float(empresa['PRECIO_APLANAMIENTO'])
                        precio_actual = float(empresa['PRECIO_ACTUAL'])
                        porcentaje = ((precio_vende - precio_actual) / precio_actual) * 100
                        orden_interna = -porcentaje
                    except (ValueError, TypeError):
                        pass
            
            elif categoria == "Riesgo de Venta" and empresa['TENDENCIA_ACTUAL'] == "Subiendo" and empresa['ESTADO_SMI'] == "Sobrecompra":
                orden_grupo = 4
                if empresa['PRECIO_APLANAMIENTO'] != "N/A" and empresa['PRECIO_ACTUAL'] is not None:
                    try:
                        precio_vende = float(empresa['PRECIO_APLANAMIENTO'])
                        precio_actual = float(empresa['PRECIO_ACTUAL'])
                        porcentaje = ((precio_vende - precio_actual) / precio_actual) * 100
                        orden_interna = -porcentaje
                    except (ValueError, TypeError):
                        pass
            
            elif categoria == "Riesgo de Venta Activada" and empresa['TENDENCIA_ACTUAL'] == "Bajando" and empresa['ESTADO_SMI'] == "Sobrecompra":
                orden_grupo = 5
                if empresa['PRECIO_APLANAMIENTO'] != "N/A" and empresa['PRECIO_ACTUAL'] is not None:
                    try:
                        precio_compra = float(empresa['PRECIO_APLANAMIENTO'])
                        precio_actual = float(empresa['PRECIO_ACTUAL'])
                        porcentaje = ((precio_compra - precio_actual) / precio_actual) * 100
                        orden_interna = porcentaje
                    except (ValueError, TypeError):
                        pass

            return (orden_grupo, orden_interna)

        datos_ordenados = sorted(datos_completos, key=obtener_clave_ordenacion)
        
        datos_ordenados = [d for d in datos_ordenados if obtener_clave_ordenacion(d)[0] != 99]
        
        # --- Fin de la lógica de ordenación integrada ---

        html_body = f"""
        <html>
        <head>
            <title>Resumen Diario de Oportunidades - {datetime.today().strftime('%d/%m/%Y')}</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 20px;
                }}
                .main-container {{
                    max-width: 1300px;
                    margin: 0 auto;
                    background-color: #fff;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                }}
                h2 {{ color: #2c3e50; text-align: center; }}
                p {{ color: #7f8c8d; text-align: center; }}
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
                    height: 80vh;
                    position: relative;
                }}
                table {{ 
                    width: 90%;
                    table-layout: fixed;
                    margin: 20px auto 0 auto;
                    border-collapse: collapse;
                }}
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 8px; 
                    text-align: center;
                    vertical-align: top;
                    white-space: normal;
                    width: 140px;
                    line-height: 1.2;
                }}
                th {{ 
                    background-color: #f2f2f2;
                    position: sticky;
                    top: 0;
                    z-index: 10;
                }}
                .compra {{ color: #1abc9c; font-weight: bold; }}
                .venta {{ color: #e74c3c; font-weight: bold; }}
                .comprado-si {{ background-color: #2ecc71; color: white; font-weight: bold; }}
                .bg-green {{ background-color: #d4edda; color: #155724; }}
                .bg-red {{ background-color: #f8d7da; color: #721c24; }}
                .bg-highlight {{ background-color: #2ecc71; color: white; font-weight: bold; }}
                .text-center {{ text-align: center; }}
                .disclaimer {{ font-size: 12px; text-align: center; color: #95a5a6; }}
                .small-text {{ font-size: 10px; color: #555; }}
                .green-cell {{ background-color: #d4edda; }}
                .red-cell {{ background-color: #f8d7da; }}
                .separator-row td {{ background-color: black; height: 5px; padding: 0; border: none; }}
                .category-header td {{
                    background-color: #34495e;
                    color: white;
                    font-size: 1.5em;
                    font-weight: bold;
                    text-align: center;
                    padding: 15px;
                    border: none;
                }}
                .stacked-text {{ 
                    line-height: 1.2;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="main-container">
                <h2 class="text-center">Resumen Diario de Oportunidades - {datetime.today().strftime('%d/%m/%Y')}</h2>
                
                <p class="text-center">Se ha generado un resumen de las empresas según su estado y tendencia del Algoritmo IBEXIA. La tabla está ordenada por oportunidad. Usa el buscador para encontrar una empresa rápidamente.</p>
                
                <div id="search-container">
                    <input type="text" id="searchInput" onkeyup="filterTable()" placeholder="Buscar por nombre de empresa...">
                </div>
                
                <div id="scroll-top" style="overflow-x: auto;">
                    <div style="min-width: 1400px;">&nbsp;</div>
                </div>
                
                <div class="table-container">
                    <table id="myTable">
                        <thead>
                            <tr>
                                <th>Empresa (Precio)</th>
                                <th>¿Estamos comprados?</th>
                                <th>Tendencia Actual</th>
                                <th>Oportunidad</th>
                                <th>Compra si...</th>
                                <th>Vende si...</th>
                                <th>Ganancia/Pérdida</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        if not datos_ordenados:
            html_body += """
                            <tr><td colspan="7">No se encontraron empresas con oportunidades claras hoy.</td></tr>
            """
        else:
            previous_oportunidad = None
            for i, data in enumerate(datos_ordenados):
                
                if i == 0:
                    html_body += """
                        <tr class="category-header"><td colspan="7">OPORTUNIDADES DE COMPRA</td></tr>
                    """
                
                if previous_oportunidad is not None and data['OPORTUNIDAD'] != previous_oportunidad:
                    if data['ORDEN_PRIORIDAD'] >= 3 and previous_oportunidad in ["Posibilidad de Compra", "Posibilidad de Compra Activada"]:
                         html_body += """
                            <tr class="category-header"><td colspan="7">ATENTOS A VENDER</td></tr>
                        """
                    html_body += """
                        <tr class="separator-row"><td colspan="7"></td></tr>
                    """

                nombre_con_precio = f"<div class='stacked-text'><b>{data['NOMBRE_EMPRESA']}</b><br>({formatear_numero(data['PRECIO_ACTUAL'])}€)</div>"
                
                clase_oportunidad = "compra" if "compra" in data['OPORTUNIDAD'].lower() else ("venta" if "venta" in data['OPORTUNIDAD'].lower() else "")
                
                celda_empresa_class = ""
                if "compra" in data['OPORTUNIDAD'].lower():
                    celda_empresa_class = "green-cell"
                elif "venta" in data['OPORTUNIDAD'].lower():
                    celda_empresa_class = "red-cell"
                
                if data['COMPRADO'] == 'SI':
                    precio_compra_formateado = formatear_numero(data['PRECIO_COMPRA'])
                    comprado_display = f"SI<br><span class='small-text'>({precio_compra_formateado}€ el {data['FECHA_COMPRA']})</span>"
                    comprado_class = "comprado-si"
                    beneficio_perdida = calcular_beneficio_perdida(data['PRECIO_COMPRA'], data['PRECIO_ACTUAL'])
                    beneficio_clase = "compra" if beneficio_perdida != "N/A" and float(beneficio_perdida.replace(',', '')) >= 0 else "venta"
                    beneficio_display = f"<span class='{beneficio_clase}'>{beneficio_perdida}€</span>"
                else:
                    comprado_display = "NO"
                    comprado_class = ""
                    beneficio_display = "N/A"

                html_body += f"""
                            <tr>
                                <td class="{celda_empresa_class}">{nombre_con_precio}</td>
                                <td class="{comprado_class}">{comprado_display}</td>
                                <td>{data['TENDENCIA_ACTUAL']}</td>
                                <td class="{clase_oportunidad}">{data['OPORTUNIDAD']}</td>
                                <td>{data['COMPRA_SI']}</td>
                                <td>{data['VENDE_SI']}</td>
                                <td>{beneficio_display}</td>
                            </tr>
                """
                previous_oportunidad = data['OPORTUNIDAD']
        
        html_body += """
                        </tbody>
                    </table>
                </div>
                
                <br>
                <p class="disclaimer"><strong>Aviso:</strong> El algoritmo de trading se basa en indicadores técnicos y no garantiza la rentabilidad. Utiliza esta información con tu propio análisis y criterio. ¡Feliz trading!</p>
            </div>

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
        
        asunto = f"🔔 Alertas y Oportunidades IBEXIA: {len(datos_ordenados)} oportunidades detectadas hoy {datetime.today().strftime('%d/%m/%Y')}"
        enviar_email_con_adjunto(html_body, asunto)

    except Exception as e:
        print(f"❌ Error al ejecutar el script principal: {e}")

if __name__ == '__main__':
    generar_reporte()
