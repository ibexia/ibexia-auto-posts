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
    smi_raw[avgdiff == 0] = 0.0 # Asegura que no haya NaNs por división por cero

    smi_smoothed = pd.Series(smi_raw).rolling(window=smooth_period).mean()
    smi_signal = smi_smoothed.ewm(span=ema_signal_len, adjust=False).mean()

    df = df.copy()
    df['SMI'] = smi_smoothed
    df['SMI_signal'] = smi_signal
    
    return df

def find_significant_supports(df, current_price, window=40, tolerance_percent=0.01, max_deviation_percent=0.15):
    """
    Identifica los 3 soportes más significativos y cercanos al precio actual
    basándose en mínimos locales y agrupaciones de precios.
    """
    if df.empty:
        return [0, 0, 0]

    recent_data = df.tail(window)
    lows = recent_data['Low']
    
    potential_supports = []
    
    for i in range(1, len(lows) - 1):
        if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
            potential_supports.append(lows.iloc[i])

    if not potential_supports:
        potential_supports = lows.tolist()
        
    support_zones = {}
    for support in potential_supports:
        found_zone = False
        for zone_level in support_zones.keys():
            if abs(support - zone_level) / support <= tolerance_percent:
                support_zones[zone_level].append(support)
                found_zone = True
                break
        if not found_zone:
            support_zones[support] = [support]
            
    final_supports = []
    for zone_level, values in support_zones.items():
        avg_support = np.mean(values)
        if avg_support < current_price:
            if abs(current_price - avg_support) / current_price <= max_deviation_percent:
                final_supports.append({'level': avg_support, 'frequency': len(values)})

    final_supports.sort(key=lambda x: (abs(x['level'] - current_price), -x['frequency']))
    
    top_3_supports = [round(s['level'], 2) for s in final_supports if s['level'] < current_price][:3]
    
    # Asegurarse de que haya 3 soportes, rellenando con valores calculados si es necesario
    while len(top_3_supports) < 3:
        if len(top_3_supports) > 0:
            top_3_supports.append(round(top_3_supports[-1] * 0.95, 2))
        else:
            top_3_supports.append(round(current_price * 0.90, 2))

    return top_3_supports




def obtener_datos_yfinance(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="60d", interval="1d")

        hist = calculate_smi_tv(hist)

        smi_actual = round(hist['SMI_signal'].iloc[-1], 2)
        current_price = round(info["currentPrice"], 2)
        current_volume = info.get("volume", 0)

        soportes = find_significant_supports(hist, current_price)
        soporte_1, soporte_2, soporte_3 = soportes

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

        if nota_empresa >= 7:
            precio_objetivo_compra = soporte_1
        else:
            drop_percentage_from_base = (7 - nota_empresa) / 7 * 0.15
            precio_objetivo_compra = soporte_1 * (1 - drop_percentage_from_base)

        precio_objetivo_compra = round(precio_objetivo_compra, 2)

        smi_history_last_30 = hist['SMI_signal'].dropna().tail(30).tolist()
        notas_historicas_ultimos_30_dias = [round((-(max(min(smi, 60), -60)) + 60) * 10 / 120, 1) for smi in smi_history_last_30]

        tendencia_smi = "No disponible"
        dias_estimados_accion = "No disponible"

        if len(notas_historicas_ultimos_30_dias) >= 2:
            x = np.arange(len(notas_historicas_ultimos_30_dias))
            y = np.array(notas_historicas_ultimos_30_dias)
            if np.std(y) > 0.01:
                slope, intercept = np.polyfit(x, y, 1)
            else:
                slope = 0.0

            if slope > 0.1:
                tendencia_smi = "mejorando"
            elif slope < -0.1:
                tendencia_smi = "empeorando"
            else:
                tendencia_smi = "estable"

            if nota_empresa <= 2:
                dias_estimados_accion = "Ya en zona de posible venta"
            elif nota_empresa >= 8:
                dias_estimados_accion = "Ya en zona de posible compra"
            elif abs(slope) < 0.01:
                dias_estimados_accion = "Tendencia estable, sin acción inmediata clara"
            elif slope < 0:
                dias = (nota_empresa - 2.0) / abs(slope)
                dias_estimados_accion = f"aprox. {int(max(1, dias))} días para alcanzar zona de venta"
            elif slope > 0:
                dias = (8.0 - nota_empresa) / abs(slope)
                dias_estimados_accion = f"aprox. {int(max(1, dias))} días para alcanzar zona de compra"

        datos = {
            "TICKER": ticker,
            "NOMBRE_EMPRESA": info.get("longName", ticker),
            "PRECIO_ACTUAL": current_price,
            "VOLUMEN": current_volume,
            "SOPORTE_1": soporte_1,
            "SOPORTE_2": soporte_2,
            "SOPORTE_3": soporte_3,
            "RESISTENCIA": round(hist["High"].max(), 2),
            "CONDICION_RSI": condicion_rsi,
            "RECOMENDACION": recomendacion,
            "SMI": smi_actual,
            "NOTA_EMPRESA": nota_empresa,
            "PRECIO_OBJETIVO_COMPRA": precio_objetivo_compra,
            "INGRESOS": info.get("totalRevenue", "N/A"),
            "EBITDA": info.get("ebitda", "N/A"),
            "BENEFICIOS": info.get("grossProfits", "N/A"),
            "DEUDA": info.get("totalDebt", "N/A"),
            "FLUJO_CAJA": info.get("freeCashflow", "N/A"),
            "EXPANSION_PLANES": "Información de planes de expansión no disponible o no traducible en este momento.",
            "ACUERDOS": "Información sobre acuerdos no disponible o no traducible en este momento.",
            "SENTIMIENTO_ANALISTAS": "Sentimiento de analistas no disponible o no traducible.",
            "TENDENCIA_SOCIAL": "No disponible",
            "EMPRESAS_SIMILARES": ", ".join(info.get("category", "").split(",")) if info.get("category") else "No disponibles",
            "RIESGOS_OPORTUNIDADES": "No disponibles",
            "TENDENCIA_NOTA": tendencia_smi,
            "DIAS_ESTIMADOS_ACCION": dias_estimados_accion,
            "NOTAS_HISTORICAS_30_DIAS": notas_historicas_ultimos_30_dias
        }

        # 🔴 Esta parte debe ir aquí, después de crear 'datos'
        cierres_ultimos_30_dias = hist['Close'].dropna().tail(30).tolist()
        datos["CIERRES_30_DIAS"] = [round(float(c), 2) for c in cierres_ultimos_30_dias]

        return datos

    except Exception as e:
        print(f"❌ Error al obtener datos de {ticker}: {e}. Saltando a la siguiente empresa...")
        return None


def formatear_numero(valor):
    try:
        numero = int(valor)
        return f"{numero:,} €".replace(",", ".") # Formato español para miles
    except (ValueError, TypeError):
        return "No disponible"
        
def construir_prompt_formateado(data):
    titulo_post = f"{data['RECOMENDACION']} {data['NOMBRE_EMPRESA']} ({data['PRECIO_ACTUAL']:,}€) {data['TICKER']}"
    
       # NUEVO: Obtener las notas históricas para el gráfico
    notas_historicas = data.get('NOTAS_HISTORICAS_30_DIAS', [])
    cierres_historicos = data.get('CIERRES_30_DIAS', [])
    if len(cierres_historicos) < 30 and cierres_historicos:
        cierres_historicos = [cierres_historicos[0]] * (30 - len(cierres_historicos)) + cierres_historicos
    elif not cierres_historicos:
        cierres_historicos = [0.0] * 30

    # Ajustar para asegurar que siempre haya 7 elementos, rellenando con el último valor si hay menos
    if len(notas_historicas) < 30 and notas_historicas:
        notas_historicas = [notas_historicas[0]] * (30 - len(notas_historicas)) + notas_historicas
    elif not notas_historicas:
        notas_historicas = [0.0] * 30 # Si no hay datos, rellenar con ceros
    
    
    # ... (el resto de tu código para soportes_unicos y tabla_resumen) ...

    # COPIA Y PEGA ESTE BLOQUE EXACTAMENTE AQUÍ (esta variable sí usa """ porque es un HTML largo)
    chart_html = ""
    if notas_historicas:
        labels = [(datetime.today() - timedelta(days=29 - i)).strftime("%d/%m") for i in range(30)]
        
        # Invertir las notas para que el gráfico muestre "Hoy" a la derecha
        notas_historicas_display = notas_historicas

        chart_html = f"""
<h2>Evolución de la Nota Técnica</h2>
<p>Para ofrecer una perspectiva visual clara de la evolución de la nota técnica de <strong>{data['NOMBRE_EMPRESA']}</strong>, he preparado un gráfico que muestra los valores de los últimos treinta días. Esta calificación es una herramienta exclusiva de <strong>ibexia.es</strong> y representa nuestra valoración técnica sobre el momento actual de una acción. La escala va de 0 (venta o cautela) a 10 (oportunidad de compra).</p>
<p>Este gráfico no solo nos permite visualizar la evolución técnica de la acción a lo largo de los últimos 30 días, sino que también incluye la línea del precio de cierre diario. Esta doble representación cumple un doble objetivo: por un lado, proyectar posibles zonas de oportunidad futura; y por otro, dejar constancia clara de nuestras valoraciones y su grado de acierto con el paso del tiempo. Así, no solo anticipamos movimientos, sino que también construimos una trazabilidad transparente de nuestras decisiones técnicas.</p>
<div style="width: 80%; margin: auto; height: 400px;">
    <canvas id="notasChart"></canvas>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@1.1.0"></script>
<script>
    document.addEventListener('DOMContentLoaded', function() {{
        var ctx = document.getElementById('notasChart').getContext('2d');
        var notasChart = new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [
                    {{
                        label: 'Nota Técnica',
                        data: {json.dumps(notas_historicas_display)},
                        backgroundColor: 'rgba(0, 128, 255, 0.4)',
                        borderColor: 'rgba(0, 128, 255, 1)',
                        borderWidth: 1,
                        type: 'bar',
                        yAxisID: 'y'
                    }},
                    {{
                        label: 'Precio de Cierre',
                        data: {json.dumps(cierres_historicos)},
                        type: 'line',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 2,
                        fill: false,
                        yAxisID: 'y1'
                    }}
                ]
            }},
            options: {{
                plugins: {{
                    annotation: {{
                        annotations: {{
                            zonaVerde: {{
                                type: 'box',
                                yMin: 8,
                                yMax: 10,
                                backgroundColor: 'rgba(0, 255, 0, 0.1)',
                                borderWidth: 0
                            }},
                            zonaRoja: {{
                                type: 'box',
                                yMin: 0,
                                yMax: 2,
                                backgroundColor: 'rgba(255, 0, 0, 0.1)',
                                borderWidth: 0
                            }},
                            lineaCompra: {{
                                type: 'line',
                                yMin: 8,
                                yMax: 8,
                                borderColor: 'rgba(0, 200, 0, 1)',
                                borderWidth: 2,
                                label: {{
                                    enabled: true,
                                    content: 'Zona de Compra (8)',
                                    position: 'end',
                                    backgroundColor: 'rgba(0, 200, 0, 0.8)'
                                }}
                            }},
                            lineaVenta: {{
                                type: 'line',
                                yMin: 2,
                                yMax: 2,
                                borderColor: 'rgba(200, 0, 0, 1)',
                                borderWidth: 2,
                                label: {{
                                    enabled: true,
                                    content: 'Zona de Venta (2)',
                                    position: 'end',
                                    backgroundColor: 'rgba(200, 0, 0, 0.8)'
                                }}
                            }}
                        }}
                    }},
                    legend: {{
                        display: true
                    }},
                    tooltip: {{
                        mode: 'index',
                        intersect: false,
                        callbacks: {{
                            label: function(context) {{
                                let label = context.dataset.label || '';
                                if (label) {{
                                    label += ': ';
                                }}
                                label += context.parsed.y.toFixed(2);
                                return label;
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 10,
                        title: {{
                            display: true,
                            text: 'Nota Técnica (0-10)'
                        }}
                    }},
                    y1: {{
                        position: 'right',
                        beginAtZero: false,
                        title: {{
                            display: true,
                            text: 'Precio de Cierre (€)'
                        }},
                        grid: {{
                            drawOnChartArea: false
                        }},
                        ticks: {{
                            padding: 5
                        }},
                        suggestedMin: Math.min(...{json.dumps(cierres_historicos)}) * 0.98,
                        suggestedMax: Math.max(...{json.dumps(cierres_historicos)}) * 1.02
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Últimos 30 Días (ibexia.es)'
                        }}
                    }}
                }},
                responsive: true,
                maintainAspectRatio: false
            }}
        }});
    }});
</script>
<br/>
"""
        # Cálculo dinámico de la descripción del gráfico
                  
        descripcion_grafico = ""
        cierres = data.get("CIERRES_30_DIAS", [])
        notas = data.get("NOTAS_HISTORICAS_30_DIAS", [])

        mejor_compra = None  # (nota_idx, cierre_inicial, cierre_maximo, porcentaje)
        mejor_venta = None   # (nota_idx, cierre_inicial, cierre_minimo, porcentaje)

        if cierres and notas and len(cierres) == len(notas):
            for i in range(len(notas)):
                nota = notas[i]
                cierre = cierres[i]
        
                if nota >= 8:
                    # Buscar el máximo después de la recomendación de compra
                    max_post = max(cierres[i:], default=cierre)
                    pct = ((max_post - cierre) / cierre) * 100
                    if (not mejor_compra) or (pct > mejor_compra[3]):
                        mejor_compra = (i, cierre, max_post, pct)
        
                elif nota <= 2:
                    # Buscar el mínimo después de la recomendación de venta
                    min_post = min(cierres[i:], default=cierre)
                    pct = ((min_post - cierre) / cierre) * 100
                    if (not mejor_venta) or (pct < mejor_venta[3]):
                        mejor_venta = (i, cierre, min_post, pct)

        # Generar el párrafo explicativo
        if mejor_compra:
            idx, inicio, maximo, pct = mejor_compra
            fecha = (datetime.today() - timedelta(days=29 - idx)).strftime("%d/%m")
            descripcion_grafico += f"<p>Destacamos especialmente nuestra recomendación de <strong>compra</strong> el día {fecha}, cuando el precio era de <strong>{inicio:.2f}€</strong>. A partir de ese momento, el valor alcanzó un máximo de <strong>{maximo:.2f}€</strong>, lo que representó una revalorización del <strong>{pct:.2f}%</strong>. Este acierto muestra cómo nuestras señales pueden anticipar movimientos significativos en el mercado.</p>"

        if mejor_venta:
            idx, inicio, minimo, pct = mejor_venta
            fecha = (datetime.today() - timedelta(days=29 - idx)).strftime("%d/%m")
            descripcion_grafico += f"<p>En el lado de las <strong>ventas</strong>, subrayamos nuestra señal del {fecha}, con un precio inicial de <strong>{inicio:.2f}€</strong>. Posteriormente, la acción cayó hasta un mínimo de <strong>{minimo:.2f}€</strong>, registrando un descenso del <strong>{-pct:.2f}%</strong>. Esto refuerza la efectividad de nuestras alertas para proteger el capital en momentos de debilidad del mercado.</p>"


        chart_html += f"""
        <div style="margin-top:20px;">
            <h3>Resumen de nuestro mejor acierto</h3>
            {descripcion_grafico}
        </div>
        """
    
    # Pre-procesamiento de soportes para agruparlos si son muy cercanos
    soportes_unicos = []
    temp_soportes = sorted([data['SOPORTE_1'], data['SOPORTE_2'], data['SOPORTE_3']], reverse=True)
    
    if len(temp_soportes) > 0:
        soportes_unicos.append(temp_soportes[0])
        for i in range(1, len(temp_soportes)):
            if abs(temp_soportes[i] - soportes_unicos[-1]) / soportes_unicos[-1] > 0.005: # Tolerancia del 0.5%
                soportes_unicos.append(temp_soportes[i])
    
    # Asegurarse de que soportes_unicos tenga al menos un elemento para la tabla
    if not soportes_unicos:
        soportes_unicos.append(0.0) # Valor por defecto si no se encontraron soportes

    # Construcción del texto de soportes
    soportes_texto = ""
    if len(soportes_unicos) == 1:
        soportes_texto = f"un soporte clave en <strong>{soportes_unicos[0]:,.2f}€</strong>."
    elif len(soportes_unicos) == 2:
        soportes_texto = f"dos soportes importantes en <strong>{soportes_unicos[0]:,.2f}€</strong> y <strong>{soportes_unicos[1]:,.2f}€</strong>."
    elif len(soportes_unicos) >= 3:
        soportes_texto = (f"tres soportes relevantes: el primero en <strong>{soportes_unicos[0]:,.2f}€</strong>, "
                          f"el segundo en <strong>{soportes_unicos[1]:,.2f}€</strong>, y el tercero en <strong>{soportes_unicos[2]:,.2f}€</strong>.")
    else:
        soportes_texto = "no presenta soportes claros en el análisis reciente, requiriendo un seguimiento cauteloso."

    # Construcción de la tabla de resumen de puntos clave
    tabla_resumen = f"""
<h2>Resumen de Puntos Clave</h2>
<table border="1" style="width:100%; border-collapse: collapse;">
    <tr>
        <th style="padding: 8px; text-align: left; background-color: #f2f2f2;">Métrica</th>
        <th style="padding: 8px; text-align: left; background-color: #f2f2f2;">Valor</th>
    </tr>
    <tr>
        <td style="padding: 8px;">Precio Actual</td>
        <td style="padding: 8px;"><strong>{data['PRECIO_ACTUAL']:,}€</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Volumen</td>
        <td style="padding: 8px;"><strong>{data['VOLUMEN']:,} acciones</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Soporte Clave</td>
        <td style="padding: 8px;"><strong>{soportes_unicos[0]:,.2f}€</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Resistencia Clave</td>
        <td style="padding: 8px;"><strong>{data['RESISTENCIA']:,}€</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Recomendación</td>
        <td style="padding: 8px;"><strong>{data['RECOMENDACION']}</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Nota Técnica (0-10)</td>
        <td style="padding: 8px;"><strong>{data['NOTA_EMPRESA']:,}</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Precio Objetivo de Compra</td>
        <td style="padding: 8px;"><strong>{data['PRECIO_OBJETIVO_COMPRA']:,}€</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Tendencia de la Nota</td>
        <td style="padding: 8px;"><strong>{data['TENDENCIA_NOTA']}</strong></td>
    </tr>
    <tr>
        <td style="padding: 8px;">Días Estimados para Acción</td>
        <td style="padding: 8px;"><strong>{data['DIAS_ESTIMADOS_ACCION']}</strong></td>
    </tr>
</table>
<br/>
"""

    # Dinámica del Impulso - Contenido generado dinámicamente
    dinamica_impulso_text = ""
    if data['TENDENCIA_NOTA'] == "mejorando":
        # Corrección de sintaxis de f-string anidada y eliminación de '***'
        dias_info = f"Según esta dinámica, estimo que podríamos estar a {data['DIAS_ESTIMADOS_ACCION']} para una posible acción de compra." if 'compra' in data['DIAS_ESTIMADOS_ACCION'] else ''
        dinamica_impulso_text = f"La tendencia de nuestra nota técnica es actualmente **mejorando**, lo que sugiere un **impulso alcista** en el comportamiento técnico de la acción. Esto indica que los indicadores del gráfico están mostrando una fortaleza creciente. {dias_info}"
    elif data['TENDENCIA_NOTA'] == "empeorando":
        # Corrección de sintaxis de f-string anidada y eliminación de '***'
        dias_info = f"Según esta dinámica, estimo que podríamos estar a {data['DIAS_ESTIMADOS_ACCION']} para una posible acción de venta." if 'venta' in data['DIAS_ESTIMADOS_ACCION'] else ''
        dinamica_impulso_text = f"La tendencia de nuestra nota técnica es actualmente **empeorando**, lo que sugiere un **impulso bajista** en el comportamiento técnico de la acción. Esto indica que los indicadores del gráfico están mostrando una debilidad creciente. {dias_info}"
    else: # Estable o "Ya en zona de posible venta/compra"
        if "Ya en zona" in data['DIAS_ESTIMADOS_ACCION']:
            # Corrección de sintaxis de f-string anidada y eliminación de '***'
            accion_type = 'posible compra' if data['NOTA_EMPRESA'] >= 8 else 'posible venta'
            entrada_salida = 'de entrada inmediata para compra' if data['NOTA_EMPRESA'] >= 8 else 'de salida inmediata para venta'
            dinamica_impulso_text = f"La nota técnica de la empresa ya se encuentra en una **zona de {accion_type}**, lo que indica que el mercado ya ha descontado gran parte del movimiento en esa dirección. Esto podría ofrecer una oportunidad {entrada_salida} para el inversor que busque una acción rápida. Si bien la nota es **{data['NOTA_EMPRESA']}**, es crucial vigilar la volatilidad y los eventos externos que puedan alterar el impulso actual."
        else:
            dinamica_impulso_text = f"La tendencia de nuestra nota técnica es actualmente **estable**, lo que sugiere que el comportamiento técnico de la acción se mantiene sin cambios significativos. Esto implica que no se proyecta una acción inminente basada únicamente en este indicador, aunque siempre es importante estar atento a cualquier cambio en el volumen o los niveles de soporte y resistencia."


    # Volumen - Contenido generado dinámicamente
    volumen_analisis_text = ""
    # Corrección: Eliminar los asteriscos de formato de Markdown dentro de la f-string
    if data['VOLUMEN'] is not None:
        volumen_analisis_text = f"Analizando el volumen de {data['VOLUMEN']:,} acciones, este volumen [El modelo debe decidir si es alto/bajo/normal en relación al historial y la tendencia. Por ejemplo: 'es consistente con la fase de acumulación que observo en el gráfico, y refuerza la validez de los niveles de soporte detectados.' o 'es ligeramente inferior al promedio reciente, lo que podría indicar una falta de convicción en el movimiento actual.']. Un incremento del volumen en la ruptura de la resistencia, por ejemplo, sería una señal inequívoca de fuerza para la tendencia alcista que preveo. La consolidación actual en torno a los soportes identificados, combinada con el volumen, sugiere [interpreta la combinación de volumen y soportes, como acumulación de posiciones, debilidad de la venta, etc.]. El hecho de que no haya un volumen explosivo en este momento refuerza la idea de un movimiento gradual y menos arriesgado, en contraste con una rápida subida impulsada por especulación."
    else:
        volumen_analisis_text = "Actualmente, no dispongo de datos de volumen reciente para realizar un análisis en profundidad. Sin embargo, en cualquier estrategia de inversión, el volumen es un indicador crucial que valida los movimientos de precio y la fuerza de las tendencias. Un volumen significativo en rupturas de niveles clave o en cambios de tendencia es una señal potente a tener en cuenta."


    prompt = f"""
Actúa como un trader profesional con amplia experiencia en análisis técnico y mercados financieros. Genera el análisis completo en **formato HTML**, ideal para publicaciones web. Utiliza etiquetas `<h2>` para los títulos de sección y `<p>` para cada párrafo de texto. Redacta en primera persona, con total confianza en tu criterio. 

Destaca los datos importantes como precios, notas de la empresa, cifras financieras y el nombre de la empresa utilizando la etiqueta `<strong>`. Asegúrate de que no haya asteriscos u otros símbolos de marcado en el texto final, solo HTML válido. Asegurate que todo este escrito en español independientemente del idioma de donde saques los datos.

Genera un análisis técnico completo de aproximadamente 1200 palabras sobre la empresa {data['NOMBRE_EMPRESA']}, utilizando los siguientes datos reales extraídos de Yahoo Finance. Presta especial atención a la **nota obtenida por la empresa**: {data['NOTA_EMPRESA']}.

**Datos clave:**
- Precio actual: {data['PRECIO_ACTUAL']}
- Volumen del último día completo: {data['VOLUMEN']}
- Soporte 1: {data['SOPORTE_1']}
- Soporte 2: {data['SOPORTE_2']}
- Soporte 3: {data['SOPORTE_3']}
- Resistencia clave: {data['RESISTENCIA']}
- Recomendación general: {data['RECOMENDACION']}
- Nota de la empresa (0-10): {data['NOTA_EMPRESA']}
- Precio objetivo de compra: {data['PRECIO_OBJETIVO_COMPRA']}€
- Resultados financieros recientes: {data['INGRESOS']}, {data['EBITDA']}, {data['BENEFICIOS']}
- Nivel de deuda y flujo de caja: {data['DEUDA']}, {data['FLUJO_CAJA']}
- Información estratégica: {data['EXPANSION_PLANES']}, {data['ACUERDOS']}
- Sentimiento del mercado: {data['SENTIMIENTO_ANALISTAS']}, {data['TENDENCIA_SOCIAL']}
- Comparativa sectorial: {data['EMPRESAS_SIMILARES']}
- Riesgos y oportunidades: {data['RIESGOS_OPORTUNIDADES']}
- Tendencia de la nota: {data['TENDENCIA_NOTA']}
- Días estimados para acción: {data['DIAS_ESTIMADOS_ACCION']}

Importante: si algún dato no está disponible ("N/A", "No disponibles", "No disponible"), no lo menciones ni digas que falta. No expliques que la recomendación proviene de un indicador o dato específico. La recomendación debe presentarse como una conclusión personal basada en tu experiencia y criterio profesional como analista. Al redactar el análisis, haz referencia a la **nota obtenida por la empresa ({data['NOTA_EMPRESA']})** en al menos dos de los párrafos principales (Recomendación General, Análisis a Corto Plazo o Predicción a Largo Plazo) como un factor clave para tu valoración.

---
<h1>{titulo_post}</h1>


<h2>Análisis Inicial y Recomendación</h2>
<p>En el dinámico mercado actual, <strong>{data['NOMBRE_EMPRESA']} ({data['TICKER']})</strong> está enviando señales claras de un potencial giro. ¿Es este el momento ideal para considerar una entrada o salida? Mi análisis técnico apunta a que sí, con una oportunidad {('de compra inminente y un rebote en el horizonte' if data['NOTA_EMPRESA'] >= 7 else 'de venta potencial o de esperar una corrección')}.</p>

<p>La empresa cotiza actualmente a <strong>{data['PRECIO_ACTUAL']:,}€</strong>, un nivel que considero estratégico. Mi precio objetivo de compra se sitúa en <strong>{data['PRECIO_OBJETIVO_COMPRA']:,}€</strong>. Este último representa el nivel más atractivo para una entrada conservadora, y aunque el precio actual está {('por encima' if float(data['PRECIO_ACTUAL']) > float(data['PRECIO_OBJETIVO_COMPRA']) else 'por debajo')}, aún puede presentar una oportunidad si se evalúa cuidadosamente la relación riesgo/recompensa. Como analista, mi visión es que la convergencia hacia este objetivo podría ser el punto de partida para un movimiento significativo. El volumen negociado recientemente, que alcanzó las <strong>{data['VOLUMEN']:,} acciones</strong>, es un factor clave que valida estos movimientos, y será crucial monitorearlo para confirmar la fuerza de cualquier tendencia emergente.</p>

<p>Asignamos una <strong>nota técnica de {data['NOTA_EMPRESA']} sobre 10</strong>. Esta puntuación refleja [elige una de las siguientes opciones basadas en la nota, manteniendo el foco en el análisis técnico]:
    {"una excelente fortaleza técnica y baja volatilidad esperada a corto plazo. La sólida puntuación se basa en la evaluación de indicadores clave de impulso, soporte y resistencia, lo que indica un bajo riesgo técnico en relación con el potencial de crecimiento a corto plazo." if data['NOTA_EMPRESA'] >= 8 else ""}
    {"una fortaleza técnica moderada, con un equilibrio entre potencial y riesgo. Se basa en el comportamiento del gráfico, soportes, resistencias e impulso, sugiriendo una oportunidad que requiere seguimiento." if 6 <= data['NOTA_EMPRESA'] < 8 else ""}
    {"una situación técnica neutral, donde el gráfico no muestra un patrón direccional claro. La puntuación se deriva del análisis de los movimientos de precio y volumen, indicando que es un momento para la observación y no para la acción inmediata." if 5 <= data['NOTA_EMPRESA'] < 6 else ""}
    {"cierta debilidad técnica, con posibles señales de corrección o continuación bajista. La puntuación se basa en los indicadores del gráfico, que muestran una pérdida de impulso alcista y un aumento de la presión vendedora." if 3 <= data['NOTA_EMPRESA'] < 5 else ""}
    {"una debilidad técnica significativa y una posible sobrecompra en el gráfico, lo que sugiere un alto riesgo de corrección. La puntuación se basa en el análisis de los patrones de precio y volumen, indicando que es un momento para la cautela extrema." if data['NOTA_EMPRESA'] < 3 else ""}
Es importante recordar que esta nota es puramente un reflejo del **análisis del gráfico y sus indicadores técnicos**, y no obedece a la situación financiera o de otro tipo de la empresa. Como profesional, esta nota es mi valoración experta al interpretar el comportamiento del precio y los indicadores.</p>
{chart_html}

<h2>Análisis a Corto Plazo: Soportes, Resistencias y Dinámica del Impulso</h2>
<p>Para entender los posibles movimientos a corto plazo en <strong>{data['NOMBRE_EMPRESA']}</strong>, es fundamental analizar el comportamiento reciente del volumen y las zonas clave de soporte y resistencia. Estos niveles no son meros puntos en un gráfico; son reflejos de la psicología del mercado y de puntos donde la oferta y la demanda han encontrado equilibrio o desequilibrio en el pasado, y pueden volver a hacerlo.</p>

<p>En este momento, observo {soportes_texto} La resistencia clave se encuentra en <strong>{data['RESISTENCIA']:,}€</strong>, situada a una distancia del <strong>{((float(data['RESISTENCIA']) - float(data['PRECIO_ACTUAL'])) / float(data['PRECIO_ACTUAL']) * 100):.2f}%</strong> desde el precio actual. Estas zonas técnicas pueden actuar como puntos de inflexión vitales, y su cercanía o lejanía tiene implicaciones operativas claras. Romper la resistencia implicaría un nuevo camino al alza, mientras que la pérdida de un soporte podría indicar una continuación de la caída. Estoy siguiendo de cerca cómo el precio interactúa con estos niveles.</p>

<h2>Rango de Precio y Recomendaciones Clave</h2>
<p>Esta barra representa el rango técnico actual de <strong>{data['NOMBRE_EMPRESA']}</strong>, desde el soporte hasta la resistencia. Los marcadores indican el <strong>precio actual</strong, el <strong>precio objetivo de compra</strong>, el <strong>soporte</strong> y la <strong>resistencia</strong>. Esta visualización facilita una comprensión inmediata de dónde se encuentra el precio en relación con los niveles clave.</p>

<div style="width: 100%; max-width: 700px; margin: auto; height: 120px;">
    <canvas id="barraRangoChart"></canvas>
</div>

<script>
document.addEventListener('DOMContentLoaded', function () {{
    const soporte = {data['SOPORTE_1']};
    const objetivo = {data['PRECIO_OBJETIVO_COMPRA']};
    const actual = {data['PRECIO_ACTUAL']};
    const resistencia = {data['RESISTENCIA']};

    const valores = [soporte, objetivo, actual, resistencia];
    const min = Math.min(...valores);
    const max = Math.max(...valores);
    const padding = (max - min) * 0.2;

    const ctx = document.getElementById('barraRangoChart').getContext('2d');
    new Chart(ctx, {{
        type: 'bar',
        data: {{
            labels: [''],
            datasets: [{{
                label: 'Rango de Precios',
                data: [max - min + padding * 2],
                backgroundColor: 'rgba(200, 200, 200, 0.4)',
                borderSkipped: false
            }}]
        }},
        options: {{
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {{
                x: {{
                    min: min - padding,
                    max: max + padding,
                    title: {{
                        display: true,
                        text: 'Precio (€)'
                    }},
                    ticks: {{
                        callback: function(value) {{
                            return value.toFixed(2) + ' €';
                        }}
                    }}
                }},
                y: {{
                    display: false
                }}
            }},
            plugins: {{
                legend: {{
                    display: false
                }},
                annotation: {{
                    annotations: {{
                        soporte: {{
                            type: 'line',
                            xMin: soporte,
                            xMax: soporte,
                            borderColor: 'rgba(75, 192, 192, 1)',
                            borderWidth: 2,
                            label: {{
                                content: 'Soporte',
                                enabled: true,
                                position: 'start',
                                backgroundColor: 'rgba(75, 192, 192, 0.8)',
                                font: {{
                                    weight: 'bold'
                                }}
                            }}
                        }},
                        objetivo: {{
                            type: 'line',
                            xMin: objetivo,
                            xMax: objetivo,
                            borderColor: 'rgba(0, 200, 0, 1)',
                            borderWidth: 3,
                            label: {{
                                content: '🎯 Precio Objetivo',
                                enabled: true,
                                position: 'start',
                                backgroundColor: 'rgba(0, 200, 0, 0.9)',
                                color: '#fff',
                                font: {{
                                    weight: 'bold'
                                }}
                            }}
                        }},
                        actual: {{
                            type: 'line',
                            xMin: actual,
                            xMax: actual,
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 2,
                            label: {{
                                content: 'Precio Actual',
                                enabled: true,
                                position: 'start',
                                backgroundColor: 'rgba(54, 162, 235, 0.9)',
                                color: '#fff',
                                font: {{
                                    weight: 'bold'
                                }}
                            }}
                        }},
                        resistencia: {{
                            type: 'line',
                            xMin: resistencia,
                            xMax: resistencia,
                            borderColor: 'rgba(255, 99, 132, 1)',
                            borderWidth: 2,
                            label: {{
                                content: 'Resistencia',
                                enabled: true,
                                position: 'start',
                                backgroundColor: 'rgba(255, 99, 132, 0.9)',
                                color: '#fff',
                                font: {{
                                    weight: 'bold'
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }});
}});
</script>


<h2>Estrategia de Inversión y Gestión de Riesgos</h2>
<p>Un aspecto crucial en el análisis de corto plazo es la dinámica del impulso de la empresa. Mi evaluación profesional indica que la tendencia actual de nuestra nota técnica es **{data['TENDENCIA_NOTA']}**. Esto sugiere {('un rebote inminente, dado que los indicadores muestran una sobreventa extrema, lo que significa que la acción ha sido \'castigada\' en exceso y hay una alta probabilidad de que los compradores tomen el control, impulsando el precio al alza. Esta situación de sobreventa, sumada al impulso alcista subyacente, nos sugiere que estamos ante el inicio de un rebote significativo.' if data['TENDENCIA_NOTA'] == 'mejorando' and data['NOTA_EMPRESA'] < 6 else '')}
{('una potencial continuación bajista, con los indicadores técnicos mostrando una sobrecompra significativa o una pérdida de impulso alcista. Esto sugiere que la acción podría experimentar una corrección. Es un momento para la cautela y la vigilancia de los niveles de soporte.' if data['TENDENCIA_NOTA'] == 'empeorando' and data['NOTA_EMPRESA'] > 4 else '')}
{('una fase de consolidación o lateralidad, donde los indicadores técnicos no muestran una dirección clara. Es un momento para esperar la confirmación de una nueva tendencia antes de tomar decisiones.' if data['TENDENCIA_NOTA'] == 'estable' else '')}
{f" Calculamos que este impulso podría llevarnos a una potencial zona de {('toma de beneficios o venta' if data['NOTA_EMPRESA'] >= 8 else 'entrada o compra')} en aproximadamente **{data['DIAS_ESTIMADOS_ACCION']}**." if "No disponible" not in data['DIAS_ESTIMADOS_ACCION'] and "Ya en zona" not in data['DIAS_ESTIMADOS_ACCION'] else ("La nota ya se encuentra en una zona de acción clara, lo que sugiere una oportunidad {('de compra' if data['NOTA_EMPRESA'] >= 8 else 'de venta')} inmediata, y por tanto, no se estima un plazo de días adicional." if "Ya en zona" in data['DIAS_ESTIMADOS_ACCION'] else "")}</p>

<p>{volumen_analisis_text}</p>

<p>Basado en nuestro análisis, una posible estrategia de entrada sería considerar una compra cerca {f"del soporte de <strong>{soportes_unicos[0]:,.2f}€</strong>" if len(soportes_unicos) > 0 else ""} o, idealmente, en {f"los <strong>{soportes_unicos[1]:,.2f}€</strong>." if len(soportes_unicos) > 1 else "."} Estos niveles ofrecen una relación riesgo/recompensa atractiva, permitiendo una entrada con mayor margen de seguridad. Para gestionar el riesgo de forma efectiva, se recomienda establecer un stop loss ajustado justo por debajo del soporte más bajo que hemos identificado, por ejemplo, en {f"<strong>{soportes_unicos[-1]:,.2f}€</strong>." if len(soportes_unicos) > 0 else "un nivel apropiado de invalidación."} Este punto actuaría como un nivel de invalidez de nuestra tesis de inversión. Nuestro objetivo de beneficio (Take Profit) a corto plazo se sitúa en la resistencia clave de <strong>{data['RESISTENCIA']:,}€</strong>, lo que representa un potencial de revalorización significativo. Esta configuración de entrada, stop loss y objetivo permite una relación riesgo/recompensa favorable para el inversor, buscando maximizar el beneficio mientras se protege el capital.</p>


<h2>Visión a Largo Plazo y Fundamentales</h2>
<p>En un enfoque a largo plazo, el análisis se vuelve más robusto y se apoya en los fundamentos reales del negocio. Aquí, la evolución de <strong>{data['NOMBRE_EMPRESA']}</strong> dependerá en gran parte de sus cifras estructurales y sus perspectivas estratégicas.</p>

<p>En el último ejercicio, los ingresos declarados fueron de <strong>{formatear_numero(data['INGRESOS'])}</strong>, el EBITDA alcanzó <strong>{formatear_numero(data['EBITDA'])}</strong>, y los beneficios netos se situaron en torno a <strong>{formatear_numero(data['BENEFICIOS'])}</strong>. 
En cuanto a su posición financiera, la deuda asciende a <strong>{formatear_numero(data['DEUDA'])}</strong>, y el flujo de caja operativo es de <strong>{formatear_numero(data['FLUJO_CAJA'])}</strong>.</p>

<p>{data['EXPANSION_PLANES'] if data['EXPANSION_PLANES'] != 'N/A' and 'no disponible' not in data['EXPANSION_PLANES'].lower() else f"Aunque la información específica sobre planes de expansión no está detallada en este momento, es crucial para <strong>{data['NOMBRE_EMPRESA']}</strong> delinear una estrategia clara de crecimiento. La capacidad de la empresa para innovar, expandir su cuota de mercado o diversificar sus operaciones será determinante para su valor a largo plazo y para mantener la competitividad en su sector. Estoy monitoreando cualquier anuncio futuro que pueda dar luz sobre estas iniciativas."} {data['ACUERDOS'] if data['ACUERDOS'] != 'No disponibles' and 'no disponible' not in data['ACUERDOS'].lower() else f"Respecto a posibles acuerdos estratégicos o colaboraciones, la información actual es limitada. Sin embargo, en el sector de <strong>{data['NOMBRE_EMPRESA']}</strong>, las alianzas y asociaciones son a menudo catalizadores clave para la expansión y la optimización de recursos, lo que podría impactar positivamente su perfil de crecimiento futuro."}</p>

<p>Considerando la información financiera disponible, <strong>{data['NOMBRE_EMPRESA']}</strong> {f"muestra unos ingresos sólidos de <strong>{formatear_numero(data['INGRESOS'])}</strong>, lo que sugiere una base de operaciones robusta. Su EBITDA de <strong>{formatear_numero(data['EBITDA'])}</strong> indica una buena capacidad para generar ganancias antes de intereses, impuestos, depreciaciones y amortizaciones, lo cual es un indicador de eficiencia operativa. Los beneficios de <strong>{formatear_numero(data['BENEFICIOS'])}</strong>, aunque importantes, deben evaluarse en el contexto de la deuda de <strong>{formatear_numero(data['DEUDA'])}</strong> y el flujo de caja de <strong>{formatear_numero(data['FLUJO_CAJA'])}</strong>. Un flujo de caja positivo es vital para la sostenibilidad y la capacidad de la empresa para invertir en su futuro y afrontar sus obligaciones financieras. Si bien la deuda es una métrica a observar, un flujo de caja saludable puede mitigar los riesgos asociados, sugiriendo que la empresa tiene la capacidad de generar liquidez para soportar sus operaciones y posibles expansiones. Mi interpretación es que la empresa presenta una situación financiera que, si bien tiene aspectos a monitorear como la deuda, se sustenta en una generación de ingresos y un EBITDA consistentes, lo que le confiere una base para un crecimiento potencial sostenido a largo plazo." if data['INGRESOS'] != 'N/A' else "carece de datos financieros recientes para una evaluación fundamental completa. En general, para cualquier empresa, la solidez financiera es la base de un crecimiento sostenible. Ingresos consistentes, un EBITDA saludable y beneficios positivos son señales de un negocio bien gestionado. Un nivel de deuda manejable y un flujo de caja libre positivo son indicadores críticos de la solvencia y capacidad de la empresa para generar valor a largo plazo. La ausencia de estos datos fundamentales impide una predicción sólida a largo plazo, por lo que recomiendo una investigación adicional exhaustiva de sus estados financieros antes de cualquier decisión de inversión a largo plazo."}</p>

<h2>Comparativa Financiera: EBITDA vs Deuda</h2>
<p>Para evaluar de forma visual la salud financiera de <strong>{data['NOMBRE_EMPRESA']}</strong>, a continuación muestro un gráfico de barras horizontales centradas que compara el <strong>EBITDA</strong> (capacidad operativa de generación de beneficios) frente a la <strong>Deuda total</strong>. Un EBITDA superior a la deuda es generalmente una señal positiva de solvencia. En cambio, una deuda que excede al EBITDA requiere análisis adicional sobre su sostenibilidad.</p>

<div style="width: 80%; margin: auto; height: 300px;">
    <canvas id="ebitdaVsDeudaChart"></canvas>
</div>

<script>
    document.addEventListener('DOMContentLoaded', function () {{
        var ctx = document.getElementById('ebitdaVsDeudaChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: ['EBITDA', 'Deuda'],
                datasets: [{{
                    label: 'Millones de Euros',
                    data: [{float(data['EBITDA']) if data['EBITDA'] != 'N/A' else 0}, {-float(data['DEUDA']) if data['DEUDA'] != 'N/A' else 0}],
                    backgroundColor: ['#2e86de', '#c0392b']
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                plugins: {{
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                let val = Math.abs(context.parsed.x);
                                return context.label + ': ' + val.toLocaleString() + ' €';
                            }}
                        }}
                    }},
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    x: {{
                        stacked: false,
                        title: {{
                            display: true,
                            text: 'Millones de Euros'
                        }},
                        ticks: {{
                            callback: function(value) {{
                                return Math.abs(value);
                            }}
                        }},
                        min: -Math.max({float(data['EBITDA']) if data['EBITDA'] != 'N/A' else 0}, {float(data['DEUDA']) if data['DEUDA'] != 'N/A' else 0}) * 1.2,
                        max: Math.max({float(data['EBITDA']) if data['EBITDA'] != 'N/A' else 0}, {float(data['DEUDA']) if data['DEUDA'] != 'N/A' else 0}) * 1.2
                    }},
                    y: {{
                        title: {{
                            display: false
                        }}
                    }}
                }}
            }}
        }});
    }});
</script>


<h2>Conclusión General y Descargo de Responsabilidad</h2>
<p>Para cerrar este análisis de <strong>{data['NOMBRE_EMPRESA']}</strong>, resumo mi visión actual basada en una integración de datos técnicos, financieros y estratégicos. Considero que las claras señales técnicas que apuntan a {('un rebote desde una zona de sobreventa extrema, configurando una oportunidad atractiva' if data['NOTA_EMPRESA'] >= 7 else 'una posible corrección, lo que exige cautela')}, junto con {f"sus sólidos ingresos de <strong>{formatear_numero(data['INGRESOS'])}</strong> y un flujo de caja positivo de <strong>{formatear_numero(data['FLUJO_CAJA'])}</strong>," if data['INGRESOS'] != 'N/A' else "aspectos fundamentales que requieren mayor claridad,"} hacen de esta empresa un activo para mantener bajo estricta vigilancia. La expectativa es que {f"en los próximos {data['DIAS_ESTIMADOS_ACCION']}" if "No disponible" not in data['DIAS_ESTIMADOS_ACCION'] and "Ya en zona" not in data['DIAS_ESTIMADOS_ACCION'] else "en el corto plazo"}, se presente una oportunidad {('de compra con una relación riesgo-recompensa favorable' if data['NOTA_EMPRESA'] >= 7 else 'de observación o de potencial venta, si los indicadores confirman la debilidad')}. Mantendremos una estrecha vigilancia sobre el comportamiento del precio y el volumen para confirmar esta hipótesis.</p>
{tabla_resumen}
<p>Descargo de responsabilidad: Este contenido tiene una finalidad exclusivamente informativa y educativa. No constituye ni debe interpretarse como una recomendación de inversión, asesoramiento financiero o una invitación a comprar o vender ningún activo. La inversión en mercados financieros conlleva riesgos, incluyendo la pérdida total del capital invertido. Se recomienda encarecidamente a cada inversor realizar su propia investigación exhaustiva (due diligence), consultar con un asesor financiero cualificado y analizar cada decisión de forma individual, teniendo en cuenta su perfil de riesgo personal, sus objetivos financieros y su situación económica antes de tomar cualquier decisión de inversión. El rendimiento pasado no es indicativo de resultados futuros.</p>

<h3>¿Qué analizaremos mañana? ¡No te lo pierdas!</h3>
<p>Mañana, pondremos bajo la lupa a otros 10 valores más. ¿Será el próximo candidato para una oportunidad de compra o venta? ¡Vuelve mañana a la misma hora para descubrirlo y seguir ampliando tu conocimiento de mercado!</p>

<h3>Tu Opinión Importa: ¡Participa!</h3>
<p>¿Considerarías comprar acciones de <strong>{data['NOMBRE_EMPRESA']} ({data['TICKER']})</strong> con este análisis?</p>
<ul>
    <li>Sí, la oportunidad es clara.</li>
    <li>No, prefiero esperar más datos.</li>
    <li>Ya las tengo en cartera.</li>
</ul>
<p>¡Déjanos tu voto y tu comentario sobre tu visión de <strong>{data['NOMBRE_EMPRESA']}</strong> en la sección de comentarios! Queremos saber qué piensas y fomentar una comunidad de inversores informada.</p>
"""

    return prompt, titulo_post


def enviar_email(texto_generado, asunto_email, nombre_archivo):
    import os
    from email.mime.base import MIMEBase
    from email import encoders

    remitente = "xumkox@gmail.com"
    destinatario = "xumkox@gmail.com"
    password = "kdgz lvdo wqvt vfkt"  # RECOMENDADO: usar variable de entorno

    # Guardar el HTML en un archivo temporal
    ruta_archivo = f"{nombre_archivo}.html"
    with open(ruta_archivo, "w", encoding="utf-8") as f:
        f.write(texto_generado)

    # Crear el email
    msg = MIMEMultipart()
    msg['From'] = remitente
    msg['To'] = destinatario
    msg['Subject'] = asunto_email
    msg.attach(MIMEText("Adjunto el análisis en formato HTML.", 'plain'))

    # Adjuntar el archivo HTML
    with open(ruta_archivo, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename= {nombre_archivo}.html")
    msg.attach(part)

    # Enviar el correo
    try:
        servidor = smtplib.SMTP('smtp.gmail.com', 587)
        servidor.starttls()
        servidor.login(remitente, password)
        servidor.sendmail(remitente, destinatario, msg.as_string())
        servidor.quit()
        print(f"✅ Correo enviado con el adjunto: {ruta_archivo}")
    except Exception as e:
        print("❌ Error al enviar el correo:", e)



def generar_contenido_con_gemini(tickers):
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise Exception("No se encontró la variable de entorno GEMINI_API_KEY")

    
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")  

    for ticker in tickers:
        print(f"\n📊 Procesando ticker: {ticker}")
        data = obtener_datos_yfinance(ticker)
        if not data:
            print(f"⏩ Saltando {ticker} debido a un error al obtener datos.")
            continue
        
        prompt, titulo_post = construir_prompt_formateado(data)

        max_retries = 3
        initial_delay = 10  
        retries = 0
        delay = initial_delay

        while retries < max_retries:
            try:
                response = model.generate_content(prompt)
                print(f"\n🧠 Contenido generado para {ticker}:\n")
                print(response.text)
                asunto_email = f"Análisis: {data['NOMBRE_EMPRESA']} ({data['TICKER']}) - {data['RECOMENDACION']}"
                nombre_archivo = f"analisis_{ticker}_{datetime.today().strftime('%Y%m%d')}"
                enviar_email(response.text, asunto_email, nombre_archivo)

                break  
            except Exception as e:
                if "429 You exceeded your current quota" in str(e):
                    try:
                        match = re.search(r"retry_delay \{\s*seconds: (\d+)", str(e))
                        if match:
                            server_delay = int(match.group(1))
                            delay = max(delay, server_delay + 1)
                    except:
                        pass
                    
                    print(f"❌ Cuota de Gemini excedida al generar contenido. Reintentando en {delay} segundos... (Intento {retries + 1}/{max_retries})")
                    time.sleep(delay)
                    retries += 1
                    delay *= 2
                else:
                    print(f"❌ Error al generar contenido con Gemini (no de cuota): {e}")
                    break
        else:  
            print(f"❌ Falló la generación de contenido para {ticker} después de {max_retries} reintentos.")
            
        # --- PAUSA DE 1 MINUTO DESPUÉS DE CADA TICKER ---
        print(f"⏳ Esperando 60 segundos antes de procesar el siguiente ticker...")
        time.sleep(60) # Pausa de 60 segundos entre cada ticker

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
