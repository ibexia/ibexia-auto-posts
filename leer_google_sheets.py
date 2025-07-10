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
        # Calcula el nombre de la empresa para el hashtag, eliminando caracteres especiales y pasando a minúsculas.
    company_name_for_hashtag = re.sub(r'[^a-zA-Z0-9]', '', data['NOMBRE_EMPRESA']).lower()
    
    # Construye el título completo, incluyendo los hashtags.
    titulo_post = f"{data['RECOMENDACION']} {data['NOMBRE_EMPRESA']} ({data['PRECIO_ACTUAL']:,}€) {data['TICKER']} #{company_name_for_hashtag} #{data['TICKER'].replace('.MC', '').lower()}"
    
    inversion_base = 10000.0
    inversion_base = 10000.0
    comision_por_operacion_porcentual = 0.001
    
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
<p>Para ofrecer una perspectiva visual clara de la evolución de la nota técnica de <strong>{data['NOMBRE_EMPRESA']}</strong>, mostramos un gráfico que muestra los valores de los últimos treinta días. Esta calificación es una herramienta exclusiva de <strong>ibexia.es</strong> y representa el histórico entre nuestra valoración técnica (barras azules) sobre el precio de cotización (linea roja). La escala va de 0 (venta o cautela) a 10 (oportunidad de compra).</p>
<p>Esta deja constancia clara de nuestras valoraciones y su grado de acierto con el paso del tiempo. Así, no solo anticipamos movimientos, sino que también construimos una trazabilidad transparente de nuestras decisiones técnicas.</p>
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
        
        chart_html += f"""
<h2>Evolución del Precio con Soportes y Resistencias</h2>
<p>A continuación, muestro un gráfico de precios de cierre de los últimos 30 días para <strong>{data['NOMBRE_EMPRESA']}</strong>, con las zonas clave de soporte, resistencia y el precio objetivo de compra claramente marcadas. Estas líneas permiten identificar visualmente los puntos más relevantes para la toma de decisiones estratégicas.</p>

<div style="width: 80%; margin: auto; height: 400px;">
    <canvas id="preciosChart"></canvas>
</div>

<script>
document.addEventListener('DOMContentLoaded', function () {{
    var ctx = document.getElementById('preciosChart').getContext('2d');
    var preciosChart = new Chart(ctx, {{
        type: 'line',
        data: {{
            labels: {json.dumps([(datetime.today() - timedelta(days=29 - i)).strftime("%d/%m") for i in range(30)])},
            datasets: [{{
                label: 'Precio de Cierre',
                data: {json.dumps(data['CIERRES_30_DIAS'])},
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                fill: false,
                tension: 0.2
            }}]
        }},
        options: {{
            plugins: {{
                annotation: {{
                    annotations: {{
                        soporte1: {{
                            type: 'line',
                            yMin: {data['SOPORTE_1']},
                            yMax: {data['SOPORTE_1']},
                            borderColor: 'rgba(0, 255, 0, 0.8)',
                            borderWidth: 2,
                            label: {{
                                enabled: true,
                                content: 'Soporte 1 ({data['SOPORTE_1']}€)',
                                position: 'end'
                            }}
                        }},
                        soporte2: {{
                            type: 'line',
                            yMin: {data['SOPORTE_2']},
                            yMax: {data['SOPORTE_2']},
                            borderColor: 'rgba(0, 200, 0, 0.6)',
                            borderWidth: 2,
                            label: {{
                                enabled: true,
                                content: 'Soporte 2 ({data['SOPORTE_2']}€)',
                                position: 'end'
                            }}
                        }},
                        soporte3: {{
                            type: 'line',
                            yMin: {data['SOPORTE_3']},
                            yMax: {data['SOPORTE_3']},
                            borderColor: 'rgba(0, 150, 0, 0.6)',
                            borderWidth: 2,
                            label: {{
                                enabled: true,
                                content: 'Soporte 3 ({data['SOPORTE_3']}€)',
                                position: 'end'
                            }}
                        }},
                        resistencia: {{
                            type: 'line',
                            yMin: {data['RESISTENCIA']},
                            yMax: {data['RESISTENCIA']},
                            borderColor: 'rgba(255, 99, 132, 1)',
                            borderWidth: 2,
                            label: {{
                                enabled: true,
                                content: 'Resistencia ({data['RESISTENCIA']}€)',
                                position: 'end'
                            }}
                        }},
                        objetivo: {{
                            type: 'line',
                            yMin: {data['PRECIO_OBJETIVO_COMPRA']},
                            yMax: {data['PRECIO_OBJETIVO_COMPRA']},
                            borderColor: 'rgba(255, 206, 86, 1)',
                            borderWidth: 2,
                            borderDash: [6, 6],
                            label: {{
                                enabled: true,
                                content: 'Objetivo ({data['PRECIO_OBJETIVO_COMPRA']}€)',
                                position: 'end',
                                backgroundColor: 'rgba(255, 206, 86, 0.8)'
                            }}
                        }}
                    }}
                }}
            }},
            scales: {{
                y: {{
                    beginAtZero: false,
                    title: {{
                        display: true,
                        text: 'Precio de Cierre (€)'
                    }}
                }},
                x: {{
                    title: {{
                        display: true,
                        text: 'Últimos 30 Días'
                    }}
                }}
            }},
            responsive: true,
            maintainAspectRatio: false
        }}
    }});
}});
</script>
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
<p><strong>{data['NOMBRE_EMPRESA']} ({data['TICKER']})</strong> cotiza actualmente a <strong>{data['PRECIO_ACTUAL']:,}€</strong>, un nivel que considero estratégico. Mi precio objetivo de compra se sitúa en <strong>{data['PRECIO_OBJETIVO_COMPRA']:,}€</strong>. Este último representa el nivel más atractivo para una entrada conservadora. El volumen negociado recientemente, que alcanzó las <strong>{data['VOLUMEN']:,} acciones</strong>, es un factor clave que valida estos movimientos, y será crucial monitorearlo para confirmar la fuerza de cualquier tendencia emergente.</p>

<p>Asignamos una <strong>nota técnica de {data['NOTA_EMPRESA']} sobre 10</strong>. Esta puntuación refleja [elige una de las siguientes opciones basadas en la nota, manteniendo el foco en el análisis técnico]:
    {"una excelente fortaleza técnica y baja volatilidad esperada a corto plazo. La sólida puntuación se basa en la evaluación de indicadores clave de impulso, soporte y resistencia, lo que indica un bajo riesgo técnico en relación con el potencial de crecimiento a corto plazo." if data['NOTA_EMPRESA'] >= 8 else ""}
    {"una fortaleza técnica moderada, con un equilibrio entre potencial y riesgo. Se basa en el comportamiento del gráfico, soportes, resistencias e impulso, sugiriendo una oportunidad que requiere seguimiento." if 6 <= data['NOTA_EMPRESA'] < 8 else ""}
    {"una situación técnica neutral, donde el gráfico no muestra un patrón direccional claro. La puntuación se deriva del análisis de los movimientos de precio y volumen, indicando que es un momento para la observación y no para la acción inmediata." if 5 <= data['NOTA_EMPRESA'] < 6 else ""}
    {"cierta debilidad técnica, con posibles señales de corrección o continuación bajista. La puntuación se basa en los indicadores del gráfico, que muestran una pérdida de impulso alcista y un aumento de la presión vendedora." if 3 <= data['NOTA_EMPRESA'] < 5 else ""}
    {"una debilidad técnica significativa y una posible sobrecompra en el gráfico, lo que sugiere un alto riesgo de corrección. La puntuación se basa en el análisis de los patrones de precio y volumen, indicando que es un momento para la cautela extrema." if data['NOTA_EMPRESA'] < 3 else ""}
Es importante recordar que esta nota técnica que IBEXIA otorga es puramente un reflejo del **análisis del gráfico y sus indicadores técnicos**, y no obedece a la situación financiera o de otro tipo de la empresa. Como profesional, esta nota es mi valoración experta al interpretar el comportamiento del precio y los indicadores.</p>
{chart_html}

<h2>Estrategia de Inversión y Gestión de Riesgos</h2>
<p>Un aspecto crucial en el análisis de corto plazo es la dinámica del impulso de la empresa. Mi evaluación profesional indica que la tendencia actual de nuestra nota técnica es **{data['TENDENCIA_NOTA']}**. Esto sugiere {('un rebote inminente, dado que los indicadores muestran una sobreventa extrema, lo que significa que la acción ha sido \'castigada\' en exceso y hay una alta probabilidad de que los compradores tomen el control, impulsando el precio al alza. Esta situación de sobreventa, sumada al impulso alcista subyacente, nos sugiere que estamos ante el inicio de un rebote significativo.' if data['TENDENCIA_NOTA'] == 'mejorando' and data['NOTA_EMPRESA'] < 6 else '')}
{('una potencial continuación bajista, con los indicadores técnicos mostrando una sobrecompra significativa o una pérdida de impulso alcista. Esto sugiere que la acción podría experimentar una corrección. Es un momento para la cautela y la vigilancia de los niveles de soporte.' if data['TENDENCIA_NOTA'] == 'empeorando' and data['NOTA_EMPRESA'] > 4 else '')}
{('una fase de consolidación o lateralidad, donde los indicadores técnicos no muestran una dirección clara. Es un momento para esperar la confirmación de una nueva tendencia antes de tomar decisiones.' if data['TENDENCIA_NOTA'] == 'estable' else '')}
{f" Calculamos que este impulso podría llevarnos a una potencial zona de {('toma de beneficios o venta' if data['NOTA_EMPRESA'] >= 8 else 'entrada o compra')} en aproximadamente **{data['DIAS_ESTIMADOS_ACCION']}**." if "No disponible" not in data['DIAS_ESTIMADOS_ACCION'] and "Ya en zona" not in data['DIAS_ESTIMADOS_ACCION'] else ("La nota ya se encuentra en una zona de acción clara, lo que sugiere una oportunidad {('de compra' if data['NOTA_EMPRESA'] >= 8 else 'de venta')} inmediata, y por tanto, no se estima un plazo de días adicional." if "Ya en zona" in data['DIAS_ESTIMADOS_ACCION'] else "")}</p>

<p>{volumen_analisis_text}</p>

<p>Basado en nuestro análisis, una posible estrategia de entrada sería considerar una compra cerca {f"del soporte de <strong>{soportes_unicos[0]:,.2f}€</strong>" if len(soportes_unicos) > 0 else ""} o, idealmente, en {f"los <strong>{soportes_unicos[1]:,.2f}€</strong>." if len(soportes_unicos) > 1 else "."} Estos niveles ofrecen una relación riesgo/recompensa atractiva, permitiendo una entrada con mayor margen de seguridad. Para gestionar el riesgo de forma efectiva, se recomienda establecer un stop loss ajustado justo por debajo del soporte más bajo que hemos identificado, por ejemplo, en {f"<strong>{soportes_unicos[-1]:,.2f}€</strong>." if len(soportes_unicos) > 0 else "un nivel apropiado de invalidación."} Este punto actuaría como un nivel de invalidez de nuestra tesis de inversión. Nuestro objetivo de beneficio (Take Profit) a corto plazo se sitúa en la resistencia clave de <strong>{data['RESISTENCIA']:,}€</strong>, lo que representa un potencial de revalorización significativo. Esta configuración de entrada, stop loss y objetivo permite una relación riesgo/recompensa favorable para el inversor, buscando maximizar el beneficio mientras se protege el capital.</p>

<h2>Visión a Largo Plazo y Fundamentales</h2>

<p>En el último ejercicio, los ingresos declarados fueron de <strong>{formatear_numero(data['INGRESOS'])}</strong>, el EBITDA alcanzó <strong>{formatear_numero(data['EBITDA'])}</strong>, y los beneficios netos se situaron en torno a <strong>{formatear_numero(data['BENEFICIOS'])}</strong>. 
En cuanto a su posición financiera, la deuda asciende a <strong>{formatear_numero(data['DEUDA'])}</strong>, y el flujo de caja operativo es de <strong>{formatear_numero(data['FLUJO_CAJA'])}</strong>.</p>

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

<h2>Descargo de Responsabilidad</h2>
<p>Descargo de responsabilidad: Este contenido tiene una finalidad exclusivamente informativa y educativa. No constituye ni debe interpretarse como una recomendación de inversión, asesoramiento financiero o una invitación a comprar o vender ningún activo. La inversión en mercados financieros conlleva riesgos, incluyendo la pérdida total del capital invertido. Se recomienda encarecidamente a cada inversor realizar su propia investigación exhaustiva (due diligence), consultar con un asesor financiero cualificado y analizar cada decisión de forma individual, teniendo en cuenta su perfil de riesgo personal, sus objetivos financieros y su situación económica antes de tomar cualquier decisión de inversión. El rendimiento pasado no es indicativo de resultados futuros.</p>


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
                    server_suggested_delay = 0 # Inicializamos a 0
                    try:
                        match = re.search(r"retry_delay \{\s*seconds: (\d+)", str(e))
                        if match:
                            server_suggested_delay = int(match.group(1))
                    except:
                        pass

                    # Calcula el retraso actual basado en la retirada exponencial o el sugerido por el servidor
                    current_delay = max(initial_delay * (2 ** retries), server_suggested_delay + 1)

                    # Añade jitter (aleatoriedad) para evitar colisiones con otras solicitudes
                    jitter = random.uniform(0.5, 1.5) # Factor aleatorio entre 0.5 y 1.5
                    delay_with_jitter = current_delay * jitter

                    print(f"❌ Cuota de Gemini excedida al generar contenido. Reintentando en {delay_with_jitter:.2f} segundos... (Intento {retries + 1}/{max_retries})")
                    time.sleep(delay_with_jitter) # Usa el retraso con jitter
                    retries += 1
                    # La variable 'delay' ya no se necesita mantener persistente ni multiplicar
                    # porque 'current_delay' se calcula de nuevo en cada intento
                else:
                    print(f"❌ Error al generar contenido con Gemini (no de cuota): {e}")
                    break
        else:  
            print(f"❌ Falló la generación de contenido para {ticker} después de {max_retries} reintentos.")
            
        # --- PAUSA DE 3 MINUTO DESPUÉS DE CADA TICKER ---
        print(f"⏳ Esperando 180 segundos antes de procesar el siguiente ticker...")
        time.sleep(180) # Pausa de 180 segundos entre cada ticker

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
