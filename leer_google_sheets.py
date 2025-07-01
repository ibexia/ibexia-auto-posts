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
        soporte_1 = soportes[0]
        soporte_2 = soportes[1]
        soporte_3 = soportes[2]

        nota_empresa = round((-(max(min(smi_actual, 60), -60)) + 60) * 10 / 120, 1)

        if nota_empresa <= 2:
            recomendacion = "Vender"
            condicion_rsi = "muy sobrecomprado" # Se mantiene el texto de "RSI" pero se interpreta como estado técnico
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

        precio_objetivo_compra = 0.0
        
        if nota_empresa >= 7:
            precio_objetivo_compra = soporte_1
        else:
            drop_percentage_from_base = (7 - nota_empresa) / 7 * 0.15
            precio_objetivo_compra = soporte_1 * (1 - drop_percentage_from_base)
            
        precio_objetivo_compra = round(precio_objetivo_compra, 2)

        # --- Aplicar traducción a los campos relevantes aquí ---
        expansion_planes_translated = info.get("longBusinessSummary", "N/A")
        expansion_planes_translated = "Información de planes de expansión no disponible o no traducible en este momento."

        acuerdos_translated = info.get("agreements", "No disponibles")
        acuerdos_translated = "Información sobre acuerdos no disponible o no traducible en este momento."

        sentimiento_analistas_translated = info.get("recommendationKey", "N/A")
        sentimiento_analistas_translated = "Sentimiento de analistas no disponible o no traducible."

        
        # --- Fin de la traducción ---

        # --- Lógica para la tendencia y días estimados ---
        smi_history_full = hist['SMI_signal'].dropna()
        smi_history_last_30 = smi_history_full.tail(30).tolist()  # Últimos 30 valores de SMI_signal
        
        # Calcular las últimas 7 notas de la empresa
        notas_historicas_ultimos_30_dias = [round((-(max(min(smi, 60), -60)) + 60) * 10 / 120, 1) for smi in smi_history_last_30]
        
        tendencia_smi = "No disponible"
        dias_estimados_accion = "No disponible"

        if len(smi_history_last_30) >= 2: # Cambiado de 5 a 2 para asegurar que haya al menos 2 puntos para tendencia
            # Calcular la tendencia
            # notas_historicas_last_5 = [round((-(max(min(smi, 60), -60)) + 60) * 10 / 120, 1) for smi in smi_history_last_5] # Esta línea ya no es necesaria con el cambio de nombre de la variable
            
            if len(notas_historicas_ultimos_30_dias) >= 2: # Asegurarse de que hay al menos 2 puntos para la regresión
                # Usar una regresión lineal simple para una estimación más robusta de la tendencia
                x = np.arange(len(notas_historicas_ultimos_30_dias))
                y = np.array(notas_historicas_ultimos_30_dias)
                # Solo si hay suficiente variación para calcular una pendiente significativa
                if len(x) > 1 and np.std(y) > 0.01:
                    slope, intercept = np.polyfit(x, y, 1)
                    tendencia_promedio_diaria = slope
                else: # Si los valores son casi constantes
                    tendencia_promedio_diaria = 0.0
                
                if tendencia_promedio_diaria > 0.1: # umbral pequeño para considerar "mejorando"
                    tendencia_smi = "mejorando"
                elif tendencia_promedio_diaria < -0.1: # umbral pequeño para considerar "empeorando"
                    tendencia_smi = "empeorando"
                else:
                    tendencia_smi = "estable"

                # Estimar días para acción
                target_nota_vender = 2.0
                target_nota_comprar = 8.0

                if nota_empresa <= target_nota_vender:
                    dias_estimados_accion = "Ya en zona de posible venta"
                elif nota_empresa >= target_nota_comprar:
                    dias_estimados_accion = "Ya en zona de posible compra"
                elif tendencia_smi == "estable" or abs(tendencia_promedio_diaria) < 0.01:
                    dias_estimados_accion = "Tendencia estable, sin acción inmediata clara"
                elif tendencia_promedio_diaria < 0: # Nota está bajando, hacia venta
                    diferencia_necesaria = nota_empresa - target_nota_vender
                    # Evitar división por cero
                    if abs(tendencia_promedio_diaria) > 0.01: 
                        dias = diferencia_necesaria / abs(tendencia_promedio_diaria)
                        dias_estimados_accion = f"aprox. {int(max(1, dias))} días para alcanzar zona de venta"
                    else:
                        dias_estimados_accion = "Tendencia muy lenta hacia venta"
                elif tendencia_promedio_diaria > 0: # Nota está subiendo, hacia compra (o recuperándose de sobreventa)
                    diferencia_necesaria = target_nota_comprar - nota_empresa
                    # Evitar división por cero
                    if abs(tendencia_promedio_diaria) > 0.01:
                        dias = diferencia_necesaria / abs(tendencia_promedio_diaria)
                        dias_estimados_accion = f"aprox. {int(max(1, dias))} días para alcanzar zona de compra"
                    else:
                        dias_estimados_accion = "Tendencia muy lenta hacia compra"
        # --- Fin de la lógica para la tendencia y días estimados ---


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
            "EXPANSION_PLANES": expansion_planes_translated,
            "ACUERDOS": acuerdos_translated,
            "SENTIMIENTO_ANALISTAS": sentimiento_analistas_translated,
            "TENDENCIA_SOCIAL": "No disponible",
            "EMPRESAS_SIMILARES": ", ".join(info.get("category", "").split(",")) if info.get("category") else "No disponibles",
            "RIESGOS_OPORTUNIDADES": "No disponibles",
            "TENDENCIA_NOTA": tendencia_smi, # Nuevo campo
            "DIAS_ESTIMADOS_ACCION": dias_estimados_accion, # Nuevo campo
            "NOTAS_HISTORICAS_30_DIAS": notas_historicas_ultimos_30_dias
        }
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
                datasets: [{{
                    label: 'Nota Técnica',
                    data: {json.dumps(notas_historicas_display)},
                    backgroundColor: 'rgba(0, 128, 255, 0.4)',
                    borderColor: 'rgba(0, 128, 255, 1)',
                    borderWidth: 1
                }}]
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
                        display: false
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return context.dataset.label + ': ' + context.parsed.y.toFixed(1);
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
                            text: 'Nota (0-10)'
                        }}
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
Eres un analista técnico profesional con años de experiencia en los mercados financieros. Redacta un análisis completo en formato HTML, ideal para publicarse en un blog financiero.

Usa exclusivamente la información real que aparece en este diccionario JSON:

{json.dumps(data, indent=2, ensure_ascii=False)}

Instrucciones:
- Escribe en primera persona, como un analista humano dando su opinión basada en experiencia.
- Redacta con un estilo natural, variable y sin sonar como una plantilla.
- Solo menciona los campos que tengan datos. Si algo falta o dice "No disponible", ignóralo sin explicarlo.
- El contenido debe tener entre 900 y 1200 palabras, y estar estructurado con etiquetas HTML como <h2> y <p>.
- Usa <strong> para destacar números importantes (precio, nota, soportes, etc.).
- Evita frases genéricas como "no se dispone de información". Mejor omitir esa parte.
- Si la nota técnica es alta (mayor o igual a 8), enfócate más en oportunidades de compra. Si es baja (menor o igual a 2), en señales de venta.
- No empieces con introducciones genéricas. Arranca directamente con el análisis de la empresa.
- Puedes incluir subtítulos como “Análisis Técnico”, “Recomendación”, “Soportes y Resistencias”, “Tendencia”, etc.
- No digas que eres una IA ni que esto fue generado automáticamente.

Incluye también este gráfico al final del texto:

{chart_html}

Y esta tabla de resumen de puntos clave:

{tabla_resumen}

Finaliza con un pequeño párrafo de cierre profesional, que anime a estar atentos al comportamiento de esta empresa.
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
