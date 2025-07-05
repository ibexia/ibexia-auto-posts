<script>
document.addEventListener('DOMContentLoaded', function () {{
    var ctx = document.getElementById('divergenciaChart').getContext('2d');

    // Normalizamos el precio entre 0 y 10 para compararlo visualmente con la nota
    var preciosOriginales = {{ {json.dumps(data['CIERRES_30_DIAS'])} }};
    var notasOriginales = {{ {json.dumps(data['NOTAS_HISTORICAS_30_DIAS'])} }};

    var minPrecio = Math.min(...preciosOriginales);
    var maxPrecio = Math.max(...preciosOriginales);

    var preciosNormalizados;
    if (maxPrecio === minPrecio) {{
        // Si el precio es constante, normalizamos a un valor medio (ej. 5)
        preciosNormalizados = preciosOriginales.map(function() {{ return 5; }}); 
    }} else {{
        preciosNormalizados = preciosOriginales.map(function(p) {{
            return ((p - minPrecio) / (maxPrecio - minPrecio)) * 10;
        }});
    }}

    var labels = {{ {json.dumps([(datetime.today() - timedelta(days=29 - i)).strftime("%d/%m") for i in range(30)])} }};

    new Chart(ctx, {{
        type: 'line',
        data: {{
            labels: labels,
            datasets: [
                {{
                    label: 'Nota Técnica (0-10)',
                    data: notasOriginales,
                    borderColor: 'rgba(0, 128, 255, 1)',
                    backgroundColor: 'rgba(0, 128, 255, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.2,
                    yAxisID: 'y'
                }},
                {{
                    label: 'Precio (normalizado 0-10)',
                    data: preciosNormalizados,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.2,
                    yAxisID: 'y'
                }}
            ]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                tooltip: {{
                    mode: 'index',
                    intersect: false
                }},
                legend: {{
                    display: true
                }}
            }},
            scales: {{
                y: {{
                    beginAtZero: true,
                    max: 10,
                    title: {{
                        display: true,
                        text: 'Escala 0-10 (Nota y Precio Normalizado)'
                    }}
                }},
                x: {{
                    title: {{
                        display: true,
                        text: 'Últimos 30 Días'
                    }}
                }}
            }}
        }}
    }});
}});
</script>
