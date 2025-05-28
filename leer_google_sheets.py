import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.oauth2 import service_account
from googleapiclient.discovery import build
import google.generativeai as genai

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

    range_name = os.getenv('RANGE_NAME', 'A1:C10')

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

    return values


def enviar_email(texto_generado):
    remitente = "xumkox@gmail.com"
    destinatario = "xumkox@gmail.com"
    asunto = "Contenido generado por Gemini"
    password = "kdgz lvdo wqvt vfkt"

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


def generar_contenido_con_gemini(datos):
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise Exception("No se encontró la variable de entorno GEMINI_API_KEY")

    genai.configure(api_key=api_key)

    print("\nModelos disponibles:")
    for modelo in genai.list_models():
        print(f"{modelo.name} -> {modelo.supported_generation_methods}")

    model = genai.GenerativeModel(model_name="models/gemini-2.0-flash-lite")

    prompt = "Crea un texto inspirador con base en estos datos:\n"
    for row in datos:
        prompt += " - " + ", ".join(row) + "\n"

    response = model.generate_content(prompt)

    print("\n🧠 Contenido generado por Gemini:\n")
    print(response.text)

    enviar_email(response.text)


def main():
    datos = leer_google_sheets()
    if datos:
        generar_contenido_con_gemini(datos)


if __name__ == '__main__':
    main()
