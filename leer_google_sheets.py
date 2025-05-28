import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
import json
import google.generativeai as genai

def leer_google_sheets():
    # Leer credenciales desde variable de entorno (GitHub Secrets)
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


def generar_contenido_con_gemini(datos):
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise Exception("No se encontró la variable de entorno GEMINI_API_KEY")

print("\nModelos disponibles:")
for modelo in genai.list_models():
    print(f"{modelo.name} -> {modelo.supported_generation_methods}")

# Usa el modelo correcto según lo que imprima arriba
model = genai.GenerativeModel(model_name="models/gemini-1.0-pro")  # Ajusta si tienes acceso a 1.5

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="models/gemini-pro")  # 🔧 Corrección aquí

    # Crear prompt a partir de los datos de Sheets
    prompt = "Crea un texto inspirador con base en estos datos:\n"
    for row in datos:
        prompt += " - " + ", ".join(row) + "\n"

    response = model.generate_content(prompt)

    print("\n🧠 Contenido generado por Gemini:\n")
    print(response.text)

def main():
    datos = leer_google_sheets()
    if datos:
        generar_contenido_con_gemini(datos)

if __name__ == '__main__':
    main()
