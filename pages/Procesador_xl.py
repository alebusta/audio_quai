import streamlit as st
import pandas as pd
import os
import google.generativeai as genai

# Configuración de la clave API de Groq


GEMMINI_API_KEY = st.secrets["GEMMINI_API_KEY"]

# Configurar API de Gemini
genai.configure(api_key=GEMMINI_API_KEY) 

model = genai.GenerativeModel("gemini-1.5-flash")



def obtener_instrucciones(tipo_procesamiento):
    instrucciones = {
        "Minuta": """
        Directrices para minutas:
        - Extrae puntos clave de la reunión
        - Identifica intervenciones importantes
        - Resume las principales discusiones
        - Destaca decisiones y compromisos
        - Señala los próximos pasos
        """,
        "Resumen": """
        Eres un experto en resumir contenido. Tomas el contenido y generas un resumen formateado en Markdown usando el siguiente formato.
        Toma un respiro profundo y piensa paso a paso cómo lograr mejor este objetivo siguiendo los siguientes pasos.

        Secciones de salida:
        - Combina todo tu entendimiento del contenido en dos oraciones de 20 palabras en una sección llamada RESUMEN:.
        - Extrae los 10 puntos más importantes del contenido como una lista con no más de 15 palabras por punto en una sección llamada PUNTOS PRINCIPALES:.
        - Genera una lista de las 5 mejores conclusiones del contenido en una sección llamada CONCLUSIONES CLAVE:.

        # INSTRUCCIONES DE SALIDA

        - Responde en el idioma en que esta el texto, incluyendo los nombres de las secciones.
        - Crea la salida usando las instrucciones indicadasa.
        - Solo genera Markdown legible para humanos.
        - Usa listas numeradas, no viñetas.
        - No incluyas advertencias o notas, solo las secciones solicitadas.
        - Importante no repetir elementos en las secciones de salida.
        - No comiences los elementos con las mismas palabras iniciales.
        - Entrega lo indicado en las instrucciones sin ningún tipo de comentario adicional.

        """,
    }
    return instrucciones.get(tipo_procesamiento, "")


def procesar_transcripcion(text, tipo_procesamiento):
    """
    Procesa la transcripción según el tipo de procesamiento seleccionado.
    """
    instrucciones = obtener_instrucciones(tipo_procesamiento)

    prompt = f"""
    Eres un experto analizando transcripciones. Procesa el siguiente texto según las instrucciones dadas.

    Instrucciones específicas:
    {instrucciones}

    Texto a procesar:
    {text}

    Resultado:
    """

    try:
        chat_completion = model.generate_content(prompt,
                                  generation_config=genai.GenerationConfig(
                                      max_output_tokens=2000,
                                      temperature=0
                                  ))
        resultado_final = chat_completion.text
    except Exception as e:
        st.error(f"Error al procesar el texto: {e}")
        return ""

    return resultado_final


def main():
    st.title("\U0001F4DD Procesador Avanzado de Transcripciones para textos grandes")
    st.write("El procesador permite generar minutas y resúmenes personalizados")
    st.sidebar.write("""
                Pasos:
                1. Carga un archivo csv o pega un texto
                2. Una vez cargado el archivo o pegado el texto selecciona el tipo de procesamiento
                3. Descarga el resultado
     """)

    input_type = st.sidebar.radio("Selecciona el tipo de entrada", ["Archivo CSV", "Texto directo"])

    text = ""

    if input_type == "Archivo CSV":
        uploaded_file = st.sidebar.file_uploader("Cargar transcripción CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                text = df['text'].str.cat(sep=' ')
            except Exception as e:
                st.error(f"Error al leer el archivo CSV: {e}")
                return
    else:
        text = st.sidebar.text_area("Pega tu transcripción aquí", height=300)

    if text:
        st.info(f"Longitud del texto: {len(text)} caracteres")
        tipo_procesamiento = st.selectbox("Selecciona el tipo de procesamiento", ["Minuta", "Resumen"])

        if st.button("Procesar Transcripción"):
            with st.spinner('Procesando transcripción... puede tardar varios minutos...'):
                resultado = procesar_transcripcion(text, tipo_procesamiento)

            if resultado:
                st.subheader("Resultado:")
                st.write(resultado)
                st.download_button(
                    label="Descargar Resultado",
                    data=resultado,
                    file_name=f"transcripcion_procesada_{tipo_procesamiento}.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
