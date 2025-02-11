import streamlit as st
import pandas as pd
from groq import Groq
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# Configuración de la clave API de Groq
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Inicializar cliente de Groq
client = Groq(api_key = GROQ_API_KEY)

#MODEL = "llama-3.1-8b-instant"
MODEL = "llama-3.3-70b-Versatile"
#MODEL = "Mixtral-8x7b-32768"

# Ajustar el tamaño máximo de tokens para evitar llegar al límite de la API
MAX_TOKENS = 3000


def split_transcript(transcript, chunk_size=3000, overlap=100):
    """
    Divide el texto en fragmentos de tamaño específico para la edición profesional.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )
    return text_splitter.split_text(transcript)


def create_edit_chain():
    """
    Configura la cadena de edición utilizando el modelo GPT-4 para edición profesional.
    """
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL, temperature=0)

    template = """
    Eres un editor experto encargado de transformar transcripciones en texto pulido y profesional para un reporte.
    Tu tarea es editar la siguiente transcripción y devolver un texto editado que sea claro, coherente y adecuado para su comprensión.
    En el texto se menciona organismos públicos, privados, académicos por lo que debes identificar palabras que estén mal transcritas y
    asociarlas a estas temáticas de acuerdo a su contexto.

    Directrices:
    1. Utiliza solo la transcripción proporcionada.
    2. Mantén el contenido y el significado original.
    3. Mejora la claridad y la fluidez del texto.
    4. Corrige errores gramaticales y de puntuación.
    5. Elimina muletillas, repeticiones innecesarias y palabras de relleno.
    6. Organiza el texto en párrafos coherentes.
    7. No hagas los párrafos tan cortos.
    8. Asegúrate de que las transiciones entre ideas sean suaves.
    9. Mantén un tono profesional y adecuado para un libro o reporte.
    10. Si hay algún acrónimo o nombre propio del que tengas dudas señálalo entre paréntesis.
    11. Si hay intervenciones en idiomas distintos al español tradúcelas a español.

    Genera la edición sin ningún comentario adicional.

    Transcripción:
    {text}
    
    Texto editado:
    """

    prompt = ChatPromptTemplate.from_template(template)
    return LLMChain(llm=llm, prompt=prompt)


def edit_transcript_with_ai(transcript):
    """
    Edita la transcripción utilizando la cadena de edición para edición profesional.
    """
    chunks = split_transcript(transcript)
    edit_chain = create_edit_chain()

    edited_chunks = []
    for chunk in chunks:
        result = edit_chain.run(text=chunk)
        edited_chunks.append(result)

    return "\n\n".join(edited_chunks)


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
        "Oportunidades": """
        # Identidad y propósito
        Eres un aplicado miembro del Centro de Innovación UC, experto en elaborar análisis de reuniones a 
        partir de una transcripción. Tu tarea es analizar la información presentada y generar un reporte que 
        identifique las oportunidades de trabajo conjunto con el cliente, posibles líneas de acción y servicios 
        específicos que el Centro puede ofrecer al cliente.
        El análisis debe estar estructurado en base a las cinco dimensiones de innovación que trabaja el Centro 
        de Innovación UC: Estrategia, Cultura, Capacidades Organizacionales, Ecosistema y Asociatividad, y 
        Actividades para la Innovación. Cada dimensión tiene asociados servicios diseñados para resolver brechas 
        de innovación en organizaciones de distintos niveles de madurez. Ten en cuenta estos detalles al redactar 
        tu análisis.
        
        # Secciones del reporte
        1.\tResumen General del Cliente y Contexto de la Reunión:
        \tIdentifica y describe brevemente al cliente y su contexto (industria, tamaño, desafíos generales mencionados) y el propósito de la reunión.
        \tIdentificación de Brechas y Necesidades:
        \t    Examina la transcripción para identificar las necesidades explícitas o implícitas del cliente relacionadas con las cinco dimensiones de innovación: 
        \t        Estrategia: ¿El cliente enfrenta dificultades para alinear su estrategia de innovación con los objetivos de negocio, establecer métricas, o priorizar iniciativas?
        \t        Cultura: ¿Mencionaron desafíos en la adopción de una cultura innovadora, liderazgo o sensibilización de equipos?
        \t        Capacidades Organizacionales: ¿Requieren capacidades específicas como gestión de datos, toma de decisiones basada en analítica avanzada, o formación de equipos?
        \t        Ecosistema: ¿Están interesados en colaboración con startups, universidades, centros tecnológicos, o en la ejecución de proyectos de innovación abierta?
        \t        Actividades: ¿Se discutió la necesidad de implementar pruebas de concepto, programas de I+D, talleres o jornadas específicas de innovación?
        3.\tPropuesta de Servicios y Oportunidades:
            Para cada dimensión en la que se detectaron brechas, vincula las necesidades con servicios específicos del Centro. Los servicios pueden incluir diagnósticos, workshops, 
            programas de formación, desarrollo de hojas de ruta, proyectos de I+D, entre otros.
            Menciona cómo estos servicios ayudarían a cerrar las brechas identificadas o a potenciar las capacidades del cliente.
        4.\tLíneas de Trabajo Recomendadas:
            Propón un conjunto de acciones claras y específicas que el cliente podría implementar junto al Centro. Para cada recomendación, indica: 
            \tObjetivo: ¿Qué problema o necesidad resuelve?
            \tServicio asociado: ¿Qué producto o actividad del Centro es el más adecuado?
            \tImpacto esperado: ¿Cómo beneficiará al cliente (p.ej., mejorar su madurez en innovación, incrementar su capacidad de colaboración, etc.)?
        5.\tPriorización de Recomendaciones y Próximos pasos:
            Ordena las recomendaciones según su relevancia, urgencia e impacto potencial para el cliente.
            Sugiere pasos inmediatos que el cliente podría tomar, como agendar una sesión de diagnóstico o participar en un taller.
        
        # Instrucciones del output:
            - Presenta el análisis en formato de informe estructurado con los siguientes apartados que incluyan todas las secciones del reporte 
            - Asume que el cliente puede tener diferentes niveles de madurez en innovación (No Innovadora, Incipiente, En Desarrollo, Sistemática, Líder). Ajusta las recomendaciones al nivel correspondiente.
            - Usa un lenguaje profesional y persuasivo, adecuado para presentar propuestas a un cliente corporativo.
        """
    }
    return instrucciones.get(tipo_procesamiento, "")


def split_text_intelligently(text, max_tokens=3000):
    """
    Divide el texto en fragmentos manejables de aproximadamente max_tokens tokens sin recursión.
    """
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_token_count = 0

    for paragraph in paragraphs:
        # Aproximar tokens dividiendo la longitud en 4
        paragraph_tokens = len(paragraph) // 4
        
        if current_token_count + paragraph_tokens > max_tokens:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = []
            current_token_count = 0
        
        current_chunk.append(paragraph)
        current_token_count += paragraph_tokens

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    # Si algún fragmento sigue siendo demasiado grande, dividirlo nuevamente de manera iterativa
    final_chunks = []
    for chunk in chunks:
        while len(chunk) // 4 > max_tokens:
            split_index = len(chunk) // 2
            part1, part2 = chunk[:split_index], chunk[split_index:]

            split_point = part1.rfind(' ')
            if split_point != -1:
                part1, part2 = chunk[:split_point], chunk[split_point + 1:]

            final_chunks.append(part1)
            chunk = part2

        final_chunks.append(chunk)

    return final_chunks


def resumir_texto(text):
    """
    Resume el texto dividiéndolo en fragmentos y resumiendo cada uno.
    """
    chunks = split_text_intelligently(text, max_tokens=2500)  # Ajusta un límite más pequeño para permitir tokens de respuesta
    resuming_results = []

    for i, chunk in enumerate(chunks, 1):
        prompt_resumen = f"""
        Eres un experto en resumen de transcripciones. Resume el siguiente fragmento para reducir su longitud manteniendo los puntos clave:

        Fragmento:
        {chunk}

        Resumen:
        """
        try:
            resumen_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt_resumen}
                ],
                model=MODEL,
                max_tokens=500,  # reducir el número de tokens para evitar límites
                temperature=0,
                top_p=0.9,
                stop=None
            )
            resumen = resumen_completion.choices[0].message.content
            resuming_results.append(resumen)
        except Exception as e:
            st.error(f"Error al resumir fragmento {i}: {e}")
            return ""

    return " ".join(resuming_results)


def procesar_transcripcion(text, tipo_procesamiento):
    """
    Procesa la transcripción según el tipo de procesamiento seleccionado.
    """
    if tipo_procesamiento == "Edición Profesional":
        return edit_transcript_with_ai(text)

    # Para otros tipos de procesamiento, mantener el flujo original
    estimated_tokens = len(text) // 4

    # Si estimamos que el texto es muy largo, resumimos primero
    if estimated_tokens > 3000:
       text = resumir_texto(text)

    instrucciones = obtener_instrucciones(tipo_procesamiento)

    prompt = f"""
    Eres un experto procesando transcripciones. Procesa el siguiente texto según las instrucciones dadas.

    Instrucciones específicas:
    {instrucciones}

    Texto a procesar:
    {text}

    Resultado:
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system", 
                    "content": "Eres un procesador experto de transcripciones."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            model=MODEL,
            max_tokens=1500,
            temperature=0,
            top_p=0.9,
            stop=None
        )
        resultado_final = chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error al procesar el texto: {e}")
        return ""

    return resultado_final


def main():
    st.title("\U0001F4DD Procesador Avanzado de Transcripciones")
    st.write("El procesador permite generar un resumen editado de la transcripción, minutas y resúmenes personalizados")
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
        tipo_procesamiento = st.selectbox("Selecciona el tipo de procesamiento", ["Edición Profesional", "Minuta", "Resumen", "Oportunidades"])

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
