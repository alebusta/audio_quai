import streamlit as st
import pandas as pd
from groq import Groq
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import time
from collections import deque
from datetime import datetime, timedelta

# Configuración de la clave API de Groq
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.3-70b-Versatile"

class TokenRateLimiter:
    def __init__(self, tokens_per_minute=6000):
        self.tokens_per_minute = tokens_per_minute
        self.token_history = deque()
    
    def wait_if_needed(self, tokens_needed):
        """
        Espera si es necesario para respetar el límite de tokens por minuto.
        """
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Limpiar historial antiguo
        while self.token_history and self.token_history[0][0] < minute_ago:
            self.token_history.popleft()
        
        # Calcular tokens usados en el último minuto
        tokens_used = sum(tokens for _, tokens in self.token_history)
        
        if tokens_used + tokens_needed > self.tokens_per_minute:
            # Calcular tiempo de espera necesario
            oldest_time = self.token_history[0][0] if self.token_history else now
            wait_seconds = (oldest_time + timedelta(minutes=1) - now).total_seconds()
            if wait_seconds > 0:
                time.sleep(wait_seconds)
                self.token_history.clear()
        
        # Registrar nuevo uso de tokens
        self.token_history.append((now, tokens_needed))

def estimate_tokens(text):
    """
    Estima el número de tokens en un texto.
    Usa una aproximación conservadora de 4 caracteres por token.
    """
    return len(text) // 4

def obtener_instrucciones(tipo_procesamiento):
    """
    Devuelve las instrucciones específicas según el tipo de procesamiento.
    """
    instrucciones = {
        "Edición Profesional": """
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
        """,
        
        "Minuta": """
        Directrices para minutas:
        - Extrae puntos clave de la reunión
        - Identifica intervenciones importantes
        - Resume las principales discusiones
        - Destaca decisiones y compromisos
        - Señala los próximos pasos
        """,
        
        "Resumen": """
        # IDENTITY and PURPOSE

        You are an expert content summarizer. You take content in and output a Markdown formatted summary using the format below.

        Take a deep breath and think step by step about how to best accomplish this goal using the following steps.

        # OUTPUT SECTIONS

        - Combine all of your understanding of the content into a single, 20-word sentence in a section called ONE SENTENCE SUMMARY:.

        - Output the 10 most important points of the content as a list with no more than 15 words per point into a section called MAIN POINTS:.

        - Output a list of the 5 best takeaways from the content in a section called TAKEAWAYS:.

        # OUTPUT INSTRUCTIONS

        - Output should be in the same language as input (i.e if input is in Spanish output should be in Spanish, including sections names)
        - Create the output using the formatting above.
        - You only output human readable Markdown.
        - Output numbered lists, not bullets.
        - Do not output warnings or notes—just the requested sections.
        - Do not repeat items in the output sections.
        - Do not start items with the same opening words.
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
        1. Resumen General del Cliente y Contexto de la Reunión:
           Identifica y describe brevemente al cliente y su contexto (industria, tamaño, desafíos generales mencionados) y el propósito de la reunión.
        
        2. Identificación de Brechas y Necesidades:
           Examina la transcripción para identificar las necesidades explícitas o implícitas del cliente relacionadas con las cinco dimensiones de innovación:
           - Estrategia: ¿El cliente enfrenta dificultades para alinear su estrategia de innovación con los objetivos de negocio, establecer métricas, o priorizar iniciativas?
           - Cultura: ¿Mencionaron desafíos en la adopción de una cultura innovadora, liderazgo o sensibilización de equipos?
           - Capacidades Organizacionales: ¿Requieren capacidades específicas como gestión de datos, toma de decisiones basada en analítica avanzada, o formación de equipos?
           - Ecosistema: ¿Están interesados en colaboración con startups, universidades, centros tecnológicos, o en la ejecución de proyectos de innovación abierta?
           - Actividades: ¿Se discutió la necesidad de implementar pruebas de concepto, programas de I+D, talleres o jornadas específicas de innovación?
        
        3. Propuesta de Servicios y Oportunidades:
           Para cada dimensión en la que se detectaron brechas, vincula las necesidades con servicios específicos del Centro.
           Los servicios pueden incluir diagnósticos, workshops, programas de formación, desarrollo de hojas de ruta, proyectos de I+D, entre otros.
           Menciona cómo estos servicios ayudarían a cerrar las brechas identificadas o a potenciar las capacidades del cliente.
        
        4. Líneas de Trabajo Recomendadas:
           Propón acciones claras y específicas que el cliente podría implementar junto al Centro. Para cada recomendación, indica:
           - Objetivo: ¿Qué problema o necesidad resuelve?
           - Servicio asociado: ¿Qué producto o actividad del Centro es el más adecuado?
           - Impacto esperado: ¿Cómo beneficiará al cliente?
        
        5. Priorización de Recomendaciones y Próximos pasos:
           Ordena las recomendaciones según su relevancia, urgencia e impacto potencial para el cliente.
           Sugiere pasos inmediatos que el cliente podría tomar.
        """
    }
    return instrucciones.get(tipo_procesamiento, "")

def create_chunks(text, max_tokens=2000, overlap_tokens=100):
    """
    Divide el texto en fragmentos respetando el límite de tokens.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens * 4,  # Convertir tokens a caracteres aproximados
        chunk_overlap=overlap_tokens * 4,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)

def process_with_rate_limit(prompt, rate_limiter, max_retries=3):
    """
    Procesa un prompt respetando el rate limit.
    """
    estimated_tokens = estimate_tokens(prompt)
    rate_limiter.wait_if_needed(estimated_tokens)
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Eres un procesador experto de transcripciones."},
                    {"role": "user", "content": prompt}
                ],
                model=MODEL,
                max_tokens=1500,
                temperature=0,
                top_p=0.9
            )
            return completion.choices[0].message.content
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))  # Backoff exponencial
                    continue
            raise e

def summarize_long_text(text, tipo_procesamiento, rate_limiter):
    """
    Resume textos largos en múltiples pasos si es necesario.
    """
    chunks = create_chunks(text, max_tokens=3000)
    summaries = []
    
    for chunk in chunks:
        prompt = f"""
        Resume este fragmento manteniendo los puntos más importantes:

        {chunk}
        """
        summary = process_with_rate_limit(prompt, rate_limiter)
        summaries.append(summary)
    
    # Combinar resúmenes en un resultado final
    combined_summary = "\n\n".join(summaries)
    final_prompt = f"""
    Genera el resultado final siguiendo el formato original:

    {obtener_instrucciones(tipo_procesamiento)}

    Basado en:
    {combined_summary}
    """
    
    return process_with_rate_limit(final_prompt, rate_limiter)

def process_chunks(chunks, tipo_procesamiento, progress_bar=None):
    """
    Procesa múltiples fragmentos respetando el rate limit.
    """
    rate_limiter = TokenRateLimiter()
    results = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        instrucciones = obtener_instrucciones(tipo_procesamiento)
        # Reducir el tamaño del prompt para el procesamiento individual
        prompt = f"""
        Procesa este fragmento según las instrucciones. Mantén la continuidad si es un fragmento intermedio.

        Instrucciones: {instrucciones}

        Texto (parte {i+1}/{total_chunks}):
        {chunk}
        """
        
        try:
            result = process_with_rate_limit(prompt, rate_limiter)
            results.append(result)
            if progress_bar:
                progress_bar.progress((i + 1) / total_chunks)
        except Exception as e:
            st.error(f"Error en fragmento {i+1}: {str(e)}")
            time.sleep(5)  # Esperar antes de continuar
            continue

    # Combinar resultados según el tipo de procesamiento
    combined_text = "\n\n".join(results)
    
    # Para tipos que necesitan un procesamiento final
    if tipo_procesamiento in ["Resumen", "Minuta", "Oportunidades"]:
        # Dividir el texto combinado si es necesario
        if estimate_tokens(combined_text) > 4000:
            combined_text = summarize_long_text(combined_text, tipo_procesamiento, rate_limiter)
    
    return combined_text

def procesar_transcripcion(text, tipo_procesamiento):
    """
    Versión mejorada del procesador de transcripciones.
    """
    # Mostrar barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Dividir en fragmentos más pequeños
        status_text.text("Dividiendo el texto en fragmentos...")
        chunks = create_chunks(text)
        
        # Procesar los fragmentos
        status_text.text("Procesando fragmentos...")
        resultado = process_chunks(chunks, tipo_procesamiento, progress_bar)
        
        # Limpiar indicadores de progreso
        progress_bar.empty()
        status_text.empty()
        
        return resultado
        
    except Exception as e:
        st.error(f"Error durante el procesamiento: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        return None
        
def main():
    st.title("📝 Procesador Avanzado de Transcripciones")
    st.write("El procesador permite generar un resumen editado de la transcripción, minutas y resúmenes personalizados")
    
    # Configuración de caché para mejorar el rendimiento
    @st.cache_data
    def load_csv(file):
        return pd.read_csv(file)
    
    st.sidebar.write("""
                Pasos:
                1. Carga un archivo csv o pega un texto
                2. Una vez cargado el archivo o pegado el texto selecciona el tipo de procesamiento
                3. Descarga el resultado
     """)

    input_type = st.sidebar.radio("Selecciona el tipo de entrada", ["Archivo CSV", "Texto directo"])
    
    if input_type == "Archivo CSV":
        uploaded_file = st.sidebar.file_uploader("Cargar transcripción CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = load_csv(uploaded_file)
                text = df['text'].str.cat(sep=' ')
            except Exception as e:
                st.error(f"Error al leer el archivo CSV: {e}")
                return
    else:
        text = st.sidebar.text_area("Pega tu transcripción aquí", height=300)
    
    if 'text' in locals() and text:
        st.info(f"Longitud del texto: {len(text)} caracteres")
        
        tipo_procesamiento = st.selectbox(
            "Selecciona el tipo de procesamiento",
            ["Edición Profesional", "Minuta", "Resumen", "Oportunidades"]
        )
        
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
