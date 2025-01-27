import streamlit as st
import pandas as pd
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        while self.token_history and self.token_history[0][0] < minute_ago:
            self.token_history.popleft()
        
        tokens_used = sum(tokens for _, tokens in self.token_history)
        
        if tokens_used + tokens_needed > self.tokens_per_minute:
            oldest_time = self.token_history[0][0] if self.token_history else now
            wait_seconds = (oldest_time + timedelta(minutes=1) - now).total_seconds()
            if wait_seconds > 0:
                time.sleep(wait_seconds)
                self.token_history.clear()
        
        self.token_history.append((now, tokens_needed))

def estimate_tokens(text):
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens * 4,
        chunk_overlap=overlap_tokens * 4,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)

def process_with_rate_limit(prompt, rate_limiter, max_retries=3):
    estimated_tokens = estimate_tokens(prompt)
    rate_limiter.wait_if_needed(estimated_tokens)
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Eres un asistente experto en procesamiento de textos."},
                    {"role": "user", "content": prompt}
                ],
                model=MODEL,
                max_tokens=3000,
                temperature=0.3,
                top_p=0.9
            )
            return completion.choices[0].message.content
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                time.sleep(5 * (attempt + 1))
                continue
            raise e

def procesar_transcripcion(text, tipo_procesamiento):
    progress_bar = st.progress(0)
    status_text = st.empty()
    rate_limiter = TokenRateLimiter()
    
    try:
        # Fase 1: Crear resumen base consolidado
        status_text.text("Preprocesando texto...")
        chunks = create_chunks(text, max_tokens=3000)
        total_chunks = len(chunks)
        
        summary_chunks = []
        for i, chunk in enumerate(chunks):
            status_text.text(f"Resumiendo fragmento {i+1}/{total_chunks}...")
            prompt = f"""Resume este fragmento conservando solo la información esencial y puntos clave:
            {chunk}
            
            Instrucciones:
            - Eliminar detalles redundantes
            - Mantener nombres propios y conceptos importantes
            - Conservar fechas, acuerdos y decisiones
            - Formato: Lista concisa de puntos clave"""
            
            summary = process_with_rate_limit(prompt, rate_limiter)
            summary_chunks.append(summary)
            progress_bar.progress((i + 1) / (total_chunks * 2))
        
        combined_summary = "\n\n".join(summary_chunks)
        
        # Resumen recursivo si es necesario
        while estimate_tokens(combined_summary) > 3000:
            status_text.text("Optimizando resumen base...")
            chunks = create_chunks(combined_summary, max_tokens=3000)
            new_summaries = []
            for chunk in chunks:
                prompt = f"""Sintetiza en 3-5 puntos clave esenciales:
                {chunk}"""
                summary = process_with_rate_limit(prompt, rate_limiter)
                new_summaries.append(summary)
            combined_summary = "\n".join(new_summaries)
        
        # Fase 2: Generación del documento final
        status_text.text("Creando documento final...")
        instrucciones = obtener_instrucciones(tipo_procesamiento)
        
        final_prompt = f"""
        INSTRUCCIONES PRINCIPALES:
        {instrucciones}
        
        CONTEXTO RESUMIDO:
        {combined_summary}
        
        REQUERIMIENTOS FINALES:
        - Generar un único documento cohesivo
        - Mantener estructura solicitada
        - Incluir todos los puntos clave
        - Evitar repeticiones
        - Usar formato adecuado para el tipo de documento
        """
        
        resultado_final = process_with_rate_limit(final_prompt, rate_limiter)
        progress_bar.progress(1.0)
        
        return resultado_final
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None
    finally:
        progress_bar.empty()
        status_text.empty()

def main():
    st.title("📝 Procesador Unificado de Transcripciones")
    st.write("Procesamiento inteligente de textos largos con salida unificada")
    
    @st.cache_data
    def load_csv(file):
        return pd.read_csv(file)
    
    st.sidebar.markdown("""
    **Instrucciones:**
    1. Sube un CSV o pega el texto
    2. Selecciona el tipo de procesamiento
    3. Espera el resultado (puede tardar varios minutos)
    4. Descarga el resultado final
    """)
    
    input_type = st.sidebar.radio("Tipo de entrada", ["Archivo CSV", "Texto directo"])
    text = ""
    
    if input_type == "Archivo CSV":
        uploaded_file = st.sidebar.file_uploader("Subir CSV", type=['csv'])
        if uploaded_file:
            try:
                df = load_csv(uploaded_file)
                text = df['text'].str.cat(sep=' ')
            except Exception as e:
                st.error(f"Error leyendo CSV: {e}")
                return
    else:
        text = st.sidebar.text_area("Pega tu texto aquí", height=300)
    
    if text:
        st.info(f"Texto cargado: {len(text)} caracteres")
        
        tipo_procesamiento = st.selectbox(
            "Tipo de procesamiento",
            ["Edición Profesional", "Minuta", "Resumen", "Oportunidades"]
        )
        
        if st.button("Iniciar Procesamiento"):
            with st.spinner('Procesando... (No recargar la página)'):
                start_time = time.time()
                resultado = procesar_transcripcion(text, tipo_procesamiento)
                
                if resultado:
                    st.subheader("Resultado Final")
                    st.markdown(resultado)
                    
                    st.download_button(
                        label="Descargar Resultado",
                        data=resultado,
                        file_name=f"{tipo_procesamiento.replace(' ', '_')}.md",
                        mime="text/markdown"
                    )
                    
                    st.info(f"Tiempo total: {time.time()-start_time:.2f} segundos")

if __name__ == "__main__":
    main()
