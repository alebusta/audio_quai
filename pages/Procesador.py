import streamlit as st
import pandas as pd
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
from collections import deque
from datetime import datetime, timedelta
import hashlib

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

        # INPUT:

        INPUT:

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
        1.	Resumen General del Cliente y Contexto de la Reunión:
        	Identifica y describe brevemente al cliente y su contexto (industria, tamaño, desafíos generales mencionados) y el propósito de la reunión.
        	Identificación de Brechas y Necesidades:
            	Examina la transcripción para identificar las necesidades explícitas o implícitas del cliente relacionadas con las cinco dimensiones de innovación: 
                	Estrategia: ¿El cliente enfrenta dificultades para alinear su estrategia de innovación con los objetivos de negocio, establecer métricas, o priorizar iniciativas?
                	Cultura: ¿Mencionaron desafíos en la adopción de una cultura innovadora, liderazgo o sensibilización de equipos?
                	Capacidades Organizacionales: ¿Requieren capacidades específicas como gestión de datos, toma de decisiones basada en analítica avanzada, o formación de equipos?
                	Ecosistema: ¿Están interesados en colaboración con startups, universidades, centros tecnológicos, o en la ejecución de proyectos de innovación abierta?
                	Actividades: ¿Se discutió la necesidad de implementar pruebas de concepto, programas de I+D, talleres o jornadas específicas de innovación?
        3.	Propuesta de Servicios y Oportunidades:
            Para cada dimensión en la que se detectaron brechas, vincula las necesidades con servicios específicos del Centro. Los servicios pueden incluir diagnósticos, workshops, 
            programas de formación, desarrollo de hojas de ruta, proyectos de I+D, entre otros.
            Menciona cómo estos servicios ayudarían a cerrar las brechas identificadas o a potenciar las capacidades del cliente.
        4.	Líneas de Trabajo Recomendadas:
            Propón un conjunto de acciones claras y específicas que el cliente podría implementar junto al Centro. Para cada recomendación, indica: 
            	Objetivo: ¿Qué problema o necesidad resuelve?
            	Servicio asociado: ¿Qué producto o actividad del Centro es el más adecuado?
            	Impacto esperado: ¿Cómo beneficiará al cliente (p.ej., mejorar su madurez en innovación, incrementar su capacidad de colaboración, etc.)?
        5.	Priorización de Recomendaciones y Próximos pasos:
            Ordena las recomendaciones según su relevancia, urgencia e impacto potencial para el cliente.
            Sugiere pasos inmediatos que el cliente podría tomar, como agendar una sesión de diagnóstico o participar en un taller.
        
        
        # Instrucciones del output:
            - Presenta el análisis en formato de informe estructurado con los siguientes apartados que incluyan todas las secciones del reporte 
            - Asume que el cliente puede tener diferentes niveles de madurez en innovación (No Innovadora, Incipiente, En Desarrollo, Sistemática, Líder). Ajusta las recomendaciones al nivel correspondiente.
            - Usa un lenguaje profesional y persuasivo, adecuado para presentar propuestas a un cliente corporativo.
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
            
            if not completion.choices:
                raise ValueError("Respuesta vacía de la API")
                
            result = completion.choices[0].message.content
            
            if not isinstance(result, str) or len(result.strip()) == 0:
                raise ValueError("Respuesta no válida o vacía")
                
            return result
            
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                time.sleep(5 * (attempt + 1))
                continue
            st.error(f"Intento {attempt + 1} fallado: {str(e)}")
            if attempt == max_retries - 1:
                raise e

@st.cache_data(show_spinner=False, ttl=3600)
def generar_resumen_base(_text, _rate_limiter):
    """Genera y cachea el resumen base usando el texto como clave de caché"""
    chunks = create_chunks(_text, max_tokens=3000)
    total_chunks = len(chunks)
    
    summary_chunks = []
    for i, chunk in enumerate(chunks):
        try:
            prompt = f"""Resume este fragmento conservando información esencial:
            {chunk}
            
            Instrucciones:
            - Formato: Lista concisa de puntos clave
            - Mantener conceptos importantes
            - Conservar datos críticos"""
            
            summary = process_with_rate_limit(prompt, _rate_limiter)
            
            if not isinstance(summary, str) or len(summary.strip()) == 0:
                continue
                
            summary_chunks.append(summary)
            
        except Exception as e:
            st.error(f"Error en fragmento {i+1}: {str(e)}")
            continue
    
    combined_summary = "\n\n".join(summary_chunks)
    
    # Resumen recursivo si es necesario
    while estimate_tokens(combined_summary) > 3000:
        chunks = create_chunks(combined_summary, max_tokens=3000)
        new_summaries = []
        for chunk in chunks:
            try:
                prompt = f"Sintetiza en 3-5 puntos clave esenciales:\n{chunk}"
                summary = process_with_rate_limit(prompt, _rate_limiter)
                if summary:
                    new_summaries.append(summary)
            except Exception as e:
                continue
        combined_summary = "\n".join(new_summaries)
    
    return combined_summary

def procesar_final(tipo_procesamiento, resumen_base, rate_limiter):
    """Procesa el resumen base cacheado según el tipo seleccionado"""
    instrucciones = obtener_instrucciones(tipo_procesamiento)
    
    final_prompt = f"""
    INSTRUCCIONES PRINCIPALES:
    {instrucciones}
    
    CONTEXTO RESUMIDO:
    {resumen_base}
    
    REQUERIMIENTOS:
    - Generar documento final único
    - Mantener estructura solicitada
    - Formato adecuado al tipo de documento
    """
    
    return process_with_rate_limit(final_prompt, rate_limiter)

def main():
    st.title("📝 Procesador Inteligente con Caché")
    st.write("Procesamiento optimizado con resumen base cacheado")
    
    @st.cache_data
    def load_csv(file):
        return pd.read_csv(file)
    
    st.sidebar.markdown("""
    **Flujo de trabajo:**
    1. Cargar texto/CSV
    2. Generar resumen base (se cachea)
    3. Seleccionar tipo de procesamiento
    4. Generar resultado final
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
        
        rate_limiter = TokenRateLimiter()
        
        # Generar o recuperar resumen base cacheado
        with st.spinner('Generando/Recuperando resumen base...'):
            resumen_base = generar_resumen_base(text, rate_limiter)
        
        tipo_procesamiento = st.selectbox(
            "Tipo de procesamiento",
            ["Edición Profesional", "Minuta", "Resumen", "Oportunidades"]
        )
        
        if st.button("Generar Resultado"):
            with st.spinner('Procesando documento final...'):
                start_time = time.time()
                
                try:
                    resultado = procesar_final(tipo_procesamiento, resumen_base, rate_limiter)
                    
                    if not resultado:
                        raise ValueError("Resultado final vacío")
                        
                    st.subheader("Resultado Final")
                    st.markdown(resultado)
                    
                    st.download_button(
                        label="Descargar Resultado",
                        data=resultado,
                        file_name=f"{tipo_procesamiento.replace(' ', '_')}.md",
                        mime="text/markdown"
                    )
                    
                    st.info(f"Tiempo total: {time.time()-start_time:.2f} segundos")
                    
                except Exception as e:
                    st.error(f"Error en procesamiento final: {str(e)}")

if __name__ == "__main__":
    main()
