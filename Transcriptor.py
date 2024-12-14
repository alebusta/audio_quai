import os
import shutil
import logging
import traceback
from typing import List, Dict
from groq import Groq
from pydub import AudioSegment
import pandas as pd
from datetime import timedelta
import streamlit as st
import base64
import json

from streamlit_javascript import st_javascript

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

logging.basicConfig(level=logging.INFO)

def format_timestamp(milliseconds: int) -> str:
    """
    Convierte milisegundos a formato HH:MM:SS
    """
    seconds = int(milliseconds / 1000)
    return str(timedelta(seconds=seconds))

def transcribe_with_groq(file_path: str, start_time: int) -> List[Dict]:
    """
    Transcribe un archivo de audio usando el servicio de transcripción de Groq.

    Args:
        file_path (str): Ruta al archivo de audio que necesita ser transcrito.
        start_time (int): Tiempo de inicio del segmento en milisegundos.
    Returns:
        List[Dict]: Lista de diccionarios con la transcripción y timestamps.
    """
    client = Groq(api_key = GROQ_API_KEY)
    filename = os.path.basename(file_path)

    try:
        with open(file_path, "rb") as file:
            result = client.audio.transcriptions.create(
                file=(filename, file.read()),
                model="whisper-large-v3",
                response_format="verbose_json"
            )

        segments = []
        for segment in result.segments:
            segment_data = {
                'start': start_time + int(segment['start'] * 1000),
                'end': start_time + int(segment['end'] * 1000),
                'text': segment['text'].strip()
            }
            segments.append(segment_data)

        logging.info(f"Segmentos transcritos: {segments}")
        return segments

    except Exception as e:
        logging.error(f"Error durante la transcripción: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def create_audio_chunks(audio_file: str, chunk_size: int, temp_dir: str) -> List[Dict]:
    """
    Divide un archivo de audio en segmentos más pequeños.

    Args:
        audio_file (str): Ruta al archivo de audio que necesita ser dividido.
        chunk_size (int): Duración de cada segmento en milisegundos.
        temp_dir (str): Directorio donde se almacenarán los segmentos temporales.
    Returns:
        List[Dict]: Lista de diccionarios con información de los chunks creados.
    """
    os.makedirs(temp_dir, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]

    try:
        audio = AudioSegment.from_file(audio_file)
    except Exception as e:
        logging.error(f"create_audio_chunks falló al cargar el archivo de audio {audio_file}: {e}")
        logging.error(traceback.format_exc())
        return []

    start = 0
    end = chunk_size
    counter = 0
    chunk_files = []

    while start < len(audio):
        chunk = audio[start:end]
        chunk_file_path = os.path.join(temp_dir, f"{counter}_{file_name}.mp3")
        try:
            chunk.export(chunk_file_path, format="mp3")
            chunk_files.append({
                'file_path': chunk_file_path,
                'start_time': start
            })
        except Exception as e:
            error_message = f"create_audio_chunks falló al exportar el segmento {counter}: {e}"
            logging.error(error_message)
            logging.error(traceback.format_exc())
            raise Exception(error_message)
        start += chunk_size
        end += chunk_size
        counter += 1
    return chunk_files

def transcribe_local_audio(audio_file: str, chunk_size: int, temp_dir: str = "temp_chunks") -> pd.DataFrame:
    """
    Transcribe un archivo de audio local y retorna un DataFrame con timestamps.

    Args:
        audio_file (str): Ruta al archivo de audio local (MP3 o MP4).
        chunk_size (int): Duración de cada segmento en milisegundos (por defecto 25 minutos).
        temp_dir (str): Directorio para almacenar los segmentos temporales.
    Returns:
        pd.DataFrame: DataFrame con columnas start_time, end_time y text.
    """
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"No se encontró el archivo de audio: {audio_file}")

    try:
        chunk_files = create_audio_chunks(audio_file, chunk_size, temp_dir)

        transcripts = []
        for chunk_info in chunk_files:
            try:
                logging.info(f"Transcribiendo {chunk_info['file_path']}")
                transcript_data = transcribe_with_groq(
                    chunk_info['file_path'],
                    chunk_info['start_time']
                )
                transcripts.extend(transcript_data)
            except Exception as e:
                error_message = f"Falló la transcripción del archivo {chunk_info['file_path']}: {e}"
                logging.error(error_message)
                logging.error(traceback.format_exc())
                raise Exception(error_message)

        # Crear DataFrame y formatear timestamps
        df = pd.DataFrame(transcripts)
        df['start_time'] = df['start'].apply(format_timestamp)
        df['end_time'] = df['end'].apply(format_timestamp)

        # Reordenar y limpiar columnas
        df = df[['start', 'end', 'start_time', 'end_time', 'text']]

    finally:
        # Limpieza de archivos temporales
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logging.error(f"Error al eliminar el directorio temporal {temp_dir}: {e}")

    return df

def main(): 
    st.title("Transcripción de Audio a Texto")
    st.write("Utiliza esta sección para cargar un archivo de audio y transcribirlo (máximo una hora de audio).")
    #st.set_option('server.maxUploadSize', 300)  # Aumentar a 1000 MB (1 GB)
    st.sidebar.write(""" 
               Pasos:
               1. Sube un archivo de audio
               2. La aplicación iniciará la transcripción (puede durar varios minutos)
               3. Revisa la transcripción y edita si es necesario
               4. Una vez terminada la revisión descarga el archivo corregido
               """)
    audio_file = st.sidebar.file_uploader("Subir archivo de audio", type=["mp3", "mp4", "wav","m4a"])

    if audio_file is not None:
        # Crear directorio temporal si no existe
        os.makedirs("temp_dir", exist_ok=True)
        # Guardar archivo subido temporalmente
        temp_audio_path = os.path.join("temp_dir", audio_file.name)
        with open(temp_audio_path, "wb") as f:
            f.write(audio_file.read())

        # Determinar tamaño del archivo y ajustar el chunk_size
        #file_size_mb = os.path.getsize(temp_audio_path) / (1024 * 1024)
        file_length = len(AudioSegment.from_file(temp_audio_path)) / (1000*60)
        #chunk_size = 40 * 60000 if file_size_mb > 25 else len(AudioSegment.from_file(temp_audio_path))
        chunk_size = 10 * 60000 if file_length > 10 else len(AudioSegment.from_file(temp_audio_path))

        # Transcribir el audio
        df_transcription = transcribe_local_audio(temp_audio_path, chunk_size=chunk_size)

        # Preparar los datos de transcripción
        df_transcription['start_seconds'] = df_transcription['start'] / 1000
        df_transcription['end_seconds'] = df_transcription['end'] / 1000

        # Leer el archivo de audio y codificarlo en base64
        with open(temp_audio_path, "rb") as f:
            audio_bytes = f.read()
            base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

        # Construir el HTML de la transcripción
        transcription_html = ""
        sentence_count = 0
        current_paragraph = ""
        
        for idx, row in df_transcription.iterrows():
            start = row['start_seconds']
            end = row['end_seconds']
            text = row['text']
            
            current_paragraph += f"<span data-start='{start}' data-end='{end}' contenteditable='true'>{text} </span>"
            sentence_count += 1
            
            # Cada 5 oraciones, crear un nuevo párrafo
            if sentence_count >= 10:
                transcription_html += f"<p>{current_paragraph}</p>"
                current_paragraph = ""
                sentence_count = 0
        
        # Agregar el último párrafo si quedaron oraciones
        if current_paragraph:
            transcription_html += f"<p>{current_paragraph}</p>"

        # Actualizar los estilos CSS para incluir el formato de párrafos
        html_content = f"""
        <style>
        .container {{
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }}
        .audio-container {{
            position: sticky;
            top: 0;
            background-color: white;
            padding: 10px 0;
            z-index: 1000;
            border-bottom: 1px solid #ddd;
        }}
        .content-container {{
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        #transcription {{
            font-size: 1.2em;
            line-height: 1.5;
            flex: 1;
            margin-bottom: 20px;
        }}
        #transcription p {{
            margin-bottom: 1.5em;
            text-align: justify;
        }}
        #transcription span {{
            cursor: pointer;
        }}
        #transcription .highlight {{
            background-color: #FFFF00;
        }}
        .button-container {{
            padding: 20px;
            background-color: white;
            border-top: 1px solid #ddd;
            bottom: 0;
            z-index: 1000;
        }}
        .download-btn {{
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            display: block;
            width: fit-content;
            margin: 0 auto;
        }}
        .download-btn:hover {{
            background-color: #45a049;
        }}
        audio {{
            width: 100%;
            max-width: 600px;
            display: block;
            margin: 0 auto;
        }}
        </style>
        <div class="container">
            <div class="audio-container">
                <audio id="audio" controls>
                    <source src="data:audio/mp3;base64,{base64_audio}" type="audio/mp3">
                    Tu navegador no soporta el elemento de audio.
                </audio>
            </div>
            <div class="content-container">
                <div id="transcription">
                    {transcription_html}
                </div>
                <div class="button-container">
                    <button class="download-btn" onclick="downloadTranscription()">Descargar Transcripción Corregida</button>
                </div>
            </div>
        </div>
        <script>
        const audio = document.getElementById('audio');
        const transcription = document.getElementById('transcription');
        const spans = transcription.getElementsByTagName('span');

        audio.ontimeupdate = function() {{
          var currentTime = audio.currentTime;
          for (var i = 0; i < spans.length; i++) {{
            var span = spans[i];
            var start = parseFloat(span.dataset.start);
            var end = parseFloat(span.dataset.end);
            if (currentTime >= start && currentTime <= end) {{
              span.classList.add('highlight');
              // Auto-scroll to the highlighted span
              span.scrollIntoView({{ behavior: 'smooth', block: 'center', inline: 'nearest' }});
            }} else {{
              span.classList.remove('highlight');
            }}
          }}
        }};

        for (var i = 0; i < spans.length; i++) {{
          spans[i].addEventListener('click', function(e) {{
            var start = parseFloat(this.dataset.start);
            audio.currentTime = start;
            audio.play();
          }});
        }}

        function formatTime(seconds) {{
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const remainingSeconds = Math.floor(seconds % 60);
            return `${{String(hours).padStart(2, '0')}}:${{String(minutes).padStart(2, '0')}}:${{String(remainingSeconds).padStart(2, '0')}}`;
        }}

        function getEditedTranscription() {{
          let editedData = [];
          for (var i = 0; i < spans.length; i++) {{
            let span = spans[i];
            let text = span.innerText.trim();
            let start = parseFloat(span.dataset.start);
            let end = parseFloat(span.dataset.end);
            editedData.push({{
              start_time: formatTime(start),
              end_time: formatTime(end),
              text: text
            }});
          }}
          return editedData;
        }}

        function downloadTranscription() {{
            const editedData = getEditedTranscription();
            let csvContent = "start_time,end_time,text\\n";
            
            editedData.forEach(row => {{
                const text = row.text.replace(/"/g, '""'); // Escape quotes in text
                csvContent += `"${{row.start_time}}","${{row.end_time}}","${{text}}"\\n`;
            }});

            const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
            const link = document.createElement("a");
            const url = URL.createObjectURL(blob);
            link.setAttribute("href", url);
            link.setAttribute("download", "transcripcion_corregida.csv");
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }}

        window.getEditedTranscription = getEditedTranscription;
        </script>
        """

        # Mostrar el componente HTML
        st.components.v1.html(html_content, height=400, scrolling=False)

        # Limpiar archivos temporales
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if os.path.exists("temp_dir"):
            shutil.rmtree("temp_dir")   

if __name__ == "__main__":
    main()
