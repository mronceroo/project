import whisper
import pyaudio
import numpy as np

# Cargar el modelo (ajusta según tus necesidades: "tiny", "base", "small", etc.)
model = whisper.load_model("small")

# Configuración de la grabación
CHUNK = 1024
FORMAT = pyaudio.paInt16  # Formato de audio (16-bit PCM)
CHANNELS = 1              # Mono
RATE = 16000              # Tasa de muestreo (16kHz, óptimo para Whisper)
RECORD_SECONDS = 10        # Duración de la grabación

# Inicializar PyAudio
p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

# Grabación
print("Grabando...")
frames = []
for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Grabación terminada.")
stream.stop_stream()
stream.close()
p.terminate()

# Convertir frames a formato compatible con Whisper
audio_np = np.frombuffer(b''.join(frames), dtype=np.int16)
audio_np = audio_np.astype(np.float32) / 32768.0  # Normalizar a [-1.0, 1.0]

# Transcribir el audio directamente
result = model.transcribe(audio_np, language="es")
print("\nTexto transcrito:", result["text"])

# Opcional: Mostrar segmentos con timestamps
print("\nDetalles:")
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")