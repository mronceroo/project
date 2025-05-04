from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
from datasets import load_dataset
import sounddevice as sd
import os

# --- Configuración de caché ---
CACHE_DIR = "tts_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# 1. Cargar modelos con caché
def load_models():
    processor = SpeechT5Processor.from_pretrained(
        "microsoft/speecht5_tts",
        cache_dir=os.path.join(CACHE_DIR, "models")
    )
    model = SpeechT5ForTextToSpeech.from_pretrained(
        "microsoft/speecht5_tts",
        cache_dir=os.path.join(CACHE_DIR, "models")
    )
    vocoder = SpeechT5HifiGan.from_pretrained(
        "microsoft/speecht5_hifigan",
        cache_dir=os.path.join(CACHE_DIR, "models")
    )
    return processor, model, vocoder

# 2. Cargar embeddings con caché
def load_embeddings():
    try:
        # Intentar cargar desde caché local
        dataset = load_dataset(os.path.join(CACHE_DIR, "embeddings"))
    except:
        # Descargar y guardar en caché si no existe
        dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        dataset.save_to_disk(os.path.join(CACHE_DIR, "embeddings"))
    return dataset

# --- Ejecución principal ---
processor, model, vocoder = load_models()
dataset = load_embeddings()

# 3. Configurar dispositivo (GPU si está disponible)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
vocoder = vocoder.to(device)

# 4. Procesar texto
text = "b two b three"
inputs = processor(text=text, return_tensors="pt").to(device)

# 5. Obtener embeddings (índice 7306 = voz femenina)
speaker_embeddings = torch.tensor(dataset[7306]["xvector"]).unsqueeze(0).to(device)

# 6. Generar y reproducir audio
with torch.no_grad():
    audio = model.generate_speech(
        inputs["input_ids"],
        speaker_embeddings=speaker_embeddings,
        vocoder=vocoder
    )

# 7. Reproducir directamente (sin guardar)
sd.play(audio.cpu().numpy(), samplerate=16000)
sd.wait()
print("Audio generado y reproducido con éxito")