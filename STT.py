import whisper
import pyaudio
import numpy as np

class STT:
    def __init__(self, model_size="medium", language="en", record_seconds=12):
        self.model = whisper.load_model(model_size)
        self.language = language
        self.record_seconds = record_seconds
        
        # Audio configuration
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
    
    def listen(self):
        """Record audio and transcribe it using whisper"""
        # Open audio stream
        stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        print(f"Recording during {self.record_seconds} seconds...")
        frames = []
        
        # Record audio
        for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk)
            frames.append(data)
        
        print("Recording completed")
        
        # Clean up
        stream.stop_stream()
        stream.close()
        
        # Process audio data
        audio_np = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_np = audio_np.astype(np.float32) / 32768.0  # Normalize to [-1.0, 1.0]
        
        # Transcribe
        result = self.model.transcribe(audio_np, language=self.language)
        
        return result["text"]
    
    def __del__(self):
        """Clean up PyAudio when the object is destroyed"""
        if hasattr(self, 'p'):
            self.p.terminate()