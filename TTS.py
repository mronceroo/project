from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
from datasets import load_dataset
import sounddevice as sd
import os
import re

class TTS:
    def __init__(self, cache_dir="tts_cache"):
        # Cache configuration
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize models and configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize_models()
        
        # Dictionary for converting digits to words
        self.digits_to_words = {
            '0': ' zero',
            '1': ' one',
            '2': ' two',
            '3': ' three',
            '4': ' four',
            '5': ' five',
            '6': ' six',
            '7': ' seven',
            '8': ' eight',
            '9': ' nine'
        }
        
    def _initialize_models(self):
        # Load models
        print("Loading TTS models...")
        self.processor = SpeechT5Processor.from_pretrained(
            "microsoft/speecht5_tts",
            cache_dir=os.path.join(self.cache_dir, "models")
        )
        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts",
            cache_dir=os.path.join(self.cache_dir, "models")
        ).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan",
            cache_dir=os.path.join(self.cache_dir, "models")
        ).to(self.device)
        
        # Load embeddings
        try:
            # Try to load from local cache
            self.dataset = load_dataset(os.path.join(self.cache_dir, "embeddings"))
        except:
            # Download and cache if it doesn't exist
            self.dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            self.dataset.save_to_disk(os.path.join(self.cache_dir, "embeddings"))
        
        # Precompute embeddings for reuse
        self.speaker_embeddings = torch.tensor(
            self.dataset[7306]["xvector"]
        ).unsqueeze(0).to(self.device)
        
        print("TTS models loaded successfully")
    
    def speak(self, text):
        # Convert all digits to words for better speech
        processed_text = self._convert_numbers_to_words(text)
        
        # Process text
        inputs = self.processor(text=processed_text, return_tensors="pt").to(self.device)
        
        # Generate audio
        with torch.no_grad():
            audio = self.model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings=self.speaker_embeddings,
                vocoder=self.vocoder
            )
        
        # Play audio
        sd.play(audio.cpu().numpy(), samplerate=16000)
        sd.wait()
        
        return True
    
    def _convert_numbers_to_words(self, text):
        # Function to replace matched digits with words
        def replace_digit(match):
            digit = match.group(0)
            if digit in self.digits_to_words:
                return self.digits_to_words[digit]
            return digit

        # Pattern to find all digits
        pattern = r'\d'

        # Replace all digits with their word equivalents
        processed_text = re.sub(pattern, replace_digit, text)
        return processed_text
    
    def _format_chess_square(self, square):
        if not square or len(square) != 2:
            return "invalid position"

        column, row = square[0].lower(), square[1]

        # digits to words
        number_words = {
            '1': 'one',
            '2': 'two',
            '3': 'three',
            '4': 'four',
            '5': 'five',
            '6': 'six',
            '7': 'seven',
            '8': 'eight'
        }

        # Format the coordinate to sound natural
        if row in number_words:
            return f"{column} {number_words[row]}"
        else:
            return f"{column} {row}" 
    
    
    def speak_move(self, origin, dest):
        origin_spoken = self._format_chess_square(origin)
        dest_spoken = self._format_chess_square(dest)
        
        move_text = f"Move from {origin_spoken} to {dest_spoken}"
        return self.speak(move_text)
