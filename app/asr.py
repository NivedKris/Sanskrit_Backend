import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class ASRManager:
    """
    Manager for Automatic Speech Recognition (ASR) services.
    Uses the NIVED47/Conformer_sanskrit Whisper model for Sanskrit transcription.
    """
    
    def __init__(self):
        # The actual model we'll always use
        self.actual_model_name = "NIVED47/Conformer_sanskrit"
        # For frontend display only
        self.frontend_model_name = "conformer"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the Whisper model and processor."""
        try:
            print(f"Loading ASR model {self.actual_model_name} on {self.device}...")
            self.processor = WhisperProcessor.from_pretrained(self.actual_model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.actual_model_name).to(self.device)
            print(f"ASR model loaded successfully.")
        except Exception as e:
            print(f"Error loading ASR model: {str(e)}")
            # Fallback to default model if specific model fails
            print("Falling back to default Whisper model...")
            self.actual_model_name = "openai/whisper-small"
            self.processor = WhisperProcessor.from_pretrained(self.actual_model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.actual_model_name).to(self.device)
    
    def update_settings(self, model=None):
        """
        Update ASR settings for frontend display only.
        
        Args:
            model: The model name from frontend (for display only)
        """
        if model:
            # Just update the frontend model name for UI consistency
            self.frontend_model_name = model
            print(f"Frontend model set to {model}, actual model remains {self.actual_model_name}")
        
        # Never reload the model - we always use the same one
        return {"model": self.frontend_model_name, "actual_model": self.actual_model_name}
    
    def transcribe(self, audio_path):
        """
        Transcribe audio using the loaded Whisper model.
        
        Args:
            audio_path: Path to the audio file (WAV format, 16kHz)
            
        Returns:
            str: Transcribed text
        """
        try:
            # Load audio
            import librosa
            audio, _ = librosa.load(audio_path, sr=16000)
            
            # Process with Whisper
            input_features = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate transcription
            predicted_ids = self.model.generate(input_features)
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            print(f"Error in transcription: {str(e)}")
            raise Exception(f"Transcription failed: {str(e)}") 