import time
import torch
import whisper

# Load the Whisper model
device_type = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device_type}")
device = torch.device(device_type)
model = whisper.load_model("medium").to(device)  # You can choose 'base', 'small', 'medium', 'large', etc.

# medium
def transcribe_audio(wav_file):
    start_time = time.time()
    result = model.transcribe(wav_file)
    end_time = time.time()
    return result['text'], end_time - start_time


wav_file = "/home/felipeagger/Dados/Dev/Python/benchgpu/audio-curto-1min.wav"
wav_file_podcast = "/home/felipeagger/Dados/Dev/Python/benchgpu/podcast13min.wav"
transcription, duration = transcribe_audio(wav_file_podcast)
print(f"duration: {duration:.2f} seconds\n")
print(transcription)
