import os
import base64
import pyttsx3
import speech_recognition as sr
import hashlib
from django.conf import settings
from pydub import AudioSegment  # Ensure pydub is installed
from gtts import gTTS

def save_audio(base64_string, folder="uploads"):
    """Decode base64 string, save as MP3, then convert to PCM WAV"""
    try:
        audio_data = base64.b64decode(base64_string)

        # File paths
        mp3_file_name = f"{folder}/audio_{hash(audio_data)}.mp3"
        wav_file_name = mp3_file_name.replace(".mp3", ".wav")

        full_mp3_path = os.path.join(settings.MEDIA_ROOT, mp3_file_name)
        full_wav_path = os.path.join(settings.MEDIA_ROOT, wav_file_name)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(full_mp3_path), exist_ok=True)

        # Save the MP3 file
        with open(full_mp3_path, "wb") as f:
            f.write(audio_data)

        # Convert MP3 to WAV (PCM format)
        audio = AudioSegment.from_file(full_mp3_path, format="mp3")
        audio = audio.set_channels(1).set_frame_rate(16000)  # Mono, 16kHz
        audio.export(full_wav_path, format="wav")

        return wav_file_name  # Return the relative path to WAV file
    except Exception as e:
        print("Error saving/converting audio:", str(e))
        return None


def speech_to_text(audio_file_path):
    """Convert speech to text using speech_recognition"""
    recognizer = sr.Recognizer()
    full_audio_path = os.path.join(settings.MEDIA_ROOT, audio_file_path)

    try:
        with sr.AudioFile(full_audio_path) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"Speech Recognition API error: {e}")
        return None
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None


def text_to_speech(text, lang="en"):
    """Convert text to speech using Google TTS & return base64-encoded MP3"""
    try:
        # Generate a unique file name
        file_name = f"responses/response_{hashlib.md5(text.encode()).hexdigest()}.mp3"
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Convert text to speech (gTTS)
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(file_path)

        # Convert saved audio file to base64
        with open(file_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")

    except Exception as e:
        print("Error generating speech:", str(e))
        return None