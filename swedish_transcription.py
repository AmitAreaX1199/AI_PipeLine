# import openai

from openai import OpenAI
# Set up your OpenAI API key

cllient = OpenAI(api_key="OPENAI_API_KEY")
from pydub import AudioSegment
import math

audio_file_path = "/home/Amit/Downloads/SUPERTRENDS - Personlig AI .mp3"  # Change to your file's path
audio = AudioSegment.from_file(audio_file_path)

# Define chunk length in milliseconds (5 minutes per chunk)
chunk_length_ms = 5 * 60 * 1000  # 5 minutes

# Calculate the number of chunks
num_chunks = math.ceil(len(audio) / chunk_length_ms)

# Transcription text
transcription_text = ""

# Process each chunk
for i in range(num_chunks):
    # Extract the chunk
    start_time = i * chunk_length_ms
    end_time = min((i + 1) * chunk_length_ms, len(audio))
    audio_chunk = audio[start_time:end_time]

    # Save the chunk as a temporary file
    chunk_filename = f"temp_chunk_{i}.mp3"
    audio_chunk.export(chunk_filename, format="mp3")

    # Transcribe the chunk
    with open(chunk_filename, "rb") as audio_file:
        response = cllient.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en", # 'en' is the language code for Swedish
            response_format="text"
        )
        transcription_text += response + " "  # Append the transcription of each chunk

# Print the full transcription
print("Full Transcription+++++++++++++++++++++++++++++++++:", transcription_text,"=================result")




