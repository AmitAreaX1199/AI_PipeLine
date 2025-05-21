import os
from google.cloud import texttospeech,speech

# Set the path to your service account key JSON file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/tricky-shivam/Desktop/AreaX_Folder/areax_ai_project/gen-lang-client-0238752775-21032a3a2bc7.json"
##Used in Function
def text_to_speech(text, output_filename="output.mp3"):
    """Convert text to speech and save as an audio file."""
    try:
        client = texttospeech.TextToSpeechClient()

        input_text = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Studio-O",
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1
        )

        response = client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )

        with open(output_filename, "wb") as out:
            out.write(response.audio_content)
            print(f'Audio content written to file "{output_filename}"')

        return output_filename

    except Exception as e:
        print(f"❌ Error generating speech: {str(e)}")
        return None

# Example usage:
# text_to_speech("Hello, I am good.", "speech_output.mp3")

###Speech to text function API


def speech_to_text(audio_path: str):
    """Converts speech from an audio file to text using Google Speech-to-Text API"""
    try:
        # Initialize the Speech-to-Text client

        client = speech.SpeechClient()

        # Read the audio file as binary
        with open(audio_path, "rb") as audio_file:
            audio_content = audio_file.read()

        # Configure the request
        audio = speech.RecognitionAudio(content=audio_content)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MP3,  # Use MP3 encoding
            sample_rate_hertz=48000,
            language_code="en-US",
        )

        # Send the request to Google API
        response = client.recognize(config=config, audio=audio)
        print(response,"-----------response")
        # Process the response
        if not response.results:
            print("No speech recognized.")
            return None

        # Print and return the transcript
        transcript = response.results[0].alternatives[0].transcript
        print(f"Transcript+++++++++++++: {transcript}")
        return transcript

    except Exception as e:
        print(f"Error: {e}")
        return None
