import io

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authentication import TokenAuthentication,SessionAuthentication
from django.contrib.auth import authenticate,login as django_login, logout as django_logout
from django.contrib.auth.models import User
from .serializers import (UserSerializer,GooglePhotosCredentialsSerializer,ImageUploadSerializer,Edit_Caption_Serializer,UserInputSerializer,AIResponseSerializer,TextInputSerializer,FeedbackSerializer,GeneratedImageSerializer,OpenaAIUsageDBSerializer,VideoGenerated_Serializer,SmartResponseSerializer,ChatSessionSerializer,EnhancedSocialContentSerializer)
from django.db.models import Q
from rest_framework.permissions import IsAuthenticated,AllowAny
from rest_framework.authtoken.models import Token
from .utils import encode_image_to_base64,describe_image
from .prompt_templates import get_prompt_template
from .google_photos_service import GooglePhotosService
from rest_framework import status
from django.conf import settings
from django.http import JsonResponse
from django.core.files.storage import default_storage

from rest_framework.parsers import MultiPartParser, FormParser
import logging
from pydub import AudioSegment
from .models import AIContentDb,AIResponse,UserInput,GeneratedImage,OpenaAI_UsageDB,ImageGenerationSD_DB,ImageCaptionGeminiDB,SmartResponse,ChatSession,EnhancedSocialContent,SchedulingAssistantLog,WellnessBotLog
from django.utils import timezone
from django.db.models import Sum
from dotenv import load_dotenv
from transformers import pipeline
from django.http import HttpResponse
import speech_recognition as sr
from PIL import Image
import os
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.googlecalendar import GoogleCalendarTools
from tzlocal import get_localzone_name
import random
from datetime import datetime
load_dotenv()
logger = logging.getLogger(__name__)

# Create your views here.
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")



def baseurl(request):
  """
  Return a BASE_URL templates context for the current request.
  """
  if request.is_secure():
    scheme = "https://"
  else:
    scheme = "http://"
  return scheme + request.get_host()


#Get User email
def get_user(email):
  try:
    user = User.objects.filter(
      Q(email=email.lower())
      | Q(username=email)
    ).first()
    if user:
      return [True, user]
    else:
      return [False, None]
  except:
    return [False, None]



##UserRegistrationAPI
class UserRegistrationView(APIView):
    def post(self, request,format=None):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
          email = serializer.validated_data.get('email')
          user = User.objects.filter(email=email).first()
          if user:
            token, created = Token.objects.get_or_create(user=user)
            return Response({"token": token.key},
                            status=status.HTTP_200_OK)
          else:
            user = serializer.save()
            token = Token.objects.create(user=user)
            return Response({"token": token.key,
                             "data": serializer.data},
                            status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



##Userlogin API
class LoginView(APIView):
  def post(self, request):
    try:
      email = request.data.get("username", None)
      password = request.data.get("password", None)

      if email and password:
        user = get_user(email)

        if user[0]:
          if not not user[1].password:
            user_data = authenticate(username=user[1], password=password)
            if user_data:
              django_login(request, user[1])
              token, created = Token.objects.get_or_create(user=user[1])
              return Response({
                "status": status.HTTP_200_OK,
                "message": "Successfully logged in",
                "user_id": user_data.id,
                "token": token.key,
                "base_url": baseurl(request),
              })
            else:
              content = {
                "status": status.HTTP_204_NO_CONTENT,
                "message": "Unable to Login with given credentials"
              }
              return Response(content)
          else:
            content = {
              "status": status.HTTP_204_NO_CONTENT,
              "message": "Please reset your password"
            }
            return Response(content)
        else:
          content = {
            "status": status.HTTP_204_NO_CONTENT,
            "message": "Unable to Login with given credentials"
          }
          return Response(content)
      return Response({
        "data": [],
        "status": status.HTTP_401_UNAUTHORIZED,
        "message": "Unable to login with given credentials"
      })
    except Exception as e:
      context = {
        "status": status.HTTP_400_BAD_REQUEST,
        "message": str(e)
      }
      return Response(context)



##Logout API
class LogoutView(APIView):
  authentication_classes = (TokenAuthentication,)

  def post(self, request, format=None):
    user_id = request.user.id
    if user_id:
      logged_in_user_id = User.objects.filter(id=user_id).first()
      if logged_in_user_id:
        django_logout(request)
      content = {"status": 200, "message": "LogOut Successfully"}
    else:
      content = {"status": 400, "message": "Invalid token"}
    return Response(content, status=status.HTTP_200_OK)



# API Gather Information by AI Agent
class Generate_ContentAPIView(APIView):

    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        email=request.user.email
        data = request.data
        new_data = []

        try:
            base_path = os.path.join(settings.BASE_DIR, 'media/upload_image')
            if not os.path.exists(base_path):
                os.makedirs(base_path)

            for index, item in enumerate(data):
                image_urls = item.get('image_url', [])
                global image_url
                for i, image_url in enumerate(image_urls, start=1):
                    local_image_path = os.path.join(base_path, f"image_{index}_{i}.jpg")
                    response = requests.get(image_url)

                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content))
                        image.save(local_image_path)
                    else:
                        logger.error(f"Failed to download image from: {image_url}")
                        continue
                    # Get the prompt templates
                    prompt_template = get_prompt_template()
                    final_prompt = prompt_template.format()
                    # Encode image to data URI
                    image_data_uri = encode_image_to_base64(local_image_path)
                    response = describe_image(image_data_uri, final_prompt)
                    print(response,"-----------response###########")
                    print(type(response))

                    if response:
                        new_data.append(response)
                        # Save the response to the database
                        AIContentDb.objects.create(
                            user=user,
                            email=email,
                            image_url=image_url,
                            caption=response
                        )
                    else:
                        logger.error(f"Error processing response for image: {local_image_path}")

            return JsonResponse({"Caption": new_data,"image_url":image_url, 'status': status.HTTP_200_OK})

        except Exception as e:
            logger.exception("An error occurred in ImageDownloadAPIView")
            return JsonResponse({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


## Retriving Google image list
class GooglePhotosAPI(APIView):

    def get(self, request, *args, **kwargs):
        serializer = GooglePhotosCredentialsSerializer(data=request.query_params)
        if serializer.is_valid():
            email = serializer.validated_data.get('email')

            credentials_info = GooglePhotosService.get_google_api_credentials(email)
            if credentials_info:
                service = GooglePhotosService.get_google_photos_service(credentials_info)
                photos = GooglePhotosService.get_photos(service)
                return Response(photos, status=status.HTTP_200_OK)
            else:
                return Response({"error": "Failed to fetch Google API credentials"}, status=status.HTTP_400_BAD_REQUEST)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

## Matching Search image API
class SearchCaptionImageAPI(APIView):

    def post(self, request, *args, **kwargs):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data.get('email')
            image = serializer.validated_data.get('image')

            uploaded_image = image.name

            credentials_info = GooglePhotosService.get_google_api_credentials(email)
            if credentials_info:
                service = GooglePhotosService.get_google_photos_service(credentials_info)

                google_photos = GooglePhotosService.fetch_all_photos(service)
                if google_photos:
                    matched_photo = GooglePhotosService.match_image(uploaded_image, google_photos)
                    if matched_photo:
                        return Response(
                            {"matched_image": matched_photo, "message": "Successfully fetched matching image"},
                            status=status.HTTP_200_OK)
                    else:
                        return Response({"error": "No matching photo found"}, status=status.HTTP_404_NOT_FOUND)
                else:
                    return Response({"error": "No photos found in Google Photos"}, status=status.HTTP_404_NOT_FOUND)
            else:
                return Response({"error": "Failed to fetch Google API credentials"}, status=status.HTTP_400_BAD_REQUEST)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



###ALl generated content
class Caption_listAPI(APIView):
    permission_classes = [IsAuthenticated]
    def get(self,request):
        try:
            user_content = AIContentDb.objects.filter(user=request.user)
            serializer = Edit_Caption_Serializer(user_content, many=True)
            return Response(serializer.data, status=200)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


###EDIT CAPTIONAPI
class EditCaptionAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        caption_id = request.query_params.get("id")  # Use query_params for GET request
        if not caption_id:
            return Response({"error": "Caption ID not provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            caption = AIContentDb.objects.get(id=caption_id)
            serializer = Edit_Caption_Serializer(caption)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except AIContentDb.DoesNotExist:
            return Response({'error': 'Caption not found'}, status=status.HTTP_404_NOT_FOUND)

    def post(self, request, format=None):
        data = request.data.copy()
        caption_id = data.pop('id', None)  # Extract caption ID from data

        if not caption_id:
            return Response({"error": "Caption ID not provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            caption = AIContentDb.objects.get(id=caption_id)
            serializer = Edit_Caption_Serializer(caption, data=data, partial=True)  # Allow partial updates
            if serializer.is_valid():
                serializer.save()  # Save updated caption
                return Response(serializer.data, status=status.HTTP_200_OK)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except AIContentDb.DoesNotExist:
            return Response({"error": f"Caption with ID {caption_id} not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


####AI Sentiment personal agent API 22aug Running smothly
# Load models Sentiment NLP Model
sentiment_model = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
import google.generativeai as genai
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
# Configuration for the model generation
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

####TextCHatAPI with Gemini model on 28feb25
class TextInputHandler(APIView):
    client = genai.GenerativeModel(model_name="gemini-2.0-flash",generation_config=generation_config)

    # List of pre-prompts and suggestion prompts
    pre_prompts = [
        "How can I assist you today?",
        "Is there anything I can help you with?",
        "What would you like to talk about?",
        "How are you feeling today?",
        "Do you need advice or just someone to talk to?",
    ]

    suggestion_prompts = {
        "general": [
            "Tell me more about your day.",
            "Do you have any upcoming plans?",
            "How are you feeling?",
            "Would you like some tips?"
        ],
        "positive": [
            "What's something good that happened today?",
            "Tell me about something exciting.",
            "What are you grateful for?"
        ],
        "negative": [
            "Is there anything bothering you?",
            "Would you like to talk about something challenging?",
            "How can I support you?"
        ]
    }

    def get_suggestions(self, tone):
        """ Generate auto-suggestions based on tone. """
        if tone == 'positive':
            return random.sample(self.suggestion_prompts['positive'], 2)
        elif tone == 'negative':
            return random.sample(self.suggestion_prompts['negative'], 2)
        else:
            return random.sample(self.suggestion_prompts['general'], 2)

    def post(self, request):
        text = request.data.get('text')
        user_reference_number = request.data.get('user_reference_number')
        user_email = request.data.get('user_email')

        if not text:
            return Response({"error": "No text input provided"}, status=400)

        # Pre-prompt selection logic
        current_hour = datetime.now().hour
        greeting_message = (
            "Good morning!" if current_hour < 12
            else "Good afternoon!" if current_hour < 18
            else "Good evening!"
        )

        # Choose a random pre-prompt from the list
        chosen_pre_prompt = random.choice(self.pre_prompts)

        # Automatically detect tone (mocked here, replace with real sentiment model)
        tone = "neutral"  # Default to neutral if sentiment detection is missing

        # Get auto-suggestions based on the detected tone
        suggestions = self.get_suggestions(tone)

        # Save user input with detected tone
        user_input = UserInput.objects.create(text=text, tone=tone)

        # Prepare messages for Gemini model
        messages = [
            f"The user's tone is {tone}. Respond to the user's query in a brief, concise, and meaningful way while ensuring clarity. Use complete sentences.",
            chosen_pre_prompt,  # Add chosen pre-prompt
            f"Consider these relevant suggestions: {', '.join(suggestions)}.",
            f"User's query: {text}"
        ]

        # Generate response using Gemini model
        try:
            response = self.client.generate_content(messages)
            response_text = response.text.strip()
        except Exception as e:
            return Response({"error": f"AI response generation failed: {str(e)}"}, status=500)

        # Save AI response
        ai_response = AIResponse.objects.create(
            user_input=user_input,
            user_reference_number=user_reference_number,
            user_email=user_email,
            response_text=response_text,
        )

        # Prepare response data
        response_data = {
            'ai_response': AIResponseSerializer(ai_response).data,
            'feedback_prompt': "Please provide your feedback on this response. Rate from 1 to 5 stars."
        }

        return Response(response_data,status.HTTP_200_OK)

###Image Generation API via SD model
###LLAMA3.2
class LlamaStatusView(APIView):
    def get(self, request, *args, **kwargs):
        return Response({"status": status.HTTP_200_OK, "message": "Llama3.2 API is working"}, status=status.HTTP_200_OK)

####new with imahe url
API_KEY = os.getenv("HUGGING_FACE_API_KEY")
class GenerateImageView(APIView):
    def post(self, request, *args, **kwargs):
        # Extract user prompt
        prompt = request.data.get('prompt', '').strip()
        user_reference_number = request.data.get('user_reference_number', '')
        user_email = request.data.get('user_email', '')
        negative_prompt = request.data.get('negative_prompt', '')
        if not prompt:
            return Response({"status": "error", "message": "Prompt is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Hugging Face API details
        API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"

        if not API_KEY:
            return Response({"status": "error", "message": "API key is not configured"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        headers = {"Authorization": f"Bearer {API_KEY}"}

        # Query the Hugging Face model
        try:
            response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
            if response.status_code == 200:
                # Ensure response content is valid image data
                if not response.content:
                    return Response({"status": "error", "message": "Empty response from model"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                # Generate a unique file name using UUID
                unique_id = uuid.uuid4()
                output_format = response.headers.get("Content-Type", "image/png").split('/')[-1]
                file_name = f"generated_image_{unique_id}.{output_format}"
                file_path = os.path.join(settings.MEDIA_ROOT, file_name)

                # Save the image to the media directory
                with open(file_path, "wb") as img_file:
                    img_file.write(response.content)

                # Construct the accessible image URL
                image_url = f"{settings.MEDIA_URL}{file_name}"

                print(baseurl(request)+image_url,"----------")
                # Save the generated image to the database
                generated_image = GeneratedImage.objects.create(
                                    prompt=prompt,
                                    user_reference_number=user_reference_number,
                                    user_email=user_email,
                                    negative_prompt=negative_prompt,
                                    image=baseurl(request)+image_url,  # Save binary content directly
                )
                # Return response with the image URL
                return Response({
                    "status": "success",
                    "image_url": baseurl(request)+image_url
                }, status=status.HTTP_200_OK)

            elif response.status_code == 401:
                return Response({"status": "error", "message": "Unauthorized: Check API key"}, status=status.HTTP_401_UNAUTHORIZED)
            elif response.status_code == 429:
                return Response({"status": "error", "message": "Rate limit exceeded"}, status=status.HTTP_429_TOO_MANY_REQUESTS)
            else:
                return Response({
                    "status": "error",
                    "message": f"Unexpected response from model: {response.status_code}",
                    "details": response.text
                }, status=response.status_code)

        except requests.exceptions.RequestException as e:
            logging.error(f"Error while calling Hugging Face API: {e}")
            return Response({"status": "error", "message": "Internal server error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




####Generated image ShowAPI
class RetrieveImageView(APIView):
    def get(self, request, pk):
        image_record = get_object_or_404(GeneratedImage, pk=pk)
        return HttpResponse(image_record.image, content_type='image/jpeg')

#####Text To Video API

class GenerateVideoAPIView(APIView):
    def post(self, request):
        try:
            prompt = request.data.get('prompt')

            if not prompt:
                return Response({"error": "Prompt is required"}, status=status.HTTP_400_BAD_REQUEST)

            # Load the model
            pipe = CogVideoXPipeline.from_pretrained(
                "THUDM/CogVideoX-5b",
                torch_dtype=torch.bfloat16
            )

            # Optional optimizations
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_tiling()

            # Generate video
            video = pipe(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=50,
                num_frames=49,
                guidance_scale=6,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).frames[0]

            # Save video
            output_file_path = "output.mp4"
            export_to_video(video, output_file_path, fps=8)

            # Return the video file URL (optional: serve the file using Django's static/media setup)
            return Response({"message": "Video generated successfully", "video_url": f"/media/{output_file_path}"})

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


## User Feedback API

class FeedbackAPIView(APIView):
    def post(self, request):
        serializer = FeedbackSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({"message": "Feedback submitted successfully!"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


from openai import OpenAI
client_open=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def text_to_speech(text):
    response = client_open.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
    )
    print(response,"-------++++++response++++++--- text to spechh--------")
    audio_data = io.BytesIO(response.content)
    return audio_data

class AudioTranscriptionView(APIView):
    # parser_classes = (MultiPartParser)  # To handle file uploads

    def post(self, request, *args, **kwargs):
        user_reference_number = request.data.get("user_reference_number", "")
        user_email = request.data.get("user_email", "")
        base64_audio = request.data.get("audio")
        audio_bytes = base64.b64decode(base64_audio)
        audio_stream = io.BytesIO(audio_bytes)
        audio_stream.name = "file.mp3"

        try:

            transcript_text = client_open.audio.transcriptions.create(
                model="whisper-1",
                file=audio_stream,
                language="en",
                response_format="text"
            )

            print(transcript_text)
            if not transcript_text:
                return Response({"error": "Could not transcribe audio"}, status=400)

            # Generate the AI response based on the transcription
            response = client_open.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a highly skilled AI persona Agent, reply based on user sentiment in complete sentences."},
                    {"role": "user", "content": transcript_text}
                ],
                max_tokens=256,
                temperature=0.5
            )
            print(response,"--------------AUdio _____response ")

            # Access token usage from the response

            output_tokens = response.usage.completion_tokens

            print(output_tokens, "--------output_tokens----")
            input_tokens = response.usage.prompt_tokens
            print(input_tokens, "-------input_prompt")
            total_tokens = response.usage.total_tokens
            print(total_tokens, "------------ALL use token")

            # Cost calculation
            cost_per_token = 0.0015 / 1000  # Example rate
            input_cost = input_tokens * cost_per_token
            print(input_cost, "---------------input cost")
            output_cost = output_tokens * cost_per_token
            print(output_cost, "-----------output cost")
            total_cost = total_tokens * cost_per_token
            print(total_cost, "-------total cost used per chat-----")

            # # Save user input with detected tone

            if response.choices[0].message.content:
                ai_response = response.choices[0].message.content
                print(ai_response,"----------ai_response voice")

                audio_data = text_to_speech(ai_response)
                audio_data.seek(0)
                audio_base64 = base64.b64encode(audio_data.read()).decode("utf-8")

                # Save to OpenaAI_UsageDB
                OpenaAI_UsageDB.objects.create(
                    user_reference_number=user_reference_number,
                    user_email=user_email,
                    prompt=base64_audio,
                    response=ai_response,
                    audio_base64=audio_base64,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    total_tokens=total_tokens,
                    input_cost=input_cost,
                    output_cost=output_cost,
                    total_cost=total_cost
                )


                return Response({"status":status.HTTP_200_OK,"user_reference_number":user_reference_number,"user_email":user_email,"transcript": transcript_text, "ai_response": ai_response, "audio": audio_base64}, status=200)
            else:
                return Response({"error": "Failed to generate a response from GPT-4"}, status=500)
        except Exception as e:
            return Response({"error": f"An error occurred: {str(e)}"}, status=500)
        finally:
            pass

#### All total cost getAPI

##MonthlyeiseAPi
class OverallCostAPIView(APIView):
    def get(self, request, *args, **kwargs):
        try:
            # Get the current month and year
            now = timezone.now()
            current_year = now.year
            current_month = now.month

            # Filter data for the current month and year and calculate the total cost
            current_month_cost = (
                AIResponse.objects
                .filter(created_at__year=current_year, created_at__month=current_month)
                .aggregate(total_cost=Sum('total_cost'))
            )

            # Extract the total cost, handling None case
            total_cost = current_month_cost['total_cost'] or 0.0

            # Prepare the response data
            response_data = {
                "status": "success",
                "reference_number" : "",
                "email" : "",
                "month": now.strftime("%B %Y"),  # Format: "September 2024"
                "total_cost": total_cost
            }

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



## 13 sep Imagegeneration by SD model API
class ImageGenerationAPIView(APIView):
    def post(self, request):
        # Extract the user prompt, output format, and model from the request body
        prompt = request.data.get('prompt')
        output_format = request.data.get('output_format', 'jpeg')  # default to jpeg if not provided
        # model = request.data.get('model', 'sd3-medium')  # default to 'sd3-medium' if not provided

        if not prompt:
            return Response({"error": "Prompt is required."}, status=status.HTTP_400_BAD_REQUEST)

        # API request to generate an image
        api_url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
        headers = {
            "authorization": "Bearer ",
            "accept": "image/*"
        }
        data = {
            "prompt": prompt,
            "output_format": output_format,
            "model": "sd3-medium",
        }

        try:
            response = requests.post(api_url, headers=headers, files={"none": ''}, data=data)

            # Check if the response is successful
            if response.status_code == 200:
                # Save the image to a temporary file (if needed)
                image_path = f"./generated_image.{output_format}"
                with open(image_path, 'wb') as file:
                    file.write(response.content)

                # Cost calculation
                credits_per_image = 3.5  # Example value, adjust as needed
                cost_per_credit = 0.01  # Example value, adjust as needed
                used_credit_in_image_generation = credits_per_image * cost_per_credit
                total_cost_in_dollar = used_credit_in_image_generation

                # Save to the database
                image_generation = ImageGenerationSD_DB.objects.create(
                    prompt=prompt,
                    image_path=image_path,
                    total_cost_in_dollar=total_cost_in_dollar
                )

                return Response({
                    "message": "Image generated successfully.",
                    "cost_in_usd": f"${total_cost_in_dollar:.4f}",
                    "total_cost_in_dollar": total_cost_in_dollar,  # Return cost in numeric format
                    "image_path": image_path
                }, status=status.HTTP_200_OK)
            else:
                return Response({
                    "error": f"Image generation failed with status code {response.status_code}.",
                    "details": response.json()
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


###new30sep
class TotalImageCostAPIView(APIView):
    def get(self, request):
        try:
            # Get the current month and year
            now = timezone.now()
            current_year = now.year
            current_month = now.month

            # Filter data for the current month and year and calculate the total cost
            current_month_cost = (
                ImageGenerationSD_DB.objects
                .filter(created_at__year=current_year, created_at__month=current_month)
                .aggregate(total_cost=Sum('total_cost_in_dollar'))
            )

            # Extract the total cost, handling None case
            total_cost = current_month_cost['total_cost'] or 0.0

            # Prepare the response data
            response_data = {
                "status": "success",
                "reference_number":"",
                "email":"",
                "month": now.strftime("%B %Y"),  # Format: "September 2024"
                "total_cost_in_dollar": total_cost
            }

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
##
#



# Configure your Gemini API Key
import google.generativeai as genai
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))


# Configuration for the model generation
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
##New caption with gemini2 032025
from google import genai
import PIL.Image
genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
class GeminiCaptionAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response({"error": "No image file provided"}, status=status.HTTP_400_BAD_REQUEST)

        image_file = request.FILES['image']
        caption_type = request.data.get('caption_type', "predefined").strip().lower()  # Normalize input
        custom_prompt = request.data.get('prompt', "").strip()

        # Save uploaded image temporarily
        image_path = default_storage.save('uploaded_images/' + image_file.name, image_file)
        image_full_path = default_storage.path(image_path)
        print(image_full_path,"---------full pa image ")

        file_path = os.path.join(settings.MEDIA_ROOT, image_path)
        print(file_path,"---------file path ")

        try:
            # Load image using PIL
            image = PIL.Image.open(image_full_path)


            # Predefined prompt
            predefined_prompt = (
                """
                Your task is to analyze the given images.
                And generate attractive caption about the given images for social-media platform.
                And Caption should be in complete sentence with hashtags.
                Important Note: Do not give descriptions out of context , Do not give emojis And Do Not Use this words 'Here's a caption for social media:'
                """

            )

            # Determine the prompt to use
            if caption_type == "custom":
                if not custom_prompt:
                    return Response({"error": "Custom prompt is required when caption_type is 'custom'."}, status=status.HTTP_400_BAD_REQUEST)
                prompt_text = custom_prompt
            elif caption_type == caption_type:
                prompt_text = predefined_prompt + "and also make sure it is " + caption_type
            else:
                return Response({"error": "Invalid caption_type. Use 'custom' or 'predefined'."}, status=status.HTTP_400_BAD_REQUEST)

            # Generate caption using Gemini model
            print(genai_client,"-----------gemai cliet")
            response = genai_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt_text, image]
            )

            generated_caption = response.text.strip()
            # Remove unwanted phrase

            unwanted_phrase = "Here's a caption for social media:\n\n"
            final_caption = generated_caption.replace(unwanted_phrase, "")

            # Save the image and caption in the database
            image_caption = ImageCaptionGeminiDB.objects.create(
                image=image_path,
                caption=final_caption + "  " + "#projectW" + " " + "#project_w",
            )

            print(image_caption,"-----------TotalData")
            Base_url = "http://52.29.182.207:4004/media/"
            # Return the saved data in the API response
            return Response({
                "status": status.HTTP_200_OK,
                "id": image_caption.id,
                "image_url": Base_url+image_path,
                "caption": image_caption.caption,
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

##Gemini Overall cost API
class Gemini_OverallCostAPIView(APIView):
    """
    API to calculate the overall cost of all generated captions for the current month.
    """

    def get(self, request, *args, **kwargs):
        try:
            # Get the current month and year
            now = timezone.now()
            current_year = now.year
            current_month = now.month

            # Filter data for the current month and year
            current_month_cost = (
                ImageCaptionGeminiDB.objects
                .filter(created_at__year=current_year, created_at__month=current_month)
                .aggregate(total_cost=Sum('total_cost'))
            )

            # Extract the total cost, handling None case
            total_cost = current_month_cost['total_cost'] or 0.0

            # Prepare the response data
            response_data = {
                "status": "success",
                "month": now.strftime("%B %Y"),  # Format: "September 2024"
                "total_cost": f"${round(total_cost,4)}"
            }

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


####Specific date range04oct :
from django.utils.dateparse import parse_date
from django.utils.timezone import make_aware, get_current_timezone
class Gemini_daterangeCostAPI(APIView):
    """
    API to calculate the overall cost of generated captions within a given date range.
    """
    def get(self, request, *args, **kwargs):
        try:
            # Get query parameters for start_date and end_date
            start_date_str = request.query_params.get('start_date')
            end_date_str = request.query_params.get('end_date')

            # Get the current date if no dates are provided
            now = timezone.now()
            current_year = now.year
            current_month = now.month

            # Parse the date strings into actual dates
            if start_date_str:
                start_date = parse_date(start_date_str)
                # Convert to datetime and make timezone-aware
                start_date = datetime.combine(start_date, datetime.min.time())  # Convert date to datetime (00:00:00)
                start_date = make_aware(start_date, timezone=get_current_timezone())
            else:
                # Default to the first day of the current month if start_date is not provided
                start_date = make_aware(
                    datetime(current_year, current_month, 1),
                    timezone=get_current_timezone()
                )

            if end_date_str:
                end_date = parse_date(end_date_str)
                # Convert to datetime and make timezone-aware
                end_date = datetime.combine(end_date, datetime.max.time())  # Convert date to datetime (23:59:59)
                end_date = make_aware(end_date, timezone=get_current_timezone())
            else:
                # Default to the current date if end_date is not provided
                end_date = now

            # Validate date order
            if start_date > end_date:
                return Response(
                    {"error": "start_date cannot be greater than end_date."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Filter data for the given date range
            cost_data = (
                ImageCaptionGeminiDB.objects
                .filter(created_at__range=[start_date, end_date])
                .aggregate(total_cost=Sum('total_cost'))
            )

            # Extract the total cost, handle the case where no data is found
            total_cost = cost_data['total_cost'] or 0.0


            # Prepare the response data
            response_data = {
                "status": "success",
                "service_name": "Gemini_service",
                "total_cost": f"${total_cost:.4f}"
            }

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

##with months and years
class OpenAI_daterangeCostAPI(APIView):
    """
    API to calculate the overall cost of generated captions within a given date range.
    """
    def get(self, request, *args, **kwargs):
        try:
            # Get query parameters for start_date and end_date
            start_date_str = request.query_params.get('start_date')
            end_date_str = request.query_params.get('end_date')

            # Get the current date if no dates are provided
            now = timezone.now()
            current_year = now.year
            current_month = now.month

            # Parse the date strings into actual dates
            if start_date_str:
                start_date = parse_date(start_date_str)
                # Convert to datetime and make timezone-aware
                start_date = datetime.combine(start_date, datetime.min.time())  # Convert date to datetime (00:00:00)
                start_date = make_aware(start_date, timezone=get_current_timezone())
            else:
                # Default to the first day of the current month if start_date is not provided
                start_date = make_aware(
                    datetime(current_year, current_month, 1),
                    timezone=get_current_timezone()
                )

            if end_date_str:
                end_date = parse_date(end_date_str)
                # Convert to datetime and make timezone-aware
                end_date = datetime.combine(end_date, datetime.max.time())  # Convert date to datetime (23:59:59)
                end_date = make_aware(end_date, timezone=get_current_timezone())
            else:
                # Default to the current date if end_date is not provided
                end_date = now

            # Validate date order
            if start_date > end_date:
                return Response(
                    {"error": "start_date cannot be greater than end_date."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Filter data for the given date range
            cost_data = (
                AIResponse.objects
                .filter(created_at__range=[start_date, end_date])
                .aggregate(total_cost=Sum('total_cost'))
            )

            # Extract the total cost, handle the case where no data is found
            total_cost = cost_data['total_cost'] or 0.0

            # Extract the month and year from start_date and end_date
            start_month = start_date.strftime('%B')  # Full month name, e.g., "January"
            start_year = start_date.year
            end_month = end_date.strftime('%B')
            end_year = end_date.year

            # Prepare the response data
            response_data = {
                "status": "success",
                "service_name": "OpenAI_service",
                "total_cost": f"${total_cost:.4f}",
                "start_month": start_month,
                "start_year": start_year,
                "end_month": end_month,
                "end_year": end_year
            }

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

###end
class SD_daterangeCostAPI(APIView):
    """
    API to calculate the overall cost of generated captions within a given date range.
    """
    def get(self, request, *args, **kwargs):
        try:
            # Get query parameters for start_date and end_date
            start_date_str = request.query_params.get('start_date')
            end_date_str = request.query_params.get('end_date')

            # Get the current date if no dates are provided
            now = timezone.now()
            current_year = now.year
            current_month = now.month

            # Parse the date strings into actual dates
            if start_date_str:
                start_date = parse_date(start_date_str)
                # Convert to datetime and make timezone-aware
                start_date = datetime.combine(start_date, datetime.min.time())  # Convert date to datetime (00:00:00)
                start_date = make_aware(start_date, timezone=get_current_timezone())
            else:
                # Default to the first day of the current month if start_date is not provided
                start_date = make_aware(
                    datetime(current_year, current_month, 1),
                    timezone=get_current_timezone()
                )

            if end_date_str:
                end_date = parse_date(end_date_str)
                # Convert to datetime and make timezone-aware
                end_date = datetime.combine(end_date, datetime.max.time())  # Convert date to datetime (23:59:59)
                end_date = make_aware(end_date, timezone=get_current_timezone())
            else:
                # Default to the current date if end_date is not provided
                end_date = now

            # Validate date order
            if start_date > end_date:
                return Response(
                    {"error": "start_date cannot be greater than end_date."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Filter data for the given date range
            cost_data = (
                ImageGenerationSD_DB.objects
                .filter(created_at__range=[start_date, end_date])
                .aggregate(total_cost=Sum('total_cost_in_dollar'))
            )

            print(cost_data,"00000000000")
            # Extract the total cost, handle the case where no data is found
            total_cost = cost_data['total_cost'] or 0.0

            print(total_cost,"-------------total cost")
            # Prepare the response data
            response_data = {
                "status": "success",
                "service_name": "StabilityAI_service",
                "total_cost": f"${total_cost:.4f}"
            }

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


####User profile API based on surveys 29 OCT:
class UserBasedProfileAPI(APIView):
    def post(self, request, *args, **kwargs):
        # Get user input data
        # name = request.data.get("name", "User")
        age_group = request.data.get("age_group", "Gen Z")
        favorite_topics = request.data.get("favorite_topics", "travel")
        platform = request.data.get("platform", "Instagram")

        # Prompt to send to OpenAI model

        base_prompt = (

            f"Create a personalized, engaging, and visually appealing social media post for a {age_group} "
            f"user who is passionate about {favorite_topics}. The post should feel authentic and resonate "
            f"with their personality, capturing attention on {platform}. Use a catchy caption, relevant emojis, "
            f"and trending hashtags that fit their interests, and end with a call to action that sparks interaction."
        )
        try:
            # Generate response using OpenAI
            response = client_open.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system",
                     "content": "You are a highly skilled AI persona Agent Called ProjectW Agent, reply based on user sentiment in complete sentences."},
                    {"role": "user", "content": base_prompt}
                ],
                max_tokens=1024
            )

            # Get the content generated by OpenAI
            generated_text = response.choices[0].message.content.strip()

            # Return the response with the generated text
            return Response({"categorised_prompt":base_prompt,"categorised_generated_content": generated_text}, status=status.HTTP_200_OK)

        except Exception as e:
            # Handle errors and return response
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

###11 pw Broadcat agentAPI

class Pw_BroadcastAgentAPI(APIView):
    def post(self, request, *args, **kwargs):
        # Get user input data

        favorite_topics = request.data.get("favorite_topics", "travel")

        # Prompt to send to OpenAI model
        base_prompt = (

            f"Create a personalized, engaging, and visually appealing social media post for a user who is passionate about {favorite_topics}. "
            f"The post should feel authentic and resonate with their personality. Use a catchy caption, relevant emojis, and trending hashtags "
            f"that fit their interests, and end with a call to action that sparks interaction."
        )
        try:
            # Generate response using OpenAI
            response = client_open.chat.completions.create(
                model="gpt-4",

                messages=[
                    {"role": "system",
                     "content": "You are a highly skilled AI persona Agent Called PW_Broadcast Agent, reply based on user sentiment in complete sentences."},
                    {"role": "user", "content": base_prompt}
                ],
                max_tokens=1024
            )

            # Get the content generated by OpenAI
            generated_text = response.choices[0].message.content.strip()

            # Return the response with the generated text
            return Response({"categorised_prompt":base_prompt,"Broadcast_generated_content": generated_text}, status=status.HTTP_200_OK)

        except Exception as e:
            # Handle errors and return response
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

##Audio Transcription agent

import math
class Swedish_AudioTranscriptionAPIView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        # Get the uploaded audio file
        audio_file = request.FILES.get("audio_file")
        if not audio_file:
            return JsonResponse({"error": "No audio file provided."}, status=400)

        # Load the audio file with pydub
        audio = AudioSegment.from_file(audio_file)

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
                response = client_open.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en",  # Adjust language as needed
                    response_format="text"
                )
                transcription_text += response + " "  # Append the transcription of each chunk

            # Clean up the temporary file
            os.remove(chunk_filename)

        return JsonResponse({"status":status.HTTP_200_OK,"transcription": transcription_text}, status=200)



####LLama3.2 API Nov18
from gradio_client import Client
class HuggingFaceChatAPI(APIView):
    def post(self, request, *args, **kwargs):
        # Extracting input data from the request
        user_text = request.data.get("text", "")
        files = request.FILES.get("files",None)

        # Initialize the Gradio client for the Hugging Face Space
        client = Client("MadsGalsgaard/Project-W")

        # Preparing the message to send to the Hugging Face Space
        input_message = {
            "text": user_text,
            "files": files
        }

        try:
            # Make the API call to the Hugging Face Space
            result = client.predict(
                message=input_message,
                max_new_tokens=2024,  # Adjust the token limit if needed
                api_name="/chat"  # Update this if the API endpoint is different
            )

            # Returning the result as a response
            return Response({
                "status": "200",
                "response": result
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({
                "error": f"An error occurred: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class VideoGenerationAPIViewRun(APIView):
    def post(self, request):
        # Extract inputs from the request
        prompt_text = request.data.get("prompt_text")
        prompt_image = request.data.get("prompt_image")  # Optional for image+text
        voice_base64 = request.data.get("voice_base64")  # Optional for voice input as base64

        print(prompt_image,prompt_text,"===================skadhkasfakh")
        if not prompt_text and not voice_base64:
            return Response(
                {"error": "At least one input is required: 'prompt_text' or 'voice_command' or 'image_with_prompt'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Process base64-encoded voice input (if provided)
        if voice_base64:
            try:
                voice_bytes = base64.b64decode(voice_base64)
                audio_file = BytesIO(voice_bytes)
                recognizer = sr.Recognizer()
                with sr.AudioFile(audio_file) as source:
                    audio_data = recognizer.record(source)
                    prompt_text = recognizer.recognize_google(audio_data)
            except Exception as e:
                return Response(
                    {"error": f"Failed to process base64 voice input: {str(e)}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        if not prompt_text:
            return Response(
                {"error": "Text prompt could not be derived from the input."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # Initialize the RunwayML client
            client = RunwayML(api_key=RUNWAYML_API_SECRET)
            print(client,"===========client")
            # Decide the task type based on inputs
            if prompt_image:
                task = client.image_to_video.create(
                    model="gen3a_turbo",
                    prompt_image=prompt_image,
                    prompt_text=prompt_text,
                    duration=5,
                )
            else:
                task = client.image_to_video.create(
                    model="gen3a_turbo",
                    # model="gen3a",
                    prompt_text=prompt_text,
                    prompt_image=prompt_image,
                    duration=5
                )

            # Poll the task until it's complete
            time.sleep(10)  # Initial wait
            task = client.tasks.retrieve(task.id)
            while task.status not in ["SUCCEEDED", "FAILED"]:
                time.sleep(10)  # Wait for ten seconds before polling
                task = client.tasks.retrieve(task.id)
                print(type(task),"------------data type")
                task_str=str(task)

                print(task_str, type(task_str),"----------task id")

            # # Check the final status of the task
            if task.status == "SUCCEEDED":
            #     task_data=client.tasks.retrieve(id=task_str)
                return Response(
                    {
                        "status": "SUCCEEDED",
                        "video_url" :task.output,

                    },
                    status=status.HTTP_200_OK,
                )
            else:
                return Response(
                    {"status": "FAILED",
                     "failure": "The provided image was flagged by content moderation.",
                     "failureCode": "SAFETY.INPUT.IMAGE",
                     },status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        except Exception as e:
            # Handle any errors from the RunwayML client
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


####ModelLAb platform 26 only txt to image

import requests
import json
from.models import VideoDB
modellab_key=os.getenv("MODELSLAB_API_KEY", "")
class VideoGenerationAPIView(APIView):
    """
    APIView for generating a video using a text prompt by calling an external API.
    """

    def post(self, request):
        # Extract and validate input data from the request
        user_reference_number = request.data.get('user_reference_number')
        user_email = request.data.get('user_email')
        prompt = request.data.get("prompt")
        negative_prompt = request.data.get("negative_prompt", "low quality")
        height = request.data.get("height", 512)
        width = request.data.get("width", 512)
        num_frames = request.data.get("num_frames", 16)
        num_inference_steps = request.data.get("num_inference_steps", 20)
        guidance_scale = request.data.get("guidance_scale", 7)
        upscale_height = request.data.get("upscale_height", 640)
        upscale_width = request.data.get("upscale_width", 1024)
        upscale_strength = request.data.get("upscale_strength", 0.6)
        upscale_guidance_scale = request.data.get("upscale_guidance_scale", 12)
        upscale_num_inference_steps = request.data.get("upscale_num_inference_steps", 20)
        output_type = request.data.get("output_type", "mp4")
        webhook = request.data.get("webhook", None)
        track_id = request.data.get("track_id", None)

        # Validate required fields
        if not prompt:
            return Response(
                {"error": "'prompt' is required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Prepare the API request payload
        payload = {
            "key": modellab_key,  # Ensure API key is set in the environment variables
            "model_id": "cogvideox",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "upscale_height": upscale_height,
            "upscale_width": upscale_width,
            "upscale_strength": upscale_strength,
            "upscale_guidance_scale": upscale_guidance_scale,
            "upscale_num_inference_steps": upscale_num_inference_steps,
            "output_type": output_type,
            "webhook": webhook,
            "track_id": track_id
        }

        headers = {
            'Content-Type': 'application/json'
        }

        try:
            # Call the external video generation API
            url = "https://modelslab.com/api/v6/video/text2video"
            response = requests.post(url, headers=headers, data=json.dumps(payload))

            # Handle the API response
            if response.status_code == 200:
                response_data = response.json()
                video_url = response_data.get("future_links")  # Assume the API returns a video URL


                # Save prompt and video URL to the database
                VideoDB.objects.create(
                    user_reference_number=user_reference_number,
                    user_email=user_email,
                    prompt=prompt,
                    video_url=video_url[0]
                )

                return Response(response_data, status=status.HTTP_200_OK)

                # return Response(response.json(), status=status.HTTP_200_OK)
            else:
                return Response(
                    {"error": "Failed to generate video", "details": response.json()},
                    status=response.status_code
                )

        except Exception as e:
            # Handle any unexpected errors
            return Response(
                {"error": "An unexpected error occurred", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ImageToVideoGenerationAPIModel(APIView):
    # parser_classes = [MultiPartParser, FormParser]
    # To handle form-data

    def post(self, request):
        user_reference_number = request.data.get('user_reference_number')
        user_email = request.data.get('user_email')
        init_image = request.FILES.get("init_image")  # Uploaded image file
        image_prompt = request.data.get("image_prompt")
        height = request.data.get("height", 512)
        width = request.data.get("width", 512)
        num_frames = request.data.get("num_frames", 25)
        num_inference_steps = request.data.get("num_inference_steps", 20)
        min_guidance_scale = request.data.get("min_guidance_scale", 1)
        max_guidance_scale = request.data.get("max_guidance_scale", 3)
        motion_bucket_id = request.data.get("motion_bucket_id", 20)
        noise_aug_strength = request.data.get("noise_aug_strength", 0.02)
        output_type = request.data.get("output_type", "mp4")
        webhook = request.data.get("webhook",None)
        track_id = request.data.get("track_id",None)

        # Validate required fields
        if not init_image:
            return Response(
                {"error": "'init_image' is required for image-to-video generation."},
                status=status.HTTP_400_BAD_REQUEST
            )
        print(user_email,user_reference_number,"==========mail number")

        try:
            # Save image to media directory
            media_path = "ModelLab_images/"
            if not os.path.exists(media_path):
                os.makedirs(media_path)

            image_path = default_storage.save(media_path + init_image.name, init_image)

            print(image_path,"---------image path detection")
            image_full_path = request.build_absolute_uri(default_storage.url(image_path))
            print(image_full_path,"----------------ahjdsdahh")
            logger.debug(f"Image saved at: {image_full_path}")

            # Prepare payload for API call
            payload = {
                # "key": os.getenv("MODELLAB_API_KEY"),
                "key": modellab_key,
                "model_id": "cogvideox",
                "init_image": image_full_path,
                "prompt": image_prompt,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "num_inference_steps": num_inference_steps,
                "min_guidance_scale": min_guidance_scale,
                "max_guidance_scale": max_guidance_scale,
                "motion_bucket_id": motion_bucket_id,
                "noise_aug_strength": noise_aug_strength,
                "output_type": request.data.get("output_type", "mp4"),
                "webhook": webhook,
                "track_id": track_id,
            }

            headers = {"Content-Type": "application/json"}
            url = "https://modelslab.com/api/v6/video/img2video"
            response = requests.post(url, headers=headers, data=json.dumps(payload))

            print(response,"==========response============")
            if response.status_code == 200:
                response_data = response.json()
                print(response_data,"=============response data")
                video_url = response_data.get("future_links")  # Assume the API returns a video URL
                print(video_url,"==========video url==========")
                # Save prompt and video URL to the database
                VideoDB.objects.create(
                    user_reference_number = user_reference_number,
                    user_email = user_email,
                    prompt=image_full_path,
                    video_url=video_url[0]
                )

                return Response(response_data, status=status.HTTP_200_OK)
                # return Response(response.json(), status=status.HTTP_200_OK)
            else:
                try:
                    error_details = response.json()
                except ValueError:
                    error_details = response.text
                return Response(
                    {"error": "Failed to generate video", "details": error_details},
                    status=response.status_code
                )

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return Response(
                {"error": "An unexpected error occurred", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )





###Voice_To Video generation 29nov

class VoiceCommandVideoModelLabAPI(APIView):
    # parser_classes = (MultiPartParser)  # To handle file uploads

    def post(self, request, *args, **kwargs):
        user_reference_number = request.data.get('user_reference_number')
        user_email = request.data.get('user_email')
        base64_audio = request.data.get("audio")
        audio_bytes = base64.b64decode(base64_audio)
        audio_stream = io.BytesIO(audio_bytes)
        audio_stream.name = "file.mp3"
        negative_prompt = request.data.get("negative_prompt", "low quality")
        height = request.data.get("height", 512)
        width = request.data.get("width", 512)
        num_frames = request.data.get("num_frames", 16)
        num_inference_steps = request.data.get("num_inference_steps", 20)
        guidance_scale = request.data.get("guidance_scale", 7)
        upscale_height = request.data.get("upscale_height", 640)
        upscale_width = request.data.get("upscale_width", 1024)
        upscale_strength = request.data.get("upscale_strength", 0.6)
        upscale_guidance_scale = request.data.get("upscale_guidance_scale", 12)
        upscale_num_inference_steps = request.data.get("upscale_num_inference_steps", 20)
        output_type = request.data.get("output_type", "mp4")
        webhook = request.data.get("webhook", None)
        track_id = request.data.get("track_id", None)

        # Validate required fields
        try:

            transcript_text = client_open.audio.transcriptions.create(
                model="whisper-1",
                file=audio_stream,
                language="en",
                response_format="text"
            )

            print(transcript_text,"-------------transcript")
            if not transcript_text:
                return Response({"error": "Could not transcribe audio"}, status=400)

                # Prepare the payload for the API request
                # Prepare the API request payload
            payload = {
                    "key": modellab_key,  # Ensure API key is set in the environment variables
                    "model_id": "cogvideox",
                    "prompt": transcript_text,
                    "negative_prompt": negative_prompt,
                    "height": height,
                    "width": width,
                    "num_frames": num_frames,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "upscale_height": upscale_height,
                    "upscale_width": upscale_width,
                    "upscale_strength": upscale_strength,
                    "upscale_guidance_scale": upscale_guidance_scale,
                    "upscale_num_inference_steps": upscale_num_inference_steps,
                    "output_type": output_type,
                    "webhook": webhook,
                    "track_id": track_id
                }

            headers = {
                    'Content-Type': 'application/json'
                }

            try:
                # Call the external video generation API
                url = "https://modelslab.com/api/v6/video/text2video"
                response = requests.post(url, headers=headers, data=json.dumps(payload))

                print(response,"---response")
                # Handle the API response
                if response.status_code == 200:
                    response_data = response.json()
                    video_url = response_data.get("future_links")  # Assume the API returns a video URL

                    print(video_url,"---------video url")
                    # Save prompt and video URL to the database
                    VideoDB.objects.create(
                        user_reference_number = user_reference_number,
                        user_email = user_email,
                        prompt=base64_audio,
                        video_url=video_url[0]
                    )

                    return Response(response_data, status=status.HTTP_200_OK)
                    # return Response(response.json(), status=status.HTTP_200_OK)
                else:
                    return Response(
                        {"error": "Failed to generate video", "details": response.json()},
                        status=response.status_code
                    )

            except Exception as e:
                # Handle any unexpected errors
                return Response(
                    {"error": "An unexpected error occurred", "details": str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        except Exception as e:
            return Response({"error": f"An error occurred: {str(e)}"}, status=500)
        finally:
            pass


####new for image thumbnail

from .models import UserLocation
import uuid
# Image generation by user prompt using DALL-E
def dall_e_image_generate(prompt):
    response = client_open.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        # quality="hd",
        n=1,
    )
    # print(response.data[0].url,"-----##############--------response.data[0].url")
    return response.data[0].url
class Notification_LocationAPI(APIView):
    def post(self, request):
        try:
            # Step 1: Extract latitude, longitude, and reference number from request
            lat = request.data.get("lat")
            lng = request.data.get("lng")
            reference_number = request.data.get("reference_number")

            api_key = os.getenv("GOOGLE_MAP_API_KEY")

            if not lat or not lng or not reference_number:
                return Response(
                    {"error": "Latitude, longitude, and reference number are required."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Step 2: Reverse geocoding to get address
            geocode_url = (
                f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={api_key}"
            )
            geocode_response = requests.get(geocode_url)
            geocode_data = geocode_response.json()

            if not geocode_data.get("results"):
                return Response({"error": "Address could not be determined."}, status=status.HTTP_400_BAD_REQUEST)

            address = geocode_data["results"][0]["formatted_address"]

            # Step 3: Generate content using OpenAI GPT
            prompt = (
                f"A user is currently at '{address}'. Create an engaging notification suggesting nearby places "
                f"or activities based on the location."
            )
            ###new for image url
            thumbnail_prompt = f"Create a visually appealing thumbnail image based on the location {address}. Highlight nearby attractions, activities, or notable places in a design that is engaging, compact, and suitable for a thumbnail format."
            image_url=dall_e_image_generate(thumbnail_prompt)
            # print(image_url,"------------noti final image ")
            # running good before
            response = client_open.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a creative notification generator for a travel app."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1020,
                temperature=0.5,
            )

            notification_content = response.choices[0].message.content.strip()
            print(notification_content,"-----------TTTTTTTT")
            ####end
            image_data = requests.get(image_url).content

            # print(image_data,"-------data image++++++++++++")

            file_uuid = str(uuid.uuid4()) + ".jpg"
            folder_path = os.path.join(settings.MEDIA_ROOT, 'ImageUpload')
            print(folder_path, "---folderpath")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            file_path = os.path.join(folder_path, file_uuid)
            with open(file_path, "wb") as buffer:
                buffer.write(image_data)

            final_path = os.path.normpath(f"{file_path.replace(settings.MEDIA_ROOT, settings.MEDIA_URL)}")
            new_imageLink = f"{baseurl(request)}{final_path}"
            # print(new_imageLink, "----------sadsssssssimageeeeeeee")
            # Step 4: Save data to database
            UserLocation.objects.get_or_create(
                reference_number=reference_number,  # Match unique reference_number
                # print(reference_number,"-----------------"),
                defaults={
                    "lat": lat,
                    "lng": lng,
                    "notification_content" : notification_content,
                    "updated_at": request.data.get("updated_at", None),  # Optional timestamp
                }
            )

            # Step 5: Return the generated content
            return Response(
                {"reference_number": reference_number,"title": address, "message": notification_content,"image_url":new_imageLink},
                status=status.HTTP_200_OK
            )

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

##############end####
#
import re
####05022025 donwload jsonl
class CleanDataPipelineAPI(APIView):
    def get(self, request, *args, **kwargs):
        # Step 1: Extract data from the model
        data = AIResponse.objects.values_list('response_text', flat=True)
        cleaned_data = self.clean_data(list(data))

        # Ensure directory exists
        file_directory = os.path.join(settings.MEDIA_ROOT, "cleaned_files")
        os.makedirs(file_directory, exist_ok=True)  # Create the directory if it doesn't exist

        # Define file path
        file_path = os.path.join(file_directory, "cleaned_data.jsonl")

        # Step 2: Save to JSONL file
        with open(file_path, "w", encoding="utf-8") as f:
            for item in cleaned_data:
                f.write(json.dumps({"cleaned_text": item}) + "\n")

        # Step 3: Return the JSONL file as a downloadable response
        with open(file_path, "rb") as f:
            response = HttpResponse(f.read(), content_type="application/jsonl")
            response["Content-Disposition"] = 'attachment; filename="cleaned_data.jsonl"'
            return response
###gs://areax-plm-bucket
    def clean_data(self, data_list):
        """
        Cleans a list of strings by removing special characters and redundant data.
        """
        cleaned_list = []
        seen = set()  # To keep track of redundant entries

        for item in data_list:
            cleaned_item = re.sub(r'[^\w\s]', '', item)  # Remove special characters
            cleaned_item = " ".join(cleaned_item.split())  # Remove extra spaces

            # Check for redundancy
            if cleaned_item not in seen:
                seen.add(cleaned_item)
                cleaned_list.append(cleaned_item)

        return cleaned_list


class ChatRetrieveAPIView(APIView):

    def get(self, request):
        # Extract query parameters for reference_number and email
        reference_number = request.query_params.get('user_reference_number')
        email = request.query_params.get('user_email')

        # Check if at least one of reference_number or email is provided
        if not reference_number and not email:
            return Response(
                {"error": "Either 'reference_number' or 'email' must be provided."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Initialize the queryset
        queryset = AIResponse.objects.all()

        # Filter the queryset by reference_number and email if provided
        if reference_number:
            queryset = queryset.filter(user_reference_number=reference_number)
        if email:
            queryset = queryset.filter(user_email=email)

        # Serialize the filtered data
        serializer = AIResponseSerializer(queryset, many=True)

        # Return the response with serialized data
        return Response(serializer.data, status=status.HTTP_200_OK)

###ALL HistorySync API :

class HistoryAPIView(APIView):
    def get(self, request, *args, **kwargs):
        # Extract query parameters
        data_type = request.query_params.get("type")  # text, image, or voice

        reference_number = request.query_params.get("user_reference_number")
        email = request.query_params.get("user_email")
        print(data_type,reference_number,email)
        # Validate query parameters
        if not data_type:
            return Response(
                {"error": "The 'type' parameter is required (text, image, or voice)."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not reference_number and not email:
            return Response(
                {"error": "Either 'reference_number' or 'email' must be provided."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Initialize queryset
        queryset = None

        # Filter data based on the type
        if data_type == "text":
            print("-------text_=====")
            queryset = AIResponse.objects.filter()
            if reference_number:
                queryset = queryset.filter(user_reference_number=reference_number)
            if email:
                queryset = queryset.filter(user_email=email)
            serializer = AIResponseSerializer(queryset, many=True)

        elif data_type == "image":
            print("-------image=====")

            queryset = GeneratedImage.objects.filter()
            if reference_number:
                queryset = queryset.filter(user_reference_number=reference_number)
            if email:
                queryset = queryset.filter(user_email=email)
            serializer = GeneratedImageSerializer(queryset, many=True)

        elif data_type == "voice":
            print("-------voice=====")
            queryset = OpenaAI_UsageDB.objects.filter()
            if reference_number:
                queryset = queryset.filter(user_reference_number=reference_number)
            if email:
                queryset = queryset.filter(user_email=email)
            serializer = OpenaAIUsageDBSerializer(queryset, many=True)

        elif data_type == "video":
            print("-------video=====")
            queryset = VideoDB.objects.filter()
            if reference_number:
                queryset = queryset.filter(user_reference_number=reference_number)
                print(queryset,"-------quvideo")
            if email:
                queryset = queryset.filter(user_email=email)
                print(queryset,"----------quer video email")
            serializer = VideoGenerated_Serializer(queryset, many=True)

            print(serializer,"------------video serilaise")

        else:
            return Response(
                {"error": "Invalid 'type' parameter. Allowed values: text, image, voice."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Check if no results were found
        if not queryset.exists():
            return Response(
                {"error": "No data found for the provided filters."},
                status=status.HTTP_404_NOT_FOUND
            )

        # Return the serialized data
        return Response(serializer.data, status=status.HTTP_200_OK)


class HistoryDeleteAPI(APIView):
    def delete(self, request, *args, **kwargs):
        # Retrieve parameters from the request
        record_id = request.data.get('id')
        user_email = request.data.get('user_email')

        # Ensure both 'id' and 'user_email' are provided
        if not record_id or not user_email:
            return Response(
                {"message": "'id' and 'user_email' are required for this operation."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Define filters for the query
        filters = {'id': record_id, 'user_email': user_email}

        # Attempt to delete from each database
        for model in [VideoDB, AIResponse, GeneratedImage,OpenaAI_UsageDB]:
            try:
                record = model.objects.get(**filters)
                record.delete()
                return Response(
                    {"message": f"Record deleted successfully from {model.__name__}."},
                    status=status.HTTP_200_OK,
                )
            except model.DoesNotExist:
                continue

        # If no record was found in any of the databases
        return Response({"message": "No matching record found in any database."}, status=status.HTTP_404_NOT_FOUND)



###16jan content generation latlong with ref and email

class Mob_LocationAPI(APIView):
    def post(self, request):
        try:
            # Step 1: Extract latitude, longitude, reference number, and place_name from request
            lat = request.data.get("lat")
            lng = request.data.get("lng")
            place_name = request.data.get("place_name")  # New field
            reference_number = request.data.get("reference_number")
            user_email = request.data.get("user_email")

            # Validate required fields
            if not lat or not lng or not reference_number:
                return Response(
                    {"error": "Latitude, longitude, and reference number are required."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Validate email format
            if user_email and '@' not in user_email:
                return Response(
                    {"error": "Invalid email format."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Fetch Google Maps API key from environment
            api_key = os.getenv("GOOGLE_MAP_API_KEY")
            if not api_key:
                return Response(
                    {"error": "Google API key is missing."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            prompt = (
                f"Imagine you are a travel guide assistant. The user is currently at '{place_name}'. "
                "Create a personalized, engaging, and friendly notification that suggests exciting things to do, "
                "interesting facts, or a special recommendation for the user based on the location. "
                "The notification should sound enthusiastic, inspiring, and informative, as if you're inviting them to explore further."
            )
            # Assuming `client` is an instance of OpenAI's API client, make sure it's initialized
            try:
                response = client_open.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a creative notification generator for a travel app."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1020,
                    temperature=0.5,
                )
                notification_content = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Error generating content with OpenAI: {e}")
                return Response({"error": "Failed to generate notification content."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Step 4: Save data to database
            updated_at = request.data.get("updated_at", timezone.now())  # Ensure timestamp is correctly handled
            UserLocation.objects.update_or_create(
                # reference_number=reference_number,
                user_email=user_email,
                # notification_content = notification_content,
                defaults={
                    "lat": lat,
                    "lng": lng,
                    "place_name": place_name,  # Store place_name in the database
                    "reference_number" : reference_number,
                    "notification_content":notification_content,
                    "updated_at": updated_at
                }
            )

            # Step 5: Return the generated content
            return Response(
                {"status": "success", "Result_content": notification_content, "reference_number": reference_number, "user_email": user_email},
                status=status.HTTP_200_OK
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Error during request to Google Maps API: {e}")
            return Response({"error": "Error during geocoding request."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return Response({"error": "An unexpected error occurred."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

from together import Together
from django.http import StreamingHttpResponse
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

class TogetherChatAPIView(APIView):
    # permission_classes = [AllowAny]

    def post(self, request):
        # TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
        # TOGETHER_API_KEY
        if not TOGETHER_API_KEY:
            return Response({"error": "Missing Together API key"}, status=400)

        client = Together(api_key=TOGETHER_API_KEY)

        user_message = request.data.get("prompt")

        response = client.chat.completions.create(
            # model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            # messages=[{"role": "user","content": user_message}],
            # messages=[{"role": "system", "content": "You are a highly skilled AI persona Agent named is PLM , reply based on user sentiment in complete sentences. And look like replica of userself"},
            messages=[{"role": "system", "content": "You are PLM, an advanced AI persona agent designed to mirror the user's tone, style, and sentiment. Engage in natural, complete sentences, responding as if you were their digital reflection, ensuring conversations feel seamless and personalized."},
                    {"role": "user", "content": user_message}],
        )

        print(response.choices[0].message.content,"----------rrrrrrr--------")
        ai_response = response.choices[0].message.content if response.choices else "No response"

        return Response({"status":"200","response": ai_response})

from .models import UserPLMProfile, ChatHistory
from .utils import generate_response  # Assuming the generate_response function is in utils.py

class PLM_API(APIView):
    permission_classes = [AllowAny]  # Allow all requests (handle authentication manually)

    def post(self, request):
        user_id = request.data.get("user_id")
        user_reference_number = request.data.get("user_reference_number")
        user_email = request.data.get("user_email")
        message = request.data.get("plm_prompt")

        # If user_id is not provided, create a new user
        if not user_id:
            user = User.objects.create(username=f"user_{User.objects.count()+1}")
            user.set_unusable_password()  # Optional: User cannot log in with a password
            user.save()

            # Generate a new authentication token for the user
            token, _ = Token.objects.get_or_create(user=user)

            return Response({"message": "New user created", "user_id": user.id, "token": token.key}, status=201)

        # Authenticate existing users
        if not request.auth:
            return Response({"error": "Authentication credentials were not provided."}, status=401)

        # Try to find the user
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=404)

        # Get or create the UserPLMProfile object
        user_profile, _ = UserPLMProfile.objects.get_or_create(user=user)

        # Load chat history
        chat_history = ChatHistory.objects.filter(user=user_profile).order_by('-timestamp')

        # Prepare chat history messages and responses
        previous_messages = [{"role": "user", "content": chat.message} for chat in chat_history]
        previous_responses = [{"role": "assistant", "content": chat.response} for chat in chat_history]

        # Combine previous messages and responses to form conversation history
        conversation_history = previous_messages + previous_responses

        # Check if the message exists in the chat history
        existing_response = ChatHistory.objects.filter(user=user_profile, message=message).first()

        if existing_response:
            return Response({"user_reference_number":user_reference_number,"user_email":user_email,"message": message,"message_response": existing_response.response})

        # Generate a new response
        response = generate_response(message, user_profile.personality_traits)

        # Save chat history with additional fields
        ChatHistory.objects.create(
            user=user_profile,
            user_reference_number=user_reference_number,
            user_email=user_email,
            message=message,
            response=response
        )

        return Response({"user_reference_number":user_reference_number,"user_email":user_email,"message":message,"message_response": response})

####gemini voice API03march25
from .Gemini_Cloud_voice import speech_to_text,text_to_speech

class PLM_Voice_API(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        user_id = request.data.get("user_id")
            ###new voice
        user_reference_number = request.data.get("user_reference_number", "")
        user_email = request.data.get("user_email", "")
        audio_base64 = request.data.get("audio_base64")

        if not audio_base64:
            return Response({"error": "No audio data provided"}, status=400)

        # Gemini model configuration
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        # client = genai.GenerativeModel(model_name="gemini-2.0-flash", generation_config=generation_config)
        try:
            # Decode Base64 audio and save it
            audio_bytes = base64.b64decode(audio_base64)
            audio_path = "tempo_audio.mp3"

            with open(audio_path, "wb") as audio_file:
                audio_file.write(audio_bytes)

            # Convert speech to text
            message = speech_to_text(audio_path)
            print(message, "----------TEXT DATA after conversion (STT)")

            if not message:
                return Response({"error": "Failed to transcribe audio"}, status=400)
            ###end
            # Authenticate User (Create if not exists)
            if not user_id:
                user = User.objects.create(username=f"user_{User.objects.count()+1}")
                user.set_unusable_password()
                user.save()
                token, _ = Token.objects.get_or_create(user=user)
                return Response({
                    "message": "New user created",
                    "user_id": user.id,
                    "token": token.key,
                    "user_reference_number": user_reference_number,
                    "user_email": user_email
                }, status=201)

            # Get Existing User
            try:
                user = User.objects.get(id=user_id)
            except User.DoesNotExist:
                return Response({"error": "User not found"}, status=404)

            user_profile, _ = UserPLMProfile.objects.get_or_create(user=user)

            # Check Chat History
            existing_response = ChatHistory.objects.filter(user=user_profile, message=message).first()
            if existing_response:
                response_audio_base64 = text_to_speech(existing_response.response)

                # Save to OpenaAI_UsageDB
                OpenaAI_UsageDB.objects.create(
                    user_reference_number=user_reference_number,
                    user_email=user_email,
                    prompt=audio_base64,
                    response=existing_response.response,
                    audio_base64=response_audio_base64,
                )

                return Response({
                    "user_reference_number": user_reference_number,
                    "user_email": user_email,
                    "transcript": message,
                    # "voice_prompt": audio_base64,
                    "response": existing_response.response,
                    "response_audio_base64": response_audio_base64
                })

            # Generate AI Response
            response_text = generate_response(message, user_profile.personality_traits)

            # Save Chat History
            ChatHistory.objects.create(user=user_profile, message=message, response=response_text)

            # Convert AI Response to Speech
            response_audio_base64 = text_to_speech(response_text)

            # Save to OpenaAI_UsageDB
            generated_entry = OpenaAI_UsageDB.objects.create(
                user_reference_number=user_reference_number,
                user_email=user_email,
                prompt=audio_base64,
                response=response_text,
                audio_base64=response_audio_base64,
            )

            return Response({
                "user_reference_number": user_reference_number,
                "user_email": user_email,
                "generated_entry_id": generated_entry.id,
                "transcript": message,
                "voice_prompt": audio_base64,
                "response": response_text,
                "response_audio_base64": response_audio_base64
            })

        except Exception as e:
            print("Error processing voice request:", str(e))
            return Response({"error": "Internal Server Error", "details": str(e)}, status=500)

from google.genai import types
from PIL import Image
from io import BytesIO
import base64
import os

from rest_framework.views import APIView
from rest_framework.permissions import AllowAny

# Initialize Gemini client with API key



class ImageGenerationAPIGemini(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        prompt = request.data.get("prompt")
        num_images = request.data.get("num_images", 1)  # Default is 1 image
        user_reference_number = request.data.get("user_reference_number")
        user_email = request.data.get("user_email")
        negative_prompt = request.data.get("negative_prompt", "")  # Optional

        if not prompt:
            return Response({"error": "No prompt provided"}, status=400)

        try:
            # Generate images using Gemini API
            response = genai_client.models.generate_images(
                model="imagen-3.0-generate-002",
                prompt=prompt,
                config=types.GenerateImagesConfig(number_of_images=num_images),
            )

            # Extract base64 images correctly
            base64_images = [img.image.image_bytes for img in response.generated_images]

            if not base64_images:
                return Response({"error": "No valid image data"}, status=500)

            saved_images = []

            for base64_data in base64_images:
                # Decode base64 to binary
                image_data = base64.b64decode(base64_data)

                # Generate a unique file name using UUID
                unique_id = uuid.uuid4()
                file_name = f"generated_image_{unique_id}.png"  # Save as PNG

                # Define full path for saving
                file_path = os.path.join(settings.MEDIA_ROOT, file_name)

                # Save the image to the media directory
                with open(file_path, "wb") as img_file:
                    img_file.write(image_data)

                # Construct the accessible image URL
                image_url = f"{settings.MEDIA_URL}{file_name}"
                full_image_url = baseurl(request) + image_url  # Add base URL

                # Save the image entry in the database with the full URL
                generated_image = GeneratedImage.objects.create(
                    prompt=prompt,
                    user_reference_number=user_reference_number,
                    user_email=user_email,
                    negative_prompt=negative_prompt,
                    image=full_image_url,  # Store full URL in the database
                )

                saved_images.append({
                    "generated_image_id": generated_image.id,
                    "user_reference_number": user_reference_number,
                    "user_email": user_email,
                    "image": full_image_url, # Return full URL
                    # "image": base64_images  # Return full URL
                })

            return Response({
                "prompt": prompt,
                "generated_images": saved_images
            })

        except Exception as e:
            return Response({"error": str(e)}, status=500)



##############GEMINI VOICECHAT 06march025


import google.generativeai as genai
###NewVoice cleaned code
class Gemini_voiceChat(APIView):

    def post(self, request, *args, **kwargs):
        user_reference_number = request.data.get("user_reference_number", "")
        user_email = request.data.get("user_email", "")
        base64_audio = request.data.get("audio")

        if not base64_audio:
            return Response({"error": "No audio data provided"}, status=400)

        # Gemini model configuration
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        client = genai.GenerativeModel(model_name="gemini-2.0-flash", generation_config=generation_config)
        try:
            # Decode Base64 audio and save it
            audio_bytes = base64.b64decode(base64_audio)
            audio_path = "temp_audio.mp3"

            with open(audio_path, "wb") as audio_file:
                audio_file.write(audio_bytes)

            # Convert speech to text
            transcript_text = speech_to_text(audio_path)
            print(transcript_text, "----------TEXT DATA after conversion (STT)")

            if not transcript_text:
                return Response({"error": "Failed to transcribe audio"}, status=400)

            # Format chat prompt for Gemini
            chat_prompt = f"You are a highly skilled AI persona agent. Reply based on user sentiment in complete sentences. User input: {transcript_text}"

            # Generate AI response using Gemini
            # response = client.generate_content()
            # model = genai.GenerativeModel(model_name="gemini-2.0-flash", generation_config=generation_config)

            response = client.generate_content(chat_prompt)

            print(response, "--------------Audio Response from Gemini")

            # Handle blocked responses
            if response.candidates and response.candidates[0].finish_reason == "SAFETY":
                return Response({"error": "Sorry, I can't respond to that. Please try a different input."}, status=403)

            # Extract AI-generated response
            ai_response = response.text if hasattr(response, "text") else None

            if not ai_response:
                return Response({"error": "Failed to generate an AI response"}, status=500)

            # Convert AI response text to speech
            audio_file_path = text_to_speech(ai_response)

            # Encode the generated speech to Base64
            with open(audio_file_path, "rb") as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")

            # Save API usage in the database
            OpenaAI_UsageDB.objects.create(
                user_reference_number=user_reference_number,
                user_email=user_email,
                prompt=base64_audio,
                response=ai_response,
                audio_base64=audio_base64
            )

            # Clean up temporary files
            os.remove(audio_path)
            os.remove(audio_file_path)

            return Response({
                "status": status.HTTP_200_OK,
                "user_reference_number": user_reference_number,
                "user_email": user_email,
                "transcript": transcript_text,
                "ai_response": ai_response,
                "audio": audio_base64
            }, status=200)

        except Exception as e:
            return Response({"error": f"An error occurred: {str(e)}"}, status=500)


####140425
from google.cloud.speech import RecognitionAudio,RecognitionConfig
import boto3
from google.cloud import speech,texttospeech
from rest_framework.parsers import JSONParser
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_S3_REGION_NAME")
)


####This is running perfect170425
from uuid import UUID
from django.shortcuts import get_object_or_404


from .veo_videoLogic import generate_video
class GeminiSmartAPIView(APIView):
    parser_classes = [JSONParser]

    def post(self, request):
        user_reference_number = request.data.get("user_reference_number", "")
        user_email = request.data.get("user_email", "")
        chat_session_id = request.data.get("chat_session_id")  # <-- UUID
        text_prompt = request.data.get("text")
        audio_base64 = request.data.get("audio")
        image_prompt = request.data.get("image_prompt")
        video_prompt = request.data.get("video_prompt")  # <--- NEW
        aspect_ratio = request.data.get("aspect_ratio", "16:9")  # optional

        try:
            text_response = None
            audio_response = None
            image_response = None
            audio_response_base64 = None
            video_response = None  # <--- NEW

            # Check if chat_session_id is a valid UUID
            if chat_session_id:
                try:
                    chat_session = ChatSession.objects.get(session_id=UUID(chat_session_id))
                except ChatSession.DoesNotExist:
                    chat_session = None
            else:
                chat_session = None

            if not chat_session:
                chat_session = ChatSession.objects.create(
                    user_reference_number=user_reference_number,
                    user_email=user_email,
                )

            if audio_base64:
                if self.is_valid_base64(audio_base64):
                    transcript = self.transcribe_audio(audio_base64)
                    audio_response = self.generate_text(transcript)
                    audio_response_base64 = self.synthesize_speech_to_base64(audio_response)
                else:
                    return Response({"error": "Invalid audio format."}, status=400)

            if image_prompt:
                image_response = self.generate_image(image_prompt)

            if text_prompt:
                text_response = self.generate_text(text_prompt)

            if video_prompt:
                video_data = generate_video(video_prompt, aspect_ratio=aspect_ratio)
                video_response = video_data["video_url"]

            smart_response = SmartResponse.objects.create(
                chat_session=chat_session,
                user_reference_number=user_reference_number,
                user_email=user_email,
                text=text_prompt,
                audio=audio_base64,
                image_prompt=image_prompt,
                text_response=text_response,
                audio_response=audio_response,
                image_response=image_response,
                video_prompt=video_prompt,
                video_response=video_response,
            )

            return Response({
                "id": smart_response.id,
                "chat_session_id": str(chat_session.session_id),
                "user_reference_number": user_reference_number,
                "user_email": user_email,
                "text_prompt": text_prompt,
                "text_response": text_response if text_prompt else None,
                "audio_response": None if text_prompt or image_prompt else audio_response_base64,
                "audio_prompt": audio_base64,
                "image": None if text_prompt or audio_base64 else image_response,
                "image_prompt": image_prompt,
                "video_prompt": video_prompt,
                "video_url": video_response
            })

        except Exception as e:
            return Response({"error": str(e)}, status=500)
    def is_valid_base64(self, data):
        try:
            base64.b64decode(data)
            return True
        except Exception:
            return False

    def transcribe_audio(self, base64_audio):
        audio_bytes = base64.b64decode(base64_audio)
        client = speech.SpeechClient()

        audio = RecognitionAudio(content=audio_bytes)
        config = RecognitionConfig(
            encoding=RecognitionConfig.AudioEncoding.MP3,
            sample_rate_hertz=16000,
            language_code="en-US"
        )

        response = client.recognize(config=config, audio=audio)

        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript + " "

        return transcript.strip()

    def synthesize_speech_to_base64(self, text):
        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        audio_base64 = base64.b64encode(response.audio_content).decode("utf-8")
        return audio_base64

    def generate_text(self, prompt):
        # Assume `client` is properly initialized and set up
        response = genai_client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=[prompt]
        )
        return response.text.strip()

    def generate_image(self, prompt):
        response = genai_client.models.generate_images(
            model='imagen-3.0-generate-002',
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                include_rai_reason=True,
                output_mime_type='image/jpeg',
            ),
        )

        if not response.generated_images:
            raise Exception("No image was generated by Gemini.")
        #Extract the image bytes
        image_base64 = response.generated_images[0].image.image_bytes
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes))
        #Convert image to file like oobject
        image_io = BytesIO()
        image.save(image_io, format="JPEG")
        image_io.seek(0)

        filename = f"generated_images/{uuid.uuid4()}.jpg"

        bucket = os.getenv("AWS_STORAGE_BUCKET_NAME")
        region = os.getenv("AWS_S3_REGION_NAME")

        s3.upload_fileobj(
            image_io,
            bucket,
            filename,
            ExtraArgs={'ContentType': 'image/jpeg'}
        )

        image_url = f"https://{bucket}.s3-{region}.amazonaws.com/{filename}"

        return image_url


class ChatHistoryAPI(APIView):
    def get(self, request, *args, **kwargs):
        session_id = request.query_params.get("session_id")

        if not session_id:
            return Response({"error": "session_id is required as a query parameter."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            session_uuid = uuid.UUID(session_id)
        except ValueError:
            return Response({"error": "Invalid session_id format. Must be a valid UUID."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            chat_session = ChatSession.objects.get(session_id=session_uuid)
            responses = SmartResponse.objects.filter(chat_session=chat_session).order_by("created_at")

            data = [
                {
                    "id" :resp.id,
                    "user_reference_number" : resp.user_reference_number,
                    "user_email" : resp.user_email,
                    "text_prompt": resp.text,
                    "text_response": resp.text_response,
                    "audio": resp.audio,
                    "audio_response": resp.audio_response,
                    "image_prompt": resp.image_prompt,
                    "image_response": resp.image_response,
                    "video_prompt" : resp.video_prompt,
                    "video_response" : resp.video_response,
                    "created_at": resp.created_at,
                }
                for resp in responses
            ]

            return Response({"session_id": session_id, "history": data})
        except ChatSession.DoesNotExist:
            return Response({"error": "Chat session not found."}, status=status.HTTP_404_NOT_FOUND)


#####Video generation # Replace the `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` values
# with appropriate values for your project.
# views.py
import os
import time
import logging
import base64
from urllib.parse import urlparse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateVideosConfig
from google.cloud import storage

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = "us-central1"
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

if not PROJECT_ID or not GCS_BUCKET_NAME:
    raise EnvironmentError("Environment variables missing (GOOGLE_CLOUD_PROJECT or GCS_BUCKET_NAME)")

genai_veo_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
storage_client = storage.Client(project=PROJECT_ID)

def parse_gcs_uri(uri):
    parsed = urlparse(uri)
    if parsed.scheme != "gs":
        raise ValueError(f"Invalid GCS URI: {uri}")
    return parsed.netloc, parsed.path.lstrip('/')

def get_signed_gcs_url(bucket_name, object_name, expiration=3600):
    """Generate a signed URL to access the GCS object (UBLA-compatible)."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)

    logger.info(f"Checking existence of blob: {bucket_name}/{object_name}")
    for _ in range(10):
        if blob.exists():
            logger.info("Blob found, generating signed URL.")
            return blob.generate_signed_url(version="v4", expiration=expiration)
        logger.warning("Blob not found yet, retrying in 2 seconds...")
        time.sleep(2)

    raise RuntimeError("Blob does not exist in GCS after waiting.")

class GenerateVideoAPI(APIView):
    def post(self, request):
        prompt = request.data.get("video_prompt")
        aspect_ratio = request.data.get("aspect_ratio", "16:9")

        if not prompt:
            return Response({"error": "Prompt is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            output_prefix = f"gs://{GCS_BUCKET_NAME}/generated_videos/{int(time.time())}"
            logger.info(f"Starting video generation with prompt: {prompt}")

            operation = genai_veo_client.models.generate_videos(
                model="veo-2.0-generate-001",
                prompt=prompt,
                config=GenerateVideosConfig(
                    aspect_ratio=aspect_ratio,
                    output_gcs_uri=output_prefix,
                    number_of_videos=1,
                    duration_seconds=5,
                ),
            )

            # Wait for the operation to complete
            timeout = 900  # 15 mins
            poll_interval = 21
            start_time = time.time()

            while True:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    raise TimeoutError("Video generation timed out.")

                operation = genai_veo_client.operations.get(operation)
                if operation.done:
                    break
                logger.info("Waiting for video generation to complete...")
                time.sleep(poll_interval)

            if operation.error:
                raise RuntimeError(f"Generation failed: {operation.error}")

            videos = operation.result.generated_videos
            if not videos:
                raise RuntimeError("No videos returned in the result.")

            video_uri = videos[0].video.uri
            logger.info(f"Generated video GCS URI: {video_uri}")

            # Extract bucket and object name
            bucket_name, object_name = parse_gcs_uri(video_uri)

            # Generate signed URL instead of making it public
            signed_url = get_signed_gcs_url(bucket_name, object_name)

            return Response({
                "message": "Video generated successfully",
                "video_url": signed_url  # Time-limited public URL
            })

        except Exception as e:
            logger.exception("Video generation failed")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class AllSessionIDsAPI(APIView):
    def get(self, request):
        sessions = ChatSession.objects.all()
        print(sessions,"-----------session")
        serializer = ChatSessionSerializer(sessions, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

### with audio title and text tile api 240425

class AllSessionTitleAPI(APIView):
    def get(self, request, *args, **kwargs):
        session_id = request.query_params.get("session_id")
        user_reference_number = request.query_params.get("user_reference_number")
        user_email = request.query_params.get("user_email")

        try:
            if user_email and not SmartResponse.objects.filter(user_email=user_email).exists():
                return Response({"error": f"No records found for user_email: {user_email}"}, status=status.HTTP_404_NOT_FOUND)

            def get_title(queryset):
                latest = queryset.order_by("-created_at").first()
                if latest:
                    title = latest.text if latest.text else latest.audio_transcript
                    return {
                        "session_id": str(latest.chat_session.session_id),
                        "user_reference_number": latest.user_reference_number,
                        "user_email": latest.user_email,
                        "title": title[:50] if title else "No title available."
                    }
                return None

            if session_id:
                try:
                    session_uuid = uuid.UUID(session_id)
                except ValueError:
                    return Response({"error": "Invalid session_id format. Must be a valid UUID."}, status=status.HTTP_400_BAD_REQUEST)

                chat_session = ChatSession.objects.get(session_id=session_uuid)
                query = SmartResponse.objects.filter(chat_session=chat_session).exclude(text__exact="", audio_transcript__exact="")
                if user_reference_number:
                    query = query.filter(user_reference_number=user_reference_number)
                if user_email:
                    query = query.filter(user_email=user_email)

                title_data = get_title(query)
                if not title_data:
                    return Response({"session_id": session_id, "title": "No matching prompt found."})
                return Response(title_data)

            else:
                session_titles = []
                all_sessions = ChatSession.objects.all()
                for session in all_sessions:
                    query = SmartResponse.objects.filter(chat_session=session).exclude(text__exact="", audio_transcript__exact="")
                    if user_reference_number:
                        query = query.filter(user_reference_number=user_reference_number)
                    if user_email:
                        query = query.filter(user_email=user_email)

                    title_data = get_title(query)
                    if title_data:
                        session_titles.append(title_data)

                return Response(session_titles, status=status.HTTP_200_OK)

        except ChatSession.DoesNotExist:
            return Response({"error": "Chat session not found."}, status=status.HTTP_404_NOT_FOUND)

####One feed API with new chnages according to Vikas
from google.genai.types import GenerateContentConfig, Modality
class GenerateSmartContentAPI(APIView):
    parser_classes = [MultiPartParser, JSONParser]
    def post(self, request, *args, **kwargs):
        # Input payload
        user_text = request.data.get('text')
        audio_base64 = request.data.get('audio')
        image_file = request.FILES.get('user_image') #In binary form
        user_image_url_input = request.data.get("user_image_url")
        user_email = request.data.get('user_email')
        user_reference_number = request.data.get('user_reference_number')
        aspect_ratio = request.data.get('aspect_ratio', '16:9')

        # Generate or retrieve chat session
        session_id = request.data.get('chat_session_id')
        if session_id:
            chat_session, _ = ChatSession.objects.get_or_create(session_id=session_id)
        else:
            chat_session = ChatSession.objects.create(session_id=str(uuid.uuid4()))
        try:
            # Initialize response holders
            text_response = None
            image_response = None
            video_response = None
            audio_response_base64 = None
            transcript = None
            user_image_url = None

            # Audio-to-Audio: Only audio provided
            if audio_base64 and not user_text:
                if self.is_valid_base64(audio_base64):
                    transcript = self.transcribe_audio(audio_base64)
                    audio_response = self.generate_text(transcript)
                    audio_response_base64 = self.synthesize_speech_to_base64(audio_response)
                else:
                    return Response({"error": "Invalid audio format."}, status=400)

            # Text-based prompt
            elif user_text and not audio_base64:
                lower_prompt = user_text.lower()

                # if any(keyword in lower_prompt for keyword in ["edit", "modify", "change"]) and image_file:
                #     user_image_url = self.edit_image_with_gemini(image_file, text_prompt, user_email,
                #                                                  user_reference_number)

                if any(keyword in lower_prompt for keyword in ["edit", "modify", "change"]):
                    if image_file:
                        print("------iffff modify")
                        image_response = self.edit_image_with_gemini(image_file, user_text, user_email,
                                                                     user_reference_number)
                    elif user_image_url_input:
                        print("elsif mofditggg")
                        # Download image from URL and convert to file-like object
                        response = requests.get(user_image_url_input)
                        if response.status_code == 200:
                            image_stream = BytesIO(response.content)
                            image_response = self.edit_image_with_gemini(image_stream, user_text, user_email,
                                                                         user_reference_number)
                        else:
                            return Response({"error": "Could not fetch image from URL."}, status=400)
                # elif "image" in lower_prompt or "generate an image" in lower_prompt or "create an image" in lower_prompt or "make an image" in lower_prompt:
                #     image_response = self.generate_image(text_prompt)

                elif any(keyword in lower_prompt for keyword in["generate an image", "create an image", "make an image", "image"]):
                    print("---generation image")
                    image_response = self.generate_image(user_text)

                # elif "video" in lower_prompt or "generate a video" in lower_prompt or "create a video" in lower_prompt or "make a video" in lower_prompt:
                #     video_data = generate_video(text_prompt, aspect_ratio=aspect_ratio)
                #     video_response = video_data.get("video_url")

                elif any(keyword in lower_prompt for keyword in["generate a video", "create a video", "make a video", "video"]):
                    video_data = generate_video(user_text, aspect_ratio=aspect_ratio)
                    print("=====generation video")
                    video_response = video_data.get("video_url")

                else:
                    print("----text inut nly")
                    text_response = self.generate_text(user_text)

            # Save the generated response
            smart_response = SmartResponse.objects.create(
                chat_session=chat_session,
                user_reference_number=user_reference_number,
                user_email=user_email,
                text=user_text,
                audio=audio_base64,
                text_response=text_response,
                audio_response=audio_response_base64,
                audio_transcript = transcript,
                image_response=image_response,
                video_response=video_response,
                user_image_url=user_image_url_input
            )

            # Return response
            return Response({
                "id": smart_response.id,
                "chat_session_id": str(chat_session.session_id),
                "user_reference_number": user_reference_number,
                "user_email": user_email,
                "user_text": user_text,
                "text_response": text_response,
                "audio_prompt": audio_base64,
                "audio_transcript": transcript,
                "audio_response": audio_response_base64,
                "image": image_response,
                "user_image_url":user_image_url_input,
                "video_prompt" : "null",
                "video_url": video_response,
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=500)
    def is_valid_base64(self, data):
        try:
            base64.b64decode(data)
            return True
        except Exception:
            return False
    def transcribe_audio(self, base64_audio):
        audio_bytes = base64.b64decode(base64_audio)
        client = speech.SpeechClient()
        audio = RecognitionAudio(content=audio_bytes)
        config = RecognitionConfig(
            encoding=RecognitionConfig.AudioEncoding.MP3,
            sample_rate_hertz=16000,
            language_code="en-US"
        )

        response = client.recognize(config=config, audio=audio)
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript + " "

        return transcript.strip()

    def synthesize_speech_to_base64(self, text):
        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        audio_base64 = base64.b64encode(response.audio_content).decode("utf-8")
        return audio_base64

    def generate_text(self, prompt):
        response = genai_client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=[prompt]
        )
        return response.text.strip()

    def generate_image(self, prompt):
        print(prompt,"----++++++++++Image prompt")
        response = genai_client.models.generate_images(
            model='imagen-3.0-generate-002',
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                include_rai_reason=True,
                output_mime_type='image/jpeg',
            ),
        )
        print(response,"-------------data response image ")
        if not response.generated_images:
            raise Exception("No image was generated by Gemini.")
        # Extract the image bytes
        image_base64 = response.generated_images[0].image.image_bytes
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes))
        # Convert image to file like oobject
        image_io = BytesIO()
        image.save(image_io, format="JPEG")
        image_io.seek(0)

        filename = f"generated_images/{uuid.uuid4()}.jpg"

        bucket = os.getenv("AWS_STORAGE_BUCKET_NAME")
        region = os.getenv("AWS_S3_REGION_NAME")

        s3.upload_fileobj(
            image_io,
            bucket,
            filename,
            ExtraArgs={'ContentType': 'image/jpeg'}
        )

        image_url = f"https://{bucket}.s3-{region}.amazonaws.com/{filename}"

        return image_url
    def edit_image_with_gemini(self, image_file, edit_prompt, email, reference_number):

        image = Image.open(image_file)
        print(image,"=======within image fn")
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[image, edit_prompt],
            config=GenerateContentConfig(response_modalities=[Modality.TEXT, Modality.IMAGE]),
        )
        # print(response,"---------response+++++++++++++WWWWWWWWWWWRRRRRRRRRRRRR")
        edited_image_url = None
        print(response.candidates[0],"-------------PARTTSTS")
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                edited_bytes = part.inline_data.data

                image_bytes = base64.b64decode(edited_bytes)
                image = Image.open(BytesIO(image_bytes))
                # Convert image to file like oobject
                image_io = BytesIO()
                image.save(image_io, format="JPEG")
                image_io.seek(0)

                filename = f"edited_images/{uuid.uuid4()}.jpg"
                print(filename,"---------finanalemm")
                bucket = os.getenv("AWS_STORAGE_BUCKET_NAME")
                print(bucket,"----BUCKET")
                region = os.getenv("AWS_S3_REGION_NAME")
                print(region,"-------region")

                s3.upload_fileobj(image_io, bucket, filename, ExtraArgs={'ContentType': 'image/jpeg'})
                edited_image_url = f"https://{bucket}.s3-{region}.amazonaws.com/{filename}"

        return edited_image_url

##Deleteapi230425
class DeleteChatHistoryAPI(APIView):
    def delete(self, request, *args, **kwargs):
        session_id = request.query_params.get("session_id")

        if not session_id:
            return Response({"error": "session_id is required as a query parameter."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            session_uuid = uuid.UUID(session_id)
        except ValueError:
            return Response({"error": "Invalid session_id format. Must be a valid UUID."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            chat_session = ChatSession.objects.get(session_id=session_uuid)
            deleted_count, _ = SmartResponse.objects.filter(chat_session=chat_session).delete()
            chat_session.delete()  # Optionally remove the session record too

            return Response({
                "message": f"Successfully deleted chat session and {deleted_count} associated messages.",
                "session_id": session_id
            }, status=status.HTTP_200_OK)
        except ChatSession.DoesNotExist:
            return Response({"error": "Chat session not found."}, status=status.HTTP_404_NOT_FOUND)
