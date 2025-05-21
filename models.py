from django.db import models
from django.contrib.auth.models import User
# Create your models here.


class AIContentDb(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE,null=True,blank=True)
    email=models.EmailField(max_length=100,null=True,blank=True)
    image_url = models.URLField(max_length=500)
    caption = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username}"


##Google userCredential

class UserCredentials(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    access_token = models.CharField(max_length=255)
    refresh_token = models.CharField(max_length=255)
    token_expiry = models.DateTimeField()

#####18jul AI AgentUser input DB
class UserInput(models.Model):
    text = models.TextField(null=True,blank=True)
    tone = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

###AI response AI

class AIResponse(models.Model):
    user_input = models.ForeignKey(UserInput, related_name='responses', on_delete=models.CASCADE)
    user_reference_number = models.CharField(max_length=200, null=True, blank=True)
    user_email = models.CharField(max_length=200, null=True, blank=True)
    response_text = models.TextField(null=True,blank=True)
    prompt_tokens = models.IntegerField(null=True,blank=True)
    completion_tokens = models.IntegerField(null=True,blank=True)
    total_tokens = models.IntegerField(null=True,blank=True)
    input_cost = models.DecimalField(max_digits=10, decimal_places=4,null=True,blank=True)
    output_cost = models.DecimalField(max_digits=10, decimal_places=4,null=True,blank=True)
    total_cost = models.DecimalField(max_digits=10, decimal_places=4,null=True,blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

###Image database SD

class GeneratedImage(models.Model):
    prompt = models.TextField()
    user_reference_number = models.CharField(max_length=200, null=True, blank=True)
    user_email = models.CharField(max_length=200, null=True, blank=True)
    negative_prompt = models.TextField(blank=True, null=True)
    image = models.TextField(null=True,blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

###User Feedback DB
class Feedback(models.Model):
    ai_response = models.ForeignKey(AIResponse, on_delete=models.CASCADE, null=True,blank=True)
    rating = models.IntegerField(default=0)  # Rating between 1 and 5
    user_feedback = models.TextField(blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)

####Real Time Cost Database 30august:

class OpenaAI_UsageDB(models.Model):
    # user = models.ForeignKey(User, on_delete=models.CASCADE)
    user_reference_number = models.CharField(max_length=200,null=True,blank=True)
    user_email = models.CharField(max_length=200,null=True,blank=True)
    prompt = models.TextField()
    response = models.TextField()
    audio_base64 = models.TextField(null=True,blank=True)
    prompt_tokens = models.IntegerField(null=True,blank=True)
    completion_tokens = models.IntegerField(null=True,blank=True)
    total_tokens = models.IntegerField(null=True,blank=True)
    input_cost = models.DecimalField(max_digits=10, decimal_places=4,null=True,blank=True)
    output_cost = models.DecimalField(max_digits=10, decimal_places=4,null=True,blank=True)
    total_cost = models.DecimalField(max_digits=10, decimal_places=4,null=True,blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Chat of {self.user_email} on {self.created_at}"


###img DB_13SEp
class ImageGenerationSD_DB(models.Model):
    prompt = models.CharField(max_length=255,null=True,blank=True)  # Store the prompt
    image_path = models.CharField(max_length=255,null=True,blank=True)  # Path to the generated image
    total_cost_in_dollar = models.DecimalField(max_digits=10, decimal_places=4)  # Store the cost
    created_at = models.DateTimeField(auto_now_add=True)  # Timestamp of generation

    def __str__(self):
        return f"ImageGeneration(prompt={self.prompt}, cost=${self.total_cost_in_dollar})"


###Google gemini database19sep

class ImageCaptionGeminiDB(models.Model):
    image = models.ImageField(upload_to='uploaded_images/')
    caption = models.TextField()
    input_cost = models.DecimalField(max_digits=10, decimal_places=5, default=0.0)
    output_cost = models.DecimalField(max_digits=10, decimal_places=5, default=0.0)
    total_cost = models.DecimalField(max_digits=10, decimal_places=5, default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.caption[:50]



####Userlocation DB06dec
from django.utils.timezone import now

class UserLocation(models.Model):
    reference_number = models.CharField(max_length=255, unique=True)
    user_email = models.CharField(max_length=100,null=True,blank=True)
    lat = models.DecimalField(max_digits=9, decimal_places=6)
    lng = models.DecimalField(max_digits=9, decimal_places=6)
    place_name = models.CharField(max_length=255, blank=True, null=True)
    notification_content = models.TextField(null=True,blank=True)
    updated_at = models.DateTimeField(default=now,null=True,blank=True)  # Default to the current timestamp
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.reference_number} - ({self.lat}, {self.lng})"



##VIdeo generation DB

class VideoDB(models.Model):
    user_reference_number = models.CharField(max_length=200, null=True, blank=True)
    user_email = models.CharField(max_length=200, null=True, blank=True)
    prompt = models.TextField(blank=True,null=True)
    video_url = models.URLField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Video for prompt: {self.prompt}"


# Model for storing user preferences and traits
class UserPLMProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    preferences = models.TextField(default="default_preferences")
    personality_traits = models.TextField(default="friendly, curious, helpful")

    def __str__(self):
        return self.user.username

# Model for storing chat history
class ChatHistory(models.Model):
    user = models.ForeignKey(UserPLMProfile, on_delete=models.CASCADE,null=True)
    user_reference_number = models.CharField(max_length=255, null=True, blank=True)
    user_email = models.CharField(max_length=100,null=True, blank=True)
    message = models.TextField(null=True,blank=True)
    response = models.TextField(null=True,blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user} - {self.timestamp}"

#####10-04-25One feed api
from uuid import uuid4
class ChatSession(models.Model):
    session_id = models.UUIDField(default=uuid4, editable=False, unique=True,null=True,blank=True)
    user_reference_number = models.CharField(max_length=255,null=True,blank=True)
    user_email = models.CharField(max_length=200,null=True,blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Session {self.session_id} - {self.user_email}"

class SmartResponse(models.Model):
    chat_session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, null=True, blank=True,
                                     related_name='responses')
    user_reference_number = models.CharField(max_length=200, null=True, blank=True)
    user_email = models.CharField(max_length=200, null=True, blank=True)
    text = models.TextField(null=True, blank=True)
    audio = models.TextField(null=True, blank=True)
    image_prompt = models.TextField(null=True, blank=True)
    video_prompt = models.TextField(null=True,blank=True)
    user_image_url = models.URLField(null=True, blank=True)

    text_response = models.TextField(null=True, blank=True)
    audio_response = models.TextField(null=True, blank=True)
    audio_transcript = models.TextField(null=True,blank=True)
    image_response = models.TextField(null=True, blank=True)
    video_response = models.TextField(null=True,blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Response at {self.created_at}"

###07-05-2025

class EnhancedSocialContent(models.Model):
    email = models.CharField(max_length=100,null=True,blank=True)
    reference_number = models.CharField(max_length=100,null=True,blank=True)
    media_url = models.TextField(null=True,blank=True)
    original_caption = models.TextField(null=True,blank=True)
    enhanced_caption = models.TextField(null=True,blank=True)
    category = models.CharField(max_length=100,null=True,blank=True)
    forwarded_at = models.DateTimeField(auto_now_add=True)

###EDIT IMAGEDB1505025
class GeminiImageEdit(models.Model):
    email = models.CharField(max_length=100, null=True, blank=True)
    reference_number = models.CharField(max_length=100, null=True, blank=True)
    image_prompt = models.TextField()
    original_image_name = models.CharField(max_length=255)
    generated_image_url = models.URLField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.email} - {self.reference_number}"

#####ACTIVITIES MODELS START##

from django.db import models


class SchedulingAssistantLog(models.Model):
    message = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)


class WellnessBotLog(models.Model):
    message = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

