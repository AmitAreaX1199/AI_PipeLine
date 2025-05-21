from rest_framework import serializers
from django.contrib.auth.models import User
from .models import UserCredentials,AIContentDb,UserInput, AIResponse,Feedback,GeneratedImage,OpenaAI_UsageDB,VideoDB,SmartResponse,ChatSession,EnhancedSocialContent

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'password',
                  'first_name', 'last_name', 'email', 'is_staff']

    def validate_email(self, value):
        """
        Check if the email address is unique.
        """
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError('This email address is already in use.')
        return value

    def create(self, validated_data):
        user = User.objects.create(
            username=validated_data['username'],
            email=validated_data['email'],
            first_name=validated_data['first_name'],
            last_name=validated_data['last_name']
        )
        user.set_password(validated_data['password'])

        if 'is_staff' in validated_data:
            user.is_staff = validated_data['is_staff']

        user.save()
        return user

##Google serializer

class GooglePhotosCredentialsSerializer(serializers.Serializer):
    email = serializers.EmailField()

## Matching image with google photos
class ImageUploadSerializer(serializers.Serializer):
    email = serializers.EmailField()
    image = serializers.ImageField()


###Edit Caption Serializer

class Edit_Caption_Serializer(serializers.ModelSerializer):
    class Meta:
        model=AIContentDb
        fields="__all__"

##UserCred Serializers
class UserCredentialsSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserCredentials
        fields = '__all__'


### AI Sentiment Agent Serializers

class TextInputSerializer(serializers.Serializer):
    text = serializers.CharField(required=False, allow_blank=True)
    audio = serializers.FileField(required=False)

class UserInputSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserInput
        fields = ['id', 'text', 'tone', 'created_at']

class AIResponseSerializer(serializers.ModelSerializer):
    user_input = UserInputSerializer()

    class Meta:
        model = AIResponse
        fields = ['id', 'user_input',"user_reference_number","user_email", 'response_text', 'created_at']

###
class FeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = Feedback
        fields = '__all__'


class GeneratedImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = GeneratedImage
        fields = '__all__'
class OpenaAIUsageDBSerializer(serializers.ModelSerializer):
    class Meta:
        model = OpenaAI_UsageDB
        fields = '__all__'

class VideoGenerated_Serializer(serializers.ModelSerializer):
    class Meta:
        model = VideoDB
        fields = '__all__'

#####PLMSEARILIzers17FEB

class ChatRequestSerializer(serializers.Serializer):
    user_id = serializers.CharField(max_length=100)
    user_input = serializers.CharField()

class ChatResponseSerializer(serializers.Serializer):
    response = serializers.CharField()


class SmartResponseSerializer(serializers.ModelSerializer):
    class Meta:
        model = SmartResponse
        fields = [
            'id',
            'user_reference_number',
            'user_email',
            'text',
            'audio',
            'image_prompt',
            'user_image_url',
            'video_prompt',
            'text_response',
            'audio_response',
            'audio_transcript',
            'image_response',
            'video_response'
            'created_at',
        ]

##ALl session id serializers
class ChatSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatSession
        # fields = ['session_id',"user_reference_number","user_email"]
        fields = ['session_id']



class EnhancedSocialContentSerializer(serializers.ModelSerializer):
    class Meta:
        model = EnhancedSocialContent
        fields = '__all__'
        extra_kwargs = {
            'original_caption': {'required': False},
            'enhanced_caption': {'required': False},
        }



