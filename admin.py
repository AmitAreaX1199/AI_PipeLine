from django.contrib import admin
from .models import AIContentDb,UserInput,AIResponse,GeneratedImage,Feedback,OpenaAI_UsageDB,ImageGenerationSD_DB,ImageCaptionGeminiDB,VideoDB,UserLocation,UserPLMProfile,ChatHistory,SmartResponse,ChatSession,EnhancedSocialContent,GeminiImageEdit

# Register your models here.
admin.site.register(AIContentDb)
admin.site.register(UserInput)
admin.site.register(AIResponse)
admin.site.register(GeneratedImage)
admin.site.register(Feedback)
admin.site.register(OpenaAI_UsageDB)
admin.site.register(ImageGenerationSD_DB)
admin.site.register(ImageCaptionGeminiDB)
admin.site.register(VideoDB)
admin.site.register(UserLocation)
admin.site.register(UserPLMProfile)
admin.site.register(ChatHistory)
admin.site.register(SmartResponse)
admin.site.register(ChatSession)
admin.site.register(EnhancedSocialContent)
admin.site.register(GeminiImageEdit)


