from langchain.prompts import PromptTemplate

def get_prompt_template():
    template = """
    Your task is to analyze the given images.
    And generate attractive caption about the given images for social-media platform.
    And Caption should be in complete sentence with hashtags. 
    Important Note: Do not give descriptions out of context and Do not give emojis.
    """
    return PromptTemplate.from_template(template)