from openai import OpenAI
from dotenv import load_dotenv
import base64
import os
import openai

load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
openai.api_key=OPENAI_API_KEY

##Encoding Images
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"


def describe_image(image_path, prompt):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_path}},
            ]}
        ],
        max_tokens=256,
        temperature=0.0,
    )

    return response.choices[0].message.content

import google.generativeai as genai

# Function to generate responses using Gemini AI
def generate_response(prompt, personality_traits):
    try:
        # Fine-tune the prompt based on personality traits
        fine_tuned_prompt = f"You are a personal Replica called PLM, an advanced AI persona agent designed to mirror the user's tone, style, and sentiment. Engage in natural, complete sentences, responding as if you were their digital reflection, ensuring conversations feel seamless and personalized. {prompt} Reflect these traits: {personality_traits}."
        print(fine_tuned_prompt,"iiiiiprompt")
        # Generate response using Gemini
        model = genai.GenerativeModel("models/gemini-1.5-pro")
        response = model.generate_content(fine_tuned_prompt)

        print(response,"---------------ajahdaj")

        # Check if the response is blocked
        if response.candidates and response.candidates[0].finish_reason == "SAFETY":
            return "Sorry, I can't respond to that. Please try a different input."

        # Return the response text
        return response.text
    except Exception as e:
        # Handle errors (e.g., safety filters triggered)
        return f"Error generating response: {str(e)}"


# Function to get or create a user profile
def get_or_create_user_profile(user_id):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()

    cursor.execute("SELECT preferences, personality_traits FROM users WHERE user_id = ?", (user_id,))
    user_data = cursor.fetchone()

    if user_data:
        preferences, personality_traits = user_data
    else:
        preferences = "default_preferences"
        personality_traits = "friendly, curious, helpful"
        cursor.execute("INSERT INTO users (user_id, preferences, personality_traits) VALUES (?, ?, ?)",
                       (user_id, preferences, personality_traits))
        conn.commit()

    conn.close()
    return preferences, personality_traits


# Function to save chat history
def save_chat_history(user_id, user_input, response):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()

    cursor.execute("INSERT INTO chat_history (user_id, message, response) VALUES (?, ?, ?)",
                   (user_id, user_input, response))
    conn.commit()
    conn.close()





##Creation of DB


import sqlite3

def init_db():
    """Initialize SQLite database and create tables if not exist."""
    conn = sqlite3.connect('user_data.db')  # Ensure this matches your database path
    cursor = conn.cursor()

    # Drop existing tables if they exist (for debugging, remove in production)
    cursor.execute("DROP TABLE IF EXISTS users")
    cursor.execute("DROP TABLE IF EXISTS chat_history")

    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            preferences TEXT,
            personality_traits TEXT
        )
    ''')

    # Create chat_history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            message TEXT,
            response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')

    conn.commit()
    conn.close()