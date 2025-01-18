import streamlit as st
import google.generativeai as genai
import io
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the api keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")

# Function to generate a prompt from the google gemini API
@st.cache_data
def generate_gemini_prompt(prompt, api_key):
    """Generates a prompt using the google gemini API"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat()
    response = chat.send_message(f"Given the following text '{prompt}', generate a very detailed image generation prompt that can be used to generate an image, the response should only be the prompt and nothing else.")
    return response.text

# Function to generate an image from a text prompt
@st.cache_data
def generate_image(prompt, api_key, width=512, height=512):
    """Generates an image using the stability-sdk."""

    stability_api = client.StabilityInference(
        key = api_key,
        verbose = True,
        engine = "stable-diffusion-xl-1024-v1-0"
    )

    answers = stability_api.generate(
        prompt = prompt,
        height = height,
        width = width,
        samples = 1,
        steps = 30
    )

    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                st.error("Your request activated the API's safety filters and could not be processed. Please modify the prompt and try again.")
                return None
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                return img
    return None

def main():
    st.title("Text-to-Image Thumbnail Generator")
    st.write("Generate an image using Google Gemini 2.0 and Stability AI.")

    prompt = st.text_input("Enter a text prompt for the image")

    if st.button("Generate Image"):
        if not GEMINI_API_KEY:
           st.error("Google Gemini API Key is missing, Please check the .env file")
        elif not STABILITY_API_KEY:
            st.error("Stability API key is missing, Please check the .env file")
        elif not prompt:
             st.error("Please enter a prompt for the image")
        else:
            with st.spinner("Generating prompt using Gemini..."):
                  gemini_prompt = generate_gemini_prompt(prompt, GEMINI_API_KEY)
                  if gemini_prompt:
                     with st.spinner("Generating image with stability AI.."):
                            image = generate_image(gemini_prompt, STABILITY_API_KEY)

                            if image:
                                st.image(image, caption="Generated Image", use_column_width=True)
                            else:
                                 st.error("Failed to generate an image. Please check the error message.")
                  else:
                       st.error("Failed to generate a prompt using Gemini. Please check the error message.")
if __name__ == "__main__":
    main()
