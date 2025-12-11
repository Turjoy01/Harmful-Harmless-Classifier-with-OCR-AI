Halal Haram Ingredient Classifier with OCR and FastAPI

This project leverages machine learning, Optical Character Recognition (OCR), and FastAPI to classify ingredients in a product based on their harmless or harmful nature. The classification is based on a model trained on over 500,000 data points using an LSTM (Long Short-Term Memory) neural network, achieving 99.99% accuracy.

Overview

Machine Learning Model: The harmful/harmless classification is based on a dataset of ingredients, which has been trained using an LSTM model. The model provides accurate predictions on whether an ingredient is harmful or harmless.

OCR Integration: Using Google Cloud Vision API, this system can extract text from images (such as product labels or ingredient lists).

FastAPI: The application is built with FastAPI to provide a web service for ingredient classification.

Classification: Each ingredient is classified into harmful/harmless with a confidence score, and additional categorization is done for animal-based and alcohol-based ingredients.

Ingredients Detection: The model checks ingredients against predefined lists of animal-based and alcohol-based ingredients.

Key Features

OCR-based text extraction: Extracts ingredient lists from images using Google Vision API.

Harmful/Harmless Classification: Classifies ingredients into harmful and harmless categories using a trained LSTM model.

Confidence Scores: Provides a confidence score for each classification.

Animal-based and Alcohol-based Ingredient Detection: Identifies ingredients that are animal-based or contain alcohol, with the ability to list them separately.

FastAPI Interface: A RESTful API for integration and usage in web and mobile applications.

Dataset & Model

Dataset: A large dataset of ingredients was used to train the model, which includes over 500,000 records of harmful and harmless ingredients.

Model Architecture: The model is based on LSTM (Long Short-Term Memory), a type of Recurrent Neural Network (RNN) suitable for sequence prediction tasks.

Accuracy: The trained model achieves an accuracy of 99.99% in classifying harmful and harmless ingredients.

API Endpoints
1. Classify Ingredient Image: /classify (POST)

Classify ingredients from an image.

Request:

Body: Upload an image with ingredient text.

Response:

{
  "product_name": "Product Name",
  "ingredients_full_text": "Full extracted text",
  "ingredients_list": [
    "ingredient1",
    "ingredient2",
    ...
  ],
  "detected": {
    "has_animal_ingredients": true,
    "has_alcohol": false
  },
  "animal_ingredients": [
    {"name": "ingredient1", "harmful": true, "confidence": 0.98}
  ],
  "alcohol_ingredients": [
    {"name": "ingredient2", "harmful": false, "confidence": 0.95}
  ],
  "harmless_ingredients": [
    "ingredient3",
    "ingredient4"
  ]
}

2. Welcome Endpoint: / (GET)

Returns a simple welcome message.

Response:

{
  "message": "Halal Haram Ingredient Checker API"
}

Requirements

Python 3.8+

FastAPI

TensorFlow

Pillow

Google Cloud Vision API

Other dependencies listed in requirements.txt

Installation

Clone the repository:

git clone https://github.com/Turjoy01/Harmful-Harmless-Classifier-with-OCR-AI.git


Navigate to the project directory:

cd Harmful-Harmless-Classifier-with-OCR-AI


Install dependencies:

pip install -r requirements.txt


Set up Google Cloud Vision API credentials:

Create a Google Cloud account and set up the Vision API.

Download your service account credentials in JSON format and set the GOOGLE_APPLICATION_CREDENTIALS environment variable:

export GOOGLE_APPLICATION_CREDENTIALS="path/to/google-key.json"


Start the FastAPI server:

uvicorn main:app --reload


Visit http://localhost:8000 to interact with the API.

How It Works

Upload an Image: The /classify endpoint accepts an image with an ingredient list.

OCR Extraction: The image is processed using Google Cloud Vision API to extract text.

Text Extraction and Classification: Extracted text is analyzed to identify harmful and harmless ingredients.

Animal & Alcohol Detection: The model checks for ingredients that are animal-based or contain alcohol and categorizes them separately.

Confidence Scoring: The system provides confidence scores for the classification of each ingredient.

Example Usage

For instance, when a user uploads an image containing a list of ingredients, the API will extract the text, classify each ingredient as harmful or harmless, and identify any animal-based or alcohol-based ingredients. It will return the results with confidence scores.
