Certainly! Below is a more **designed** and **structured** version of your `README.md` that you can use directly for your GitHub project. This version includes some formatting to make it look cleaner and more appealing.

````markdown
# Halal Haram Ingredient Classifier with OCR and FastAPI

This project leverages **machine learning**, **Optical Character Recognition (OCR)**, and **FastAPI** to classify ingredients in a product based on their **harmless** or **harmful** nature. The classification is based on a model trained on over **500,000** data points using an **LSTM** (Long Short-Term Memory) neural network, achieving **99.99% accuracy**.

![Halal Haram Classifier](https://example.com/image.png)  <!-- Replace with your project image -->

---

## 🛠️ Features

- **OCR-based text extraction**: Extracts ingredient lists from images using **Google Vision API**.
- **Harmful/Harmless Classification**: Classifies ingredients as harmful or harmless using a trained **LSTM model**.
- **Confidence Scores**: Provides confidence scores for each classification.
- **Animal-based and Alcohol-based Ingredient Detection**: Identifies animal-based or alcohol-based ingredients separately.
- **FastAPI Interface**: RESTful API for integrating with web and mobile applications.

---

## 🧠 Dataset & Model

- **Dataset**: A large dataset of ingredients used to train the model, with over 500,000 records.
- **Model Architecture**: The model is based on **LSTM** (Long Short-Term Memory), a type of **Recurrent Neural Network (RNN)** suitable for sequence-based tasks.
- **Accuracy**: Achieved **99.99% accuracy** in classifying harmful and harmless ingredients.

---

## 📜 API Endpoints

### 1. **Classify Ingredient Image**: `/classify` (POST)

Classify ingredients from an image.

#### Request:
- **Body**: Upload an image with ingredient text.

#### Response:
```json
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
````

### 2. **Welcome Endpoint**: `/` (GET)

Returns a simple welcome message.

#### Response:

```json
{
  "message": "Halal Haram Ingredient Checker API"
}
```

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Turjoy01/Harmful-Harmless-Classifier-with-OCR-AI.git
cd Harmful-Harmless-Classifier-with-OCR-AI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Google Cloud Vision API credentials:

* Set your **Google Cloud Vision API credentials** by exporting the path to the JSON file containing your credentials:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/google-key.json"
```

### 4. Run the FastAPI Server

```bash
uvicorn main:app --reload
```

Visit `http://localhost:8000` to interact with the API.

---

## 🚀 How It Works

1. **Upload an Image**: The `/classify` endpoint accepts an image containing the ingredient list.
2. **OCR Extraction**: The image is processed using the **Google Vision API** to extract text.
3. **Text Classification**: The extracted text is classified as harmful or harmless.
4. **Animal & Alcohol Detection**: Identifies ingredients based on predefined lists of animal-based and alcohol-based items.
5. **Confidence Scoring**: The system provides confidence scores for the classification of each ingredient.

---

## 🧑‍💻 Example Usage

Upload an image with an ingredient list, and the API will:

* Extract the text from the image using **OCR**.
* Classify each ingredient as harmful or harmless using the trained **LSTM model**.
* Separate ingredients into **animal-based** or **alcohol-based** categories.
* Return a response with **confidence scores** for each classification.

### Example Request

```bash
curl -X 'POST' \
  'http://localhost:8000/classify' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/image.jpg'
```

### Example Response

```json
{
  "product_name": "Sample Product",
  "ingredients_full_text": "Milk, cheese, and alcohol",
  "ingredients_list": ["milk", "cheese", "vodka"],
  "detected": {
    "has_animal_ingredients": true,
    "has_alcohol": true
  },
  "animal_ingredients": [
    {"name": "milk", "harmful": true, "confidence": 0.98}
  ],
  "alcohol_ingredients": [
    {"name": "vodka", "harmful": false, "confidence": 0.95}
  ],
  "harmless_ingredients": ["cheese"]
}
```

---

## 📝 License

This project is licensed under the MIT License.
