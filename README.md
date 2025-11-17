# Harmful / Harmless Text Classifier with OCR (Bangla + English)

Real-time image → text → harmful content detection system  
Built for content moderation, social media safety, and online platform compliance in Bangladesh & South Asia.

### Features
- OCR using Google Cloud Vision (supports Bangla + English handwritten/printed text)
- Custom-trained Deep Learning classifier (94%+ accuracy on harmful vs harmless)
- FastAPI backend + live Swagger UI
- Accepts image upload → returns classification + confidence score

### Live Demo
Running at: http://127.0.0.1:8000/docs (after uvicorn)

### Quick Start
```bash
pip install -r requirements.txt
uvicorn main:app --reload
