import os
import pickle
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import tensorflow as tf
from google.cloud import vision
import re

# ==================== CONFIG ====================
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-key.json"

# ==================== LOAD MODEL ====================
with open("best_harmful_classifier.pkl", "rb") as f:
    harmful_classifier = pickle.load(f)

tokenizer = harmful_classifier['tokenizer']
max_len = harmful_classifier['max_len']
model_architecture = harmful_classifier['model_architecture']
model_weights = harmful_classifier['model_weights']
label_mapping = harmful_classifier['label_mapping']

model = tf.keras.models.model_from_json(model_architecture)
model.set_weights(model_weights)

# ==================== HARMFUL LISTS ====================
ANIMAL_KEYWORDS = [
    # === DAIRY & MILK ===
    "milk", "cow milk", "goat milk", "sheep milk", "buffalo milk", "camel milk",
    "whey", "whey powder", "deproteinized whey", "whey protein", "whey concentrate",
    "lactose", "lactalbumin", "lactoglobulin", "lacto", "casein", "caseinate",
    "sodium caseinate", "calcium caseinate", "potassium caseinate", "ammonium caseinate",
    "milk powder", "skim milk", "whole milk", "skimmed milk", "nonfat milk", "milk fat",
    "butter", "butterfat", "butteroil", "ghee", "clarified butter", "anhydrous milk fat",
    "cream", "heavy cream", "whipping cream", "sour cream", "crème fraîche",
    "cheese", "cheddar", "mozzarella", "parmesan", "gouda", "emmental", "camembert",
    "yogurt", "yoghurt", "greek yogurt", "curd", "kefir", "buttermilk", "acidophilus milk",

    # === ANIMAL FATS & OILS ===
    "lard", "pork fat", "bacon fat", "tallow", "beef fat", "suet", "dripping",
    "animal fat", "animal shortening", "chicken fat", "duck fat", "goose fat",
    "omega-3 from fish", "fish oil", "cod liver oil", "shark liver oil", "marine oil",

    # === GELATIN & COLLAGEN ===
    "gelatin", "gelatine", "bovine gelatin", "porcine gelatin", "pork gelatin",
    "fish gelatin", "marine gelatin", "hydrolyzed gelatin", "collagen", "hydrolyzed collagen",
    "bovine collagen", "porcine collagen", "marine collagen", "gelatin hydrolysate",

    # === ENZYMES ===
    "rennet", "animal rennet", "calf rennet", "pepsin", "pepsine", "chymosin",
    "microbial rennet", "fermentation produced chymosin", "fpc", "lipase", "animal lipase",
    "pancreatin", "trypsin", "protease", "catalyst enzyme", "enzyme from animal",

    # === COLORS ===
    "carmine", "cochineal", "carminic acid", "e120", "natural red 4", "crimson lake",
    "shellac", "lac resin", "confectioners glaze", "resinous glaze", "e904", "lacquer",

    # === AMINO ACIDS & PROTEINS ===
    "l-cysteine", "l-cystine", "cysteine", "keratin", "hydrolyzed keratin",
    "albumin", "egg albumin", "ovalbumin", "serum albumin", "blood albumin",
    "isinglass", "fish bladder", "bone phosphate", "e542", "calcium phosphate from bone",

    # === EMULSIFIERS & STABILIZERS (Suspicious/Animal Possible) ===
    "e322", "lecithin", "soy lecithin", "e471", "mono and diglycerides", "monoglycerides",
    "diglycerides", "glycerides", "glycerol", "glycerine", "glycerin", "glyceryl",
    "e472a", "e472b", "e472c", "e472e", "e472f", "datem", "e473", "e470", "e477",
    "e481", "e482", "e483", "stearoyl", "sodium stearoyl", "calcium stearoyl",
    "magnesium stearate", "stearic acid", "palmitic acid", "e570", "fatty acids",

    # === E-NUMBERS (Animal Origin Possible) ===
    "e101", "e101a", "e120", "e153", "e161g", "e252", "e270", "e304", "e322",
    "e325", "e326", "e327", "e422", "e430–436", "e441", "e470a", "e470b", "e471",
    "e472a", "e472b", "e472c", "e472e", "e472f", "e473", "e474", "e475", "e476",
    "e477", "e479b", "e481", "e482", "e483", "e491–495", "e542", "e570", "e572",
    "e627", "e631", "e635", "e640", "e904", "e913", "e920", "e921", "e966", "e1105",

    # === MEAT & MEAT PRODUCTS ===
    "beef", "pork", "bacon", "ham", "sausage", "salami", "pepperoni", "chorizo",
    "chicken", "turkey", "duck", "goose", "lamb", "mutton", "veal", "venison",
    "gelatin from beef", "gelatin from pork", "broth", "stock", "bone broth",

    # === HONEY & BEE PRODUCTS ===
    "honey", "raw honey", "manuka honey", "royal jelly", "propolis", "bee pollen",
    "beeswax", "e901", "honeycomb",

    # === OTHER ANIMAL SOURCES ===
    "lanolin", "vitamin d3", "cholecalciferol", "vitamin a from fish", "squalene",
    "squalane", "ambergris", "civet", "castoreum", "musk", "collagen peptide",
    "elastin", "silk powder", "pearl powder", "conchiolin", "snail secretion",

    # === HARAM ANIMALS & PARTS ===
    "pig", "boar", "swine", "porcine", "hog", "carrion", "blood", "plasma", "serum"
]

ALCOHOL_KEYWORDS = [
    # === DIRECT ALCOHOL NAMES ===
    "alcohol", "ethanol", "ethyl alcohol", "grain alcohol", "rectified spirit",
    "denatured alcohol", "alcohol denat", "sd alcohol", "alcohol 40", "alcohol 40-b",
    "sd alcohol 40", "alcohol denatured", "spirit", "spirits", "liquor",

    # === WINES & FERMENTED GRAPE ===
    "wine", "red wine", "white wine", "rose wine", "rosé", "sparkling wine",
    "champagne", "prosecco", "cava", "lambrusco", "asti", "moscato", "riesling",
    "pinot grigio", "sauvignon blanc", "chardonnay", "cabernet", "merlot",
    "sherry", "port wine", "madeira", "marsala", "vermouth", "vermouth rosso",
    "dubonnet", "lillet", "martini", "campari", "aperol",

    # === BEERS & MALT ===
    "beer", "lager", "pilsner", "ale", "stout", "porter", "ipa", "pale ale",
    "bitter", "mild", "wheat beer", "hefeweizen", "bock", "doppelbock", "oktoberfest",
    "craft beer", "malt liquor", "malt", "malted barley", "brewers yeast",

    # === CIDERS & FRUIT FERMENTED ===
    "cider", "hard cider", "apple cider alcohol", "pear cider", "perry",

    # === DISTILLED SPIRITS ===
    "vodka", "gin", "rum", "white rum", "dark rum", "spiced rum", "tequila",
    "mezcal", "whisky", "whiskey", "scotch", "bourbon", "rye whiskey", "irish whiskey",
    "canadian whisky", "tennessee whiskey", "brandy", "cognac", "armagnac",
    "calvados", "grappa", "pisco", "eau de vie", "kirsch", "slivovitz", "rakija",

    # === LIQUEURS & CREAM LIQUEURS ===
    "baileys", "kahlua", "tia maria", "amaretto", "disaronno", "frangelico",
    "grand marnier", "cointreau", "triple sec", "curacao", "drambuie", "benedictine",
    "chartreuse", "absinthe", "pastis", "ouzo", "raki", "arak", "arrack", "sambuca",
    "galliano", "jägermeister", "fernet branca", "campari", "aperol", "limoncello",

    # === ASIAN ALCOHOLS ===
    "sake", "nihonshu", "soju", "shochu", "baijiu", "maotai", "huangjiu", "shaoxing wine",
    "mirin", "cooking sake", "cooking wine", "rice wine", "plum wine", "makgeolli",

    # === EXTRACTS & FLAVORS WITH ALCOHOL ===
    "vanilla extract", "almond extract", "lemon extract", "orange extract",
    "peppermint extract", "rose extract", "orange blossom", "flavor extract",
    "natural flavor with alcohol", "artificial flavor with alcohol", "spirit flavor",
    "bourbon vanilla", "rum flavor", "brandy flavor", "whisky flavor", "liqueur flavor",

    # === VINEGARS & FERMENTED SAUCES ===
    "wine vinegar", "red wine vinegar", "white wine vinegar", "sherry vinegar",
    "balsamic vinegar", "malt vinegar", "spirit vinegar", "alcohol vinegar",
    "rice vinegar alcohol", "fermented vinegar",

    # === CHEMICAL NAMES & E-NUMBERS ===
    "e1510", "ethanol e1510", "ethyl acetate", "ethyl butyrate", "ethyl vanillin",
    "ethyl maltol", "ethyl lactate", "fusel oil", "isoamyl alcohol", "benzyl alcohol",

    # === HIDDEN IN "NATURAL FLAVOR" ===
    "natural flavor", "natural flavour", "flavouring", "aroma", "aroma compound",
    "flavoring substances", "spirit-based flavor", "alcohol-based flavor",

    # === ARABIC, URDU, MALAY, INDONESIAN, TURKISH, PERSIAN ===
    "khamr", "خمر", "الكحول", "إيثانول", "نبيذ", "بيرة", "شراب", "مسكر",
    "arak", "arrack", "toddy", "tuak", "brem", "tapai", "bir", "bira", "şarap",
    "wijn", "vin", "vino", "alcool", "alkohol", "alkoholi", "alkohol", "спирт",

    # === FOODS THAT MAY CONTAIN HIDDEN ALCOHOL ===
    "fermented", "fermentation", "yeast extract", "autolyzed yeast", "torula yeast",
    "soy sauce alcohol", "miso alcohol", "kombucha", "kefir alcohol", "kvass",
    "booze", "boozy", "infused with alcohol", "alcohol infused", "contains alcohol",
    "may contain alcohol", "trace alcohol", "ethyl alcohol added", "alcohol as solvent",

    # === COSMETICS & PHARMA ===
    "cetearyl alcohol", "cetyl alcohol", "stearyl alcohol", "benzyl alcohol",
    "phenethyl alcohol", "alcohol in flavor carrier", "ethanol as carrier",

    # === MISCELLANEOUS ===
    "distillate", "distilled", "spirit distillate", "alcohol distillate",
    "cooking alcohol", "culinary alcohol", "beverage alcohol", "potable alcohol",
    "ethyl alcohol of agricultural origin", "neutral spirit", "extra neutral alcohol"
]

# ==================== OCR & TEXT EXTRACT ====================
def ocr_google(image_bytes: bytes) -> str:
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    resp = client.text_detection(image=image)
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    return resp.text_annotations[0].description if resp.text_annotations else ""

def extract_text(full_ocr: str):
    lines = [ln.strip() for ln in full_ocr.splitlines() if ln.strip()]
    if not lines:
        return "UNKNOWN", "", []

    product_name = lines[0]
    ingredients_lines = []
    capture = False

    for i, line in enumerate(lines):
        if "INGREDIENTS" in line.upper():
            capture = True
            remainder = line.split("INGREDIENTS:", 1)[-1].strip()
            if remainder:
                ingredients_lines.append(remainder)
            start_idx = i + 1
            break
    else:
        start_idx = 1

    if not capture:
        ingredients_lines = lines[start_idx:start_idx + 6]
    else:
        for line in lines[start_idx:start_idx + 7]:
            up = line.upper()
            if any(x in up for x in ["NUTRITION", "ALLERGEN", "DIRECTIONS", "NET WT", "DISTRIBUTED", "MANUFACTURED"]):
                break
            ingredients_lines.append(line)

    raw_text = " ".join(ingredients_lines)
    cleaned = " ".join(raw_text.lower().split())
    
    # Extract individual ingredients (split by comma, and, ; etc.)
    ingredients_raw = re.split(r',|\sand\s|\s+or\s+|\s*;\s*|\n', raw_text)
    ingredients_list = []
    for ing in ingredients_raw:
        ing = ing.strip().strip('().').lower()
        if ing and len(ing) > 1:
            ingredients_list.append(ing)

    return product_name, cleaned, list(set(ingredients_list))  # unique

# ==================== CLASSIFY SINGLE INGREDIENT ====================
def classify_single_ingredient(text: str):
    if not text.strip():
        return "harmless", 0.0
    tokenized = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(tokenized, maxlen=max_len)
    probs = model.predict(padded, verbose=0)[0]
    idx = np.argmax(probs)
    pred = label_mapping[idx]
    conf = float(probs[idx])
    return pred, round(conf, 4)

# ==================== FASTAPI APP ====================
app = FastAPI()

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        Image.open(BytesIO(contents)).convert("RGB")
    except:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    full_ocr = ocr_google(contents)
    if not full_ocr.strip():
        return JSONResponse(status_code=400, content={"error": "No text detected"})

    product_name, full_text, ingredients_list = extract_text(full_ocr)

    # Detect animal & alcohol
    animal_found = []
    alcohol_found = []
    harmless = []

    for ing in ingredients_list:
        ing_lower = ing.lower()
        is_animal = any(k in ing_lower for k in ANIMAL_KEYWORDS)
        is_alcohol = any(k in ing_lower for k in ALCOHOL_KEYWORDS)

        pred, conf = classify_single_ingredient(ing)

        item = {"name": ing, "harmful": pred != "harmless", "confidence": conf}

        if is_alcohol:
            alcohol_found.append(item)
        elif is_animal:
            animal_found.append(item)
        else:
            if pred == "harmless":
                harmless.append(ing)

    response = {
        "product_name": product_name,
        "ingredients_full_text": full_text,
        "ingredients_list": ingredients_list,
        "detected": {
            "has_animal_ingredients": len(animal_found) > 0,
            "has_alcohol": len(alcohol_found) > 0
        },
        "animal_ingredients": animal_found,
        "alcohol_ingredients": alcohol_found,
        "harmless_ingredients": harmless
    }

    return response

@app.get("/")
async def root():
    return {"message": "Halal Haram Ingredient Checker API"}