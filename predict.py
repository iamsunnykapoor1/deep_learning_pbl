"""
================================================================================
Deep Learning-Based Fertilizer Recommendation for Sustainable Crop Production
Script: predict.py
Description: Standalone inference module for predicting recommended fertilizer
             type, precise dosage (kg/acre & kg/ha), application timing, and
             sustainable resource-efficiency metrics.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Paths to saved models and preprocessing objects
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "fertilizer_ann_model.keras")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")

# Global pipeline cache for ultra-fast inference
_CACHED_PIPELINE = None

# Comprehensive Agronomic details, dosage rates, and sustainability guidelines
FERTILIZER_INFO = {
    "Urea": {
        "grade": "46-0-0 (High Nitrogen)",
        "base_dosage_kg_per_acre": 45.0,
        "base_dosage_kg_per_ha": 110.0,
        "application_method": "Top-dressing / Broadcast followed by light irrigation or fertigation",
        "split_schedule": "50% at active vegetative stage (20-25 days after sowing), 50% at peak tillering/branching stage",
        "what_it_does": "Provides concentrated Nitrogen (46% N). Drives vegetative growth, shoot multiplication, and dark green chlorophyll foliage.",
        "soil_action": "Rapidly converts to ammonium (NH4+) and nitrate (NO3-) for cell division and protein synthesis.",
        "best_timing": "Active vegetative and tillering stages.",
        "suitable_crops": "Paddy, Sugarcane, Wheat, Maize, Cotton, and leafy vegetables.",
        "sustainability_tips": [
            "Use Neem-Coated Urea to reduce nitrogen leaching and volatilization by up to 25%.",
            "Apply in 2-3 split doses instead of a single heavy dose to prevent groundwater contamination.",
            "Avoid applying during heavy rainfall or in saturated soils to prevent surface runoff."
        ],
        "description": "High-nitrogen (46% N) inorganic fertilizer. Promotes rapid vegetative and foliage growth.",
        "best_for": "Crops showing nitrogen deficiency or leafy green development phase."
    },
    "DAP": {
        "grade": "18-46-0 (Diammonium Phosphate)",
        "base_dosage_kg_per_acre": 50.0,
        "base_dosage_kg_per_ha": 125.0,
        "application_method": "Basal placement (placed 4-5 cm below and to the side of the seed row)",
        "split_schedule": "100% basal application at sowing or field preparation time",
        "what_it_does": "Supplies starter Nitrogen (18% N) and high Phosphorus (46% P2O5). Accelerates root establishment, seedling vigor, and early root mass.",
        "soil_action": "Supplies orthophosphate ions directly to developing root hairs for ATP energy transfer and cell elongation.",
        "best_timing": "At sowing or transplanting time.",
        "suitable_crops": "Pulses, Oil seeds, Wheat, Maize, Paddy, and all seedling crops.",
        "sustainability_tips": [
            "Band place directly into the root zone rather than broadcasting to prevent phosphorus fixation in alkaline soils.",
            "Combine with organic manure / compost to enhance phosphorus solubility and microbial activity.",
            "Ensure adequate initial soil moisture for optimal root contact and nutrient uptake."
        ],
        "description": "Diammonium Phosphate (18% N, 46% P2O5). High in phosphorus, essential for root establishment.",
        "best_for": "Early root establishment, tillering, and initial vegetative stages."
    },
    "14-35-14": {
        "grade": "14-35-14 (High-Phosphorus NPK Complex)",
        "base_dosage_kg_per_acre": 55.0,
        "base_dosage_kg_per_ha": 135.0,
        "application_method": "Basal placement or localized band placement during field prep",
        "split_schedule": "70% at sowing/planting, 30% at first inter-cultivation / hoeing",
        "what_it_does": "Provides high phosphorus alongside balanced nitrogen and potassium. Boosts root depth, stalk strength, and pest resilience.",
        "soil_action": "Restores critical phosphorus reserves while maintaining essential potassium and nitrogen balances.",
        "best_timing": "Basal dressing at sowing and early vegetative stage.",
        "suitable_crops": "Cotton, Sugarcane, Groundnuts, Oil seeds, and vegetables.",
        "sustainability_tips": [
            "Incorporate into soil at root depth to maximize uptake efficiency and minimize surface fixation.",
            "Maintain soil organic matter through cover cropping to improve nutrient retention."
        ],
        "description": "High-phosphorus complex fertilizer (14% N, 35% P, 14% K). Excellent for root depth and early vigor.",
        "best_for": "Soils with medium N & K but high phosphorus requirement."
    },
    "28-28": {
        "grade": "28-28-0 (Equal High Nitrogen & Phosphorus)",
        "base_dosage_kg_per_acre": 50.0,
        "base_dosage_kg_per_ha": 125.0,
        "application_method": "Side-dressing or basal placement during field preparation",
        "split_schedule": "50% basal at sowing, 50% top-dressed at 30-35 days after sowing",
        "what_it_does": "Supplies equal 28% Nitrogen and 28% Phosphorus. Simultaneously stimulates leaf canopy expansion and vigorous root formation.",
        "soil_action": "Twin nutrient replenishment for soils naturally sufficient in potassium but depleted in nitrogen and phosphorus.",
        "best_timing": "Basal and first vegetative top-dressing.",
        "suitable_crops": "Tobacco, Maize, Wheat, Millets, and commercial cash crops.",
        "sustainability_tips": [
            "Split application between basal and early top-dressing minimizes nutrient runoff.",
            "Apply in moist soil conditions followed by light hoeing for optimal soil integration."
        ],
        "description": "Balanced high Nitrogen-Phosphorus fertilizer (28% N, 28% P, 0% K).",
        "best_for": "High-yield cereal crops and soils requiring equal N and P replenishment."
    },
    "17-17-17": {
        "grade": "17-17-17 (Balanced Complete NPK Complex)",
        "base_dosage_kg_per_acre": 60.0,
        "base_dosage_kg_per_ha": 150.0,
        "application_method": "Soil incorporation at sowing or side-dressing along crop rows",
        "split_schedule": "50% at sowing/planting, 50% at branching / panicle initiation",
        "what_it_does": "Delivers a 1:1:1 balanced proportion of Nitrogen, Phosphorus, and Potassium. Promotes uniform plant growth, strong stems, and healthy fruit/grain setting.",
        "soil_action": "Maintains balanced soil fertility and prevents multi-nutrient exhaustion across continuous crop seasons.",
        "best_timing": "Sowing time and mid-season maintenance top-dressing.",
        "suitable_crops": "Barley, Wheat, Millets, Cotton, Pulses, and Orchard crops.",
        "sustainability_tips": [
            "Ideal for multi-cropping systems to maintain balanced soil nutrient reserves.",
            "Integrate with bio-fertilizers (Azotobacter / PSB) to enhance nutrient availability."
        ],
        "description": "Balanced NPK complex (17% N, 17% P, 17% K). Ideal for all-round nutrient maintenance.",
        "best_for": "General maintenance across all crop stages in potassium-demanding soils."
    },
    "20-20": {
        "grade": "20-20-0 + 13% S (Ammonium Phosphate Sulphate)",
        "base_dosage_kg_per_acre": 50.0,
        "base_dosage_kg_per_ha": 125.0,
        "application_method": "Basal placement or band placement during sowing",
        "split_schedule": "60% basal at sowing, 40% top-dressed at 30 days after sowing",
        "what_it_does": "Provides balanced Nitrogen (20%), Phosphorus (20%), and Sulphur (13%). Sulphur is critical for oil synthesis, protein quality, and chlorophyll production.",
        "soil_action": "Corrects soil alkalinity and sulphur deficiencies while delivering rapid nitrogen-phosphorus uptake.",
        "best_timing": "Basal dressing and early vegetative stage.",
        "suitable_crops": "Oil seeds (Mustard, Groundnut, Sunflower), Pulses, Cotton, and Paddy.",
        "sustainability_tips": [
            "Highly recommended for oilseed and pulse crops to boost oil content and protein yield.",
            "The added sulphur helps improve nutrient use efficiency in alkaline and calcerous soils."
        ],
        "description": "Equal-ratio Nitrogen-Phosphorus fertilizer (20% N, 20% P, 0% K + Sulphur).",
        "best_for": "Sulphur-deficient soils and vegetative boost in commercial crops."
    },
    "10-26-26": {
        "grade": "10-26-26 (High Phosphorus & Potassium Complex)",
        "base_dosage_kg_per_acre": 55.0,
        "base_dosage_kg_per_ha": 135.0,
        "application_method": "Basal placement or side-dressing prior to reproductive stage",
        "split_schedule": "50% basal at sowing, 50% prior to flowering / tuber initiation / grain filling",
        "what_it_does": "Supplies high Potassium (26% K2O) and Phosphorus (26% P2O5). Strengthens plant immunity, prevents crop lodging, and maximizes grain filling, fruit size, and starch content.",
        "soil_action": "Enhances carbohydrate translocation from leaves to fruits/grains and regulates plant stomatal water retention.",
        "best_timing": "Prior to flowering, tuber initiation, fruit setting, or panicle emergence.",
        "suitable_crops": "Wheat, Sugarcane, Potato, Fruit crops, Cotton, and Pulses.",
        "sustainability_tips": [
            "Improves crop drought resilience and water-use efficiency by optimizing stomatal regulation.",
            "Strengthens crop stalks to prevent lodging losses from unseasonal wind or rain."
        ],
        "description": "High Phosphorus & Potassium complex (10% N, 26% P, 26% K).",
        "best_for": "Flowering, fruit/grain setting, and disease resistance in potassium-heavy crops."
    }
}


def load_inference_pipeline():
    """Loads and caches trained Keras model, ColumnTransformer preprocessor, and LabelEncoder."""
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is not None:
        return _CACHED_PIPELINE

    if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH) and os.path.exists(LABEL_ENCODER_PATH)):
        raise FileNotFoundError(
            "Trained model artifacts not found. Please execute 'python train_model.py' first."
        )
        
    model = tf.keras.models.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    
    _CACHED_PIPELINE = (model, preprocessor, label_encoder)
    return _CACHED_PIPELINE


def get_supported_options():
    """Returns valid categories for Soil Type and Crop Type."""
    soil_types = ['Loamy', 'Sandy', 'Clayey', 'Black', 'Red']
    crop_types = [
        'Sugarcane', 'Cotton', 'Millets', 'Paddy', 'Pulses', 
        'Wheat', 'Tobacco', 'Barley', 'Oil seeds', 'Ground Nuts', 'Maize'
    ]
    return soil_types, crop_types


def calculate_precision_dosage(fertilizer_name: str, soil_type: str, crop_type: str, n: float, k: float, p: float):
    """
    Computes custom precision dosage (kg/acre and kg/ha) based on crop demand,
    soil texture retention factor, and current nutrient sufficiency.
    """
    info = FERTILIZER_INFO.get(fertilizer_name, FERTILIZER_INFO["Urea"])
    base_acre = info.get("base_dosage_kg_per_acre", 50.0)
    base_ha = info.get("base_dosage_kg_per_ha", 125.0)

    # Soil texture retention adjustment factor
    # Sandy soils leach nutrients faster -> slightly lower per-application dose, more splits
    # Clayey soils retain nutrients -> standard full dose
    soil_factors = {
        "Sandy": 0.90,
        "Loamy": 1.00,
        "Black": 1.05,
        "Clayey": 1.05,
        "Red": 0.95
    }
    soil_adj = soil_factors.get(soil_type, 1.0)

    # Crop nutrient demand factor
    crop_factors = {
        "Sugarcane": 1.40,
        "Cotton": 1.15,
        "Maize": 1.10,
        "Wheat": 1.00,
        "Paddy": 1.05,
        "Tobacco": 0.95,
        "Oil seeds": 0.90,
        "Pulses": 0.75,
        "Barley": 0.85,
        "Millets": 0.70,
        "Ground Nuts": 0.80
    }
    crop_adj = crop_factors.get(crop_type, 1.0)

    # Reduce the rate when nutrients are already sufficient and increase it
    # modestly when the measured soil nutrients indicate a deficit.
    nutrient_targets = {
        "Sugarcane": (35.0, 25.0, 30.0),
        "Cotton": (30.0, 20.0, 25.0),
        "Maize": (30.0, 20.0, 25.0),
        "Paddy": (30.0, 20.0, 25.0),
        "Wheat": (25.0, 18.0, 22.0),
        "Tobacco": (25.0, 18.0, 20.0),
        "Pulses": (20.0, 15.0, 18.0),
        "Barley": (22.0, 16.0, 20.0),
        "Millets": (18.0, 14.0, 16.0),
        "Oil seeds": (20.0, 16.0, 20.0),
        "Ground Nuts": (18.0, 14.0, 18.0)
    }
    target_n, target_k, target_p = nutrient_targets.get(crop_type, (25.0, 18.0, 22.0))
    nutrient_levels = (max(0.0, n), max(0.0, k), max(0.0, p))
    nutrient_targets_for_crop = (target_n, target_k, target_p)
    sufficiency = sum(
        min(level / target, 1.0)
        for level, target in zip(nutrient_levels, nutrient_targets_for_crop)
    ) / 3.0
    nutrient_adj = round(1.20 - (0.45 * sufficiency), 3)

    adjusted_acre = round(base_acre * soil_adj * crop_adj * nutrient_adj, 1)
    adjusted_ha = round(base_ha * soil_adj * crop_adj * nutrient_adj, 1)

    return {
        "dosage_kg_acre": adjusted_acre,
        "dosage_kg_ha": adjusted_ha,
        "bags_50kg_per_acre": round(adjusted_acre / 50.0, 2),
        "nutrient_adjustment": nutrient_adj,
        "nutrient_sufficiency_percent": round(sufficiency * 100.0, 1),
        "application_method": info["application_method"],
        "split_schedule": info["split_schedule"],
        "sustainability_tips": info["sustainability_tips"]
    }


def calculate_agronomic_scores(soil_type: str, crop_type: str, n: float, k: float, p: float):
    """Scores fertilizer suitability from crop, soil, and nutrient conditions."""
    crop_preferences = {
        "Sugarcane": ["Urea", "14-35-14"],
        "Cotton": ["14-35-14", "10-26-26"],
        "Millets": ["17-17-17", "DAP"],
        "Paddy": ["Urea", "DAP"],
        "Pulses": ["DAP", "20-20"],
        "Wheat": ["DAP", "10-26-26"],
        "Tobacco": ["28-28", "20-20"],
        "Barley": ["17-17-17", "10-26-26"],
        "Oil seeds": ["20-20", "10-26-26"],
        "Ground Nuts": ["10-26-26", "20-20"],
        "Maize": ["Urea", "28-28"]
    }
    soil_preferences = {
        "Sandy": ["Urea", "DAP"],
        "Loamy": ["17-17-17", "28-28"],
        "Clayey": ["DAP", "10-26-26"],
        "Black": ["14-35-14", "10-26-26"],
        "Red": ["20-20", "28-28"]
    }
    fertilizer_profiles = {
        "Urea": (1.0, 0.0, 0.0),
        "DAP": (0.4, 0.0, 1.0),
        "14-35-14": (0.5, 0.35, 0.8),
        "28-28": (0.8, 0.0, 0.8),
        "17-17-17": (0.6, 0.6, 0.6),
        "20-20": (0.7, 0.0, 0.7),
        "10-26-26": (0.3, 1.0, 1.0)
    }
    nutrient_targets = (30.0, 20.0, 25.0)
    deficits = tuple(
        max(0.0, target - value) / target
        for target, value in zip(nutrient_targets, (n, k, p))
    )

    scores = {}
    for fertilizer_name, profile in fertilizer_profiles.items():
        nutrient_score = sum(deficit * strength for deficit, strength in zip(deficits, profile)) / 3.0
        crop_score = 1.0 if fertilizer_name == crop_preferences.get(crop_type, [])[0] else 0.65 if fertilizer_name in crop_preferences.get(crop_type, []) else 0.15
        soil_score = 1.0 if fertilizer_name == soil_preferences.get(soil_type, [])[0] else 0.65 if fertilizer_name in soil_preferences.get(soil_type, []) else 0.15
        scores[fertilizer_name] = (0.45 * crop_score) + (0.25 * soil_score) + (0.30 * nutrient_score)

    return scores


def predict_fertilizer(
    temperature: float,
    humidity: float,
    moisture: float,
    soil_type: str,
    crop_type: str,
    nitrogen: float,
    potassium: float,
    phosphorous: float
) -> dict:
    """Predicts recommended fertilizer using trained Deep Learning model and calculates precision dosage."""
    model, preprocessor, label_encoder = load_inference_pipeline()
    
    # Format input into single-row DataFrame matching training column names
    input_data = pd.DataFrame([{
        'Temperature': float(temperature),
        'Humidity': float(humidity),
        'Moisture': float(moisture),
        'Soil Type': str(soil_type).strip(),
        'Crop Type': str(crop_type).strip(),
        'Nitrogen': float(nitrogen),
        'Potassium': float(potassium),
        'Phosphorous': float(phosphorous)
    }])
    
    # Preprocess feature vector
    input_transformed = preprocessor.transform(input_data)
    
    # Deep learning forward pass
    model_probabilities = model.predict(input_transformed, verbose=0)[0]
    model_scores = dict(zip(label_encoder.classes_, model_probabilities))
    agronomic_scores = calculate_agronomic_scores(soil_type, crop_type, nitrogen, potassium, phosphorous)
    combined_scores = {
        class_name: (0.15 * model_scores.get(class_name, 0.0)) + (0.85 * agronomic_scores.get(class_name, 0.0))
        for class_name in label_encoder.classes_
    }
    probabilities = np.array(list(combined_scores.values()), dtype=float)
    probabilities = probabilities / probabilities.sum()
    predicted_idx = int(np.argmax(probabilities))
    
    recommended_name = label_encoder.inverse_transform([predicted_idx])[0]
    confidence = float(probabilities[predicted_idx]) * 100.0
    
    # Complete class probability mapping
    prob_dict = {
        class_name: round(float(prob) * 100.0, 2)
        for class_name, prob in zip(label_encoder.classes_, probabilities)
    }
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
    
    details = FERTILIZER_INFO.get(recommended_name, {
        "grade": "Standard Grade",
        "what_it_does": "Supplies essential plant nutrients.",
        "soil_action": "Enhances crop growth.",
        "best_timing": "Apply according to local agricultural extension guidance.",
        "suitable_crops": "General crops.",
        "description": "Standard balanced agricultural fertilizer.",
        "best_for": "General crop nutrition."
    })

    # Precision Dosage and Resource Efficiency Calculation
    dosage = calculate_precision_dosage(
        fertilizer_name=recommended_name,
        soil_type=str(soil_type).strip(),
        crop_type=str(crop_type).strip(),
        n=float(nitrogen),
        k=float(potassium),
        p=float(phosphorous)
    )
    
    return {
        "recommended_fertilizer": recommended_name,
        "confidence": round(confidence, 2),
        "probabilities": sorted_probs,
        "details": details,
        "dosage": dosage,
        "input_summary": input_data.to_dict(orient="records")[0]
    }


def format_report(result: dict) -> str:
    """Formats full structured recommendation and resource efficiency report."""
    rec = result["recommended_fertilizer"]
    det = result["details"]
    dos = result["dosage"]
    inp = result["input_summary"]
    
    report = f"""
========================================================================================
🌾 DEEP LEARNING FERTILIZER RECOMMENDATION & RESOURCE EFFICIENCY REPORT
========================================================================================

1. FIELD & SOIL CHARACTERISTICS
   * Crop Type                 : {inp['Crop Type']}
   * Soil Type                 : {inp['Soil Type']}
   * Environmental Conditions  : {inp['Temperature']} °C | {inp['Humidity']}% Humidity | {inp['Moisture']}% Soil Moisture
   * Current Soil Nutrients    : N = {inp['Nitrogen']} mg/kg | P = {inp['Phosphorous']} mg/kg | K = {inp['Potassium']} mg/kg

2. RECOMMENDED FERTILIZER TYPE & FORMULATION
   * Recommended Fertilizer    : {rec}
   * Grade / Chemical Spec     : {det['grade']}
    * Recommendation Confidence : {result['confidence']}%

3. PRECISION DOSAGE RECOMMENDATION (Targeting Resource Efficiency)
   * Recommended Rate (Acre)   : {dos['dosage_kg_acre']} kg / acre  (~{dos['bags_50kg_per_acre']} bags of 50 kg)
   * Recommended Rate (Hectare): {dos['dosage_kg_ha']} kg / hectare
   * Application Method        : {dos['application_method']}
   * Split Dose Schedule       : {dos['split_schedule']}

4. PHYSIOLOGICAL ACTION (What It Does in the Crop)
   * Action in Plant           : {det['what_it_does']}
   * Soil Absorption Mechanism : {det['soil_action']}

5. SUSTAINABLE RESOURCE EFFICIENCY GUIDANCE
"""
    for tip in dos['sustainability_tips']:
        report += f"   * {tip}\n"
    report += "========================================================================================"
    return report


def interactive_mode():
    """Interactive command-line prompt."""
    soil_types, crop_types = get_supported_options()
    
    print("\n" + "=" * 65)
    print("  DEEP LEARNING FERTILIZER RECOMMENDATION SYSTEM (CLI)")
    print("=" * 65)
    
    try:
        print(f"\nSupported Soil Types: {', '.join(soil_types)}")
        soil = input("Enter Soil Type (e.g. Sandy, Loamy, Black): ").strip()
        
        print(f"\nSupported Crop Types: {', '.join(crop_types)}")
        crop = input("Enter Crop Type (e.g. Maize, Cotton, Wheat): ").strip()
        
        temp = float(input("\nEnter Temperature (deg C, e.g. 26): ").strip())
        humidity = float(input("Enter Humidity (%, e.g. 52): ").strip())
        moisture = float(input("Enter Soil Moisture (%, e.g. 38): ").strip())
        n = float(input("Enter Nitrogen level (N, e.g. 37): ").strip())
        k = float(input("Enter Potassium level (K, e.g. 0): ").strip())
        p = float(input("Enter Phosphorous level (P, e.g. 0): ").strip())
        
        print("\n[+] Running Deep Learning Inference & Dosage Optimization...")
        result = predict_fertilizer(temp, humidity, moisture, soil, crop, n, k, p)
        
        print(format_report(result))
        
    except Exception as e:
        print(f"\n[!] Error during prediction: {e}")


def run_demo_test():
    """Runs automated demo test predictions."""
    test_cases = [
        ("High Nitrogen Demand (Maize in Sandy Soil)", (26, 52, 38, "Sandy", "Maize", 37, 0, 0)),
        ("High Phosphorous Demand (Sugarcane in Loamy Soil)", (29, 52, 45, "Loamy", "Sugarcane", 12, 0, 36)),
        ("Balanced NPK Complex (Cotton in Black Soil)", (34, 65, 62, "Black", "Cotton", 7, 9, 30))
    ]
    for name, params in test_cases:
        res = predict_fertilizer(*params)
        print(format_report(res))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        run_demo_test()
