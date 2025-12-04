# rag_core.py (Enhanced for detailed explanations)

# --- Enhanced Mock Knowledge Base ---
# This provides detailed info for each specific class.
MOCK_KNOWLEDGE_BASE = {
    "akiec": {
        "source": "American Academy of Dermatology (AAD)",
        "text": "Actinic Keratoses (AK) are rough, scaly patches on the skin caused by years of sun exposure. They are considered precancerous, as they can sometimes develop into squamous cell carcinoma. It's crucial to have these treated by a dermatologist to prevent progression."
    },
    "bcc": {
        "source": "Skin Cancer Foundation",
        "text": "Basal Cell Carcinoma (BCC) is the most common type of skin cancer. It frequently appears as a flesh-colored, pearl-like bump or a pinkish patch of skin. While it grows slowly and rarely spreads, it can be disfiguring if not treated early."
    },
    "bkl": {
        "source": "Mayo Clinic",
        "text": "Benign Keratosis-like lesions (BKL) are non-cancerous skin growths. Seborrheic keratoses are common examples, often appearing as waxy, wart-like growths. While harmless, they can sometimes be mistaken for skin cancer."
    },
    "df": {
        "source": "DermNet NZ",
        "text": "Dermatofibromas (DF) are common, harmless, button-like reddish-brown nodules, often found on the legs. They are firm to the touch and may dimple when pinched. No treatment is necessary unless it causes discomfort."
    },
    "mel": {
        "source": "American Cancer Society",
        "text": "Melanoma is the most serious type of skin cancer because it can spread rapidly to other parts of the body. The 'ABCDE' rule is a key guide: Look for Asymmetry, an irregular Border, uneven Color, a Diameter larger than a pencil eraser, and any Evolving change. Early detection and treatment are critical."
    },
    "nv": {
        "source": "AAD",
        "text": "Melanocytic Nevi (NV) are common moles. Most people have between 10 and 40. They are typically benign, uniform in color, and round or oval. Monitoring moles for changes is a key part of skin health."
    },
    "vasc": {
        "source": "Cleveland Clinic",
        "text": "Vascular lesions are abnormalities of the blood vessels. This includes cherry angiomas (small, red bumps) and port-wine stains. Most are benign and primarily a cosmetic concern, but any new or changing lesion should be evaluated."
    }
}

def get_recommendation(predicted_class):
    """
    Enhanced function to get a detailed recommendation based on the specific class.
    The `predicted_class` (e.g., 'mel') is passed from the app.
    """
    return MOCK_KNOWLEDGE_BASE.get(predicted_class, {
        "source": "DermaScan AI",
        "text": "This is a recognized skin condition. For a detailed explanation and next steps, please consult a board-certified dermatologist."
    })

def chat_with_ai(query):
    """
    Mock function for the AI assistant chat.
    """
    query_lower = query.lower()
    if "asymmetry" in query_lower:
        return {"source": "AAD", "text": "Asymmetry means one half of a mole does not match the other. This is a key characteristic of melanoma."}
    elif "border" in query_lower:
        return {"source": "Skin Cancer Foundation", "text": "An irregular, scalloped, or poorly defined border is a warning sign for potential melanoma."}
    elif "color" in query_lower:
        return {"source": "Mayo Clinic", "text": "Having a variety of colors is a warning signal. Look for different shades of brown, tan, or black."}
    else:
        return {"source": "DermaScan AI", "text": "Based on your scan, it's important to monitor the lesion for any changes over time. If you notice any of the ABCDE signs, consult a dermatologist."}