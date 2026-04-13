from __future__ import annotations
import re
from PIL import Image
import pytesseract
from fertilizer_logic import SoilCard

def extract_soil_values(image: Image.Image) -> SoilCard:
    """
    Experimental OCR to extract N, P, K, pH, and OC from a Soil Health Card image.
    Returns a SoilCard with default values if extraction fails.
    """
    # Default values for fallback
    data = {
        "n": 200.0,
        "p": 15.0,
        "k": 200.0,
        "ph": 6.8,
        "oc": 0.7
    }

    try:
        # Perform OCR
        text = pytesseract.image_to_string(image)
        lines = text.lower().split('\n')

        # Simple regex patterns for Indian Soil Health Cards
        patterns = {
            "n": r"(?:nitrogen|n)\s*[:\-]?\s*(\d+(?:\.\d+)?)",
            "p": r"(?:phosphorus|p)\s*[:\-]?\s*(\d+(?:\.\d+)?)",
            "k": r"(?:potassium|k)\s*[:\-]?\s*(\d+(?:\.\d+)?)",
            "ph": r"(?:ph|soil ph)\s*[:\-]?\s*(\d+(?:\.\d+)?)",
            "oc": r"(?:oc|organic carbon|carbon)\s*[:\-]?\s*(\d+(?:\.\d+)?)",
        }

        for key, pattern in patterns.items():
            for line in lines:
                match = re.search(pattern, line)
                if match:
                    data[key] = float(match.group(1))
                    break
    except Exception as e:
        print(f"OCR Error: {e}")

    return SoilCard(
        n=data["n"],
        p=data["p"],
        k=data["k"],
        ph=data["ph"],
        organic_carbon=data["oc"]
    )
