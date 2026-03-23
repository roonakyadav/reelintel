import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_fingerprint(full_text: str):

    prompt = f"""
You are analyzing a reel promoting a hidden SaaS product.

Important:
The reel mentions tools like Lovable, V0, Cursor — but those are export targets,
NOT the main product.

Identify the main product being promoted.

Focus on:
- What website the user is told to visit
- What that website provides
- What makes it different from Lovable/V0/etc

Extract:

1. main_product_role (what it actually is, very specific)
2. core_mechanism (how it works)
3. business_model_hint (free templates? marketplace? prompt library?)
4. export_targets (where outputs are sent)
5. distinctive_visual_clues
6. rare_or_brand_like_terms_from_ocr

Return ONLY valid JSON:

{{
  "main_product_role": "",
  "core_mechanism": "",
  "business_model_hint": "",
  "export_targets": [],
  "distinctive_visual_clues": [],
  "rare_or_brand_like_terms_from_ocr": []
}}

Content:
{full_text}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You analyze product promotions and return structured JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    print("GROQ RAW FINGERPRINT RESPONSE:")
    print(content)

    import re

    json_match = re.search(r"\{[\s\S]*\}", content)

    if not json_match:
        return {}

    json_string = json_match.group(0)

    try:
        return json.loads(json_string)
    except Exception as e:
        print("JSON Parse Error:", e)
        return {}