import os
import pandas as pd
import google.generativeai as genai
import time

def run_gemini_annotation():
    print("🚀 Initializing Gemini API for true Zero-Shot Annotation...")
    genai.configure(api_key="AIzaSyDL2gcVHsGIPcnQ3EXepOkBtxb0aKzSchA")
    
    generation_config = {
      "temperature": 0.0,    # 0.0 is critical for deterministic mathematical evaluation
      "top_p": 0.95,
      "top_k": 64,
      "max_output_tokens": 1024,
      "response_mime_type": "text/plain",
    }
    
    # Using the production 1.5 Pro model for maximum reasoning capabilities
    model = genai.GenerativeModel(
      model_name="gemini-1.5-flash-8b",
      generation_config=generation_config,
    )
    
    # 1. Load the Exact Prompt text we used for Claude
    with open('data/GPT4_EVAL_PROMPT.txt', 'r', encoding='utf-8') as f:
        prompt = f.read()

    # 2. Add strict guardrails so Gemini ONLY outputs the CSV
    prompt += "\n\nCRITICAL INSTRUCTION: Output ONLY the 60 rows of the CSV formatted exactly as requested. Do not output anything else."

    print("📊 Sending the 60 evaluations to Gemini-1.5-Pro...")
    
    try:
        response = model.generate_content(prompt)
        output = response.text.strip()
        
        # Super strict cleanup to ensure it's a perfect CSV
        # Remove markdown ticks if the model stubbornly includes them
        if output.startswith("```csv"):
            output = output[6:]
        if output.startswith("```"):
            output = output[3:]
        if output.endswith("```"):
            output = output[:-3]
        output = output.strip()
        
        # 3. Save purely to disk as our Annotator 1
        with open('data/annotator_gemini_api.csv', 'w', encoding='utf-8') as f:
            f.write(output)
            
        print("✅ Success! Generated pure LLM annotations from Gemini 1.5 Pro.")
        print("💾 Saved to: data/annotator_gemini_api.csv")
        
    except Exception as e:
        print(f"❌ API Call Failed: {e}")

if __name__ == '__main__':
    run_gemini_annotation()
