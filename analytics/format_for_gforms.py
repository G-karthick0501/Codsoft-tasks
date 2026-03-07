import pandas as pd

df = pd.read_csv('data/human_eval_form_annotator_1.csv')

form_records = []
for _, row in df.iterrows():
    # Merge the Query and the Component Stats into one massive question string
    question_text = (
        f"QUERY: '{row['Query']}'\n\n"
        f"--- IS THIS COMPONENT RELEVANT? ---\n"
        f"{row['Component Snapshot (AST Data)']}"
    )
    
    form_records.append({
        "Question": question_text,
        "Type": "Multiple Choice",
        "Option 1": "1 - Yes, Relevant",
        "Option 2": "0 - No, Not Relevant"
    })

# Save to a new Google Forms optimized sheet
pd.DataFrame(form_records).to_csv('data/GOOGLE_FORMS_UPLOAD.csv', index=False)
print("✅ Saved data/GOOGLE_FORMS_UPLOAD.csv. Ready for Google Sheets!")
