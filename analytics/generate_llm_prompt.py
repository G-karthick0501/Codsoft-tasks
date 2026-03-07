import pandas as pd

def generate_gpt4_prompt():
    df = pd.read_csv('data/human_eval_form_annotator_1.csv')
    
    prompt = """You are an expert React developer participating in a double-blind annotation study for an academic paper on Code Search and Retrieval.

Your task is to mathematically evaluate whether 60 React components meet the specific requirements of 60 user search queries. You will not see the source code; instead, you are provided with purely structural Abstract Syntax Tree (AST) properties of the code (e.g., hooks, lines of code, event handlers).

For each Pair ID below, respond with ONLY the Pair ID and either the number `1` (Yes, it matches the query's structural constraints) or `0` (No, it violates the constraints). 

Output your answers in a STRICT CSV format block like this:
Pair ID,Is Relevant
Q1-C1,1
Q1-C2,0
...

DO NOT include any other text, reasoning, or markdown around the CSV block. Only output the 60 rows.

#########################################
EVALUATION DATASET:
#########################################

"""
    for _, row in df.iterrows():
        prompt += f"Pair ID: {row['Pair ID']}\n"
        prompt += f"Query: {row['Query']}\n"
        prompt += f"{row['Component Snapshot (AST Data)']}\n"
        prompt += "-"*40 + "\n\n"
        
    with open('data/GPT4_EVAL_PROMPT.txt', 'w', encoding='utf-8') as f:
        f.write(prompt)
        
    print("✅ Created GPT-4 Prompt at data/GPT4_EVAL_PROMPT.txt")

if __name__ == '__main__':
    generate_gpt4_prompt()
