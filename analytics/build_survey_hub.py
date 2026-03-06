import pandas as pd
import json

def build_html():
    df = pd.read_csv('data/human_eval_form_annotator_1.csv')
    
    questions = []
    for _, row in df.iterrows():
        code_snapshot = row['Component Snapshot (AST Data)'].replace('Component Name: nan', 'Component Name: (Anonymous)')
        questions.append({
            "id": row['Pair ID'],
            "query": row['Query'],
            "code": code_snapshot
        })
        
    js_data = json.dumps(questions)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Relevance Survey</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f3f4f6;
            color: #1f2937;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            overflow: hidden;
        }}
        .container {{
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            max-width: 600px;
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }}
        .header {{ text-align: center; }}
        .header h1 {{ font-size: 1.5rem; margin-bottom: 0.5rem; color: #111827; }}
        .progress {{ font-size: 0.875rem; color: #6b7280; font-weight: bold; }}
        
        .query-box {{
            background: #eef2ff;
            border-left: 4px solid #4f46e5;
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            font-size: 1.125rem;
            font-weight: 500;
        }}
        
        .code-box {{
            background: #1e293b;
            color: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.9rem;
            white-space: pre-wrap;
            overflow-x: auto;
            max-height: 250px;
            overflow-y: auto;
        }}
        
        .buttons {{
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
        }}
        
        button {{
            flex: 1;
            padding: 0.75rem;
            border: none;
            border-radius: 6px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            color: white;
        }}
        
        .btn-yes {{ background-color: #10b981; }}
        .btn-yes:hover {{ background-color: #059669; transform: translateY(-2px); }}
        
        .btn-no {{ background-color: #ef4444; }}
        .btn-no:hover {{ background-color: #dc2626; transform: translateY(-2px); }}
        
        .hidden {{ display: none !important; }}
        
        #download-btn {{
            background-color: #4f46e5;
            width: 100%;
            margin-top: 1rem;
        }}
        #download-btn:hover {{ background-color: #4338ca; }}
    </style>
</head>
<body>

    <div class="container" id="survey-container">
        <div class="header">
            <h1>Component Relevance Assessment</h1>
            <div class="progress" id="progress-text">Question 1 of 60</div>
        </div>
        
        <div class="query-box">
            <strong>Query:</strong> <span id="q-text">loading...</span>
        </div>
        
        <div class="code-box" id="c-text">
            loading...
        </div>
        
        <div style="text-align: center; font-weight: bold; font-size: 1.1rem;">
            Does this component successfully match the query request?
        </div>
        
        <div class="buttons">
            <button class="btn-yes" onclick="submitAnswer(1)">Yes, Looks Relevant</button>
            <button class="btn-no" onclick="submitAnswer(0)">No, Not Relevant</button>
        </div>
    </div>

    <!-- Done Screen -->
    <div class="container hidden" id="done-container" style="text-align: center;">
        <h1 style="font-size: 3rem; margin: 0;">🎉</h1>
        <h2>All Done! Thank You!</h2>
        <p style="color: #4b5563;">You have completed the annotation task.</p>
        <button id="download-btn" onclick="submitToServer()">Submit Answers securely</button>
    </div>

    <script>
        const questions = {js_data};
        let currentIndex = 0;
        const answers = [];

        function renderQuestion() {{
            if (currentIndex >= questions.length) {{
                document.getElementById('survey-container').classList.add('hidden');
                document.getElementById('done-container').classList.remove('hidden');
                return;
            }}
            
            const q = questions[currentIndex];
            document.getElementById('progress-text').innerText = `Question ${{currentIndex + 1}} of ${{questions.length}}`;
            document.getElementById('q-text').innerText = q.query;
            document.getElementById('c-text').innerText = q.code;
        }}

        function submitAnswer(score) {{
            const q = questions[currentIndex];
            answers.push({{
                "Pair ID": q.id,
                "Query": q.query,
                "Is Relevant? (1=Yes, 0=No)": score
            }});
            
            currentIndex++;
            renderQuestion();
        }}

        async function submitToServer() {{
            const username = prompt("Please enter your Discord Handle or Name:");
            if (!username) return;
            
            const btn = document.getElementById('download-btn');
            btn.innerText = "Submitting...";
            btn.disabled = true;

            const payload = {{
                annotator_id: username,
                answers: answers
            }};

            try {{
                const res = await fetch('https://script.google.com/macros/s/AKfycbxWIuGWCFPzBARkL1t_mM7xGUe0Qv92wfTmfYD9d1HlJ1nzOqdd3E1zm4yRe35PiffwkQ/exec', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'text/plain;charset=utf-8' }},
                    body: JSON.stringify(payload)
                }});
                
                if (res.ok) {{
                    document.getElementById('done-container').innerHTML = 
                        "<h2>✅ Success!</h2><p>Your answers have been securely submitted direct to Karthick!</p><p>You can close this tab now. Thank you so much!</p>";
                }} else {{
                    alert("Error submitting. Please try again.");
                    btn.innerText = "Submit Answers securely";
                    btn.disabled = false;
                }}
            }} catch (err) {{
                alert("Error submitting to server: " + err);
                btn.innerText = "Submit Answers securely";
                btn.disabled = false;
            }}
        }}

        // Start
        renderQuestion();
    </script>
</body>
</html>
"""
    with open('survey.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ Created auto-submitting Survey App: survey.html")

if __name__ == "__main__":
    build_html()
