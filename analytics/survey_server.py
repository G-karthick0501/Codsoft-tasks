import http.server
import socketserver
import json
from datetime import datetime
import pandas as pd
import os

PORT = 8001

class SurveyHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Disable caching logic to ensure the survey reloads properly
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

    def do_POST(self):
        if self.path == '/submit':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                
                # Format Data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                raw_id = data.get('annotator_id', f"anonymous_{timestamp}")
                annotator_id = ''.join(e for e in raw_id if e.isalnum() or e in ['-', '_'])
                answers = data.get('answers', [])
                
                # Save Data as CSV magically onto the host's machine
                df = pd.DataFrame(answers)
                df['Annotator'] = annotator_id
                
                os.makedirs('data', exist_ok=True)
                out_file = f"data/annotator_{annotator_id}.csv"
                df.to_csv(out_file, index=False)
                
                print(f"\n{'='*60}")
                print(f"🎉 RECEIVED NEW SUBMISSION FROM DISCORD: @{annotator_id}")
                print(f"✅ Securely saved to {out_file} ({len(answers)} answers)")
                print(f"{'='*60}\n")
                
                # Return standard 200 OK so the browser closes
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))
            
            except Exception as e:
                print(f"❌ Error processing POST: {e}")
                self.send_response(500)
                self.end_headers()
            return
            
        self.send_response(404)
        self.end_headers()

print(f"Starting auto-save HTTP survey server on port {PORT}...")
print("Waiting for Discord submissions to arrive...")
print("(Press Ctrl+C to close the server when you have all the answers you need)")

# Set socket options to avoid "address already in use"
import socket
socketserver.TCPServer.allow_reuse_address = True

with socketserver.TCPServer(("127.0.0.1", PORT), SurveyHandler) as httpd:
    httpd.serve_forever()
