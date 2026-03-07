from http.server import BaseHTTPRequestHandler
import json
import urllib.parse
from datetime import datetime
import os

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-Type")
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
            
            # Since Vercel serverless has a read-only filesystem, 
            # we need to proxy the answers to a free webhook or Google Sheets, 
            # BUT an even easier trick is just sending it back as a massive encoded URL redirect parameter! 
            # Wait, no, we can post it right into a free MongoDB data API, 
            # or even a free Discord Webhook!
                
            raise NotImplementedError("Vercel File system is read only.")
            
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode('utf-8'))
