## my_app.py
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys

class GETHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type','text/plain')
        self.end_headers()
        self.wfile.write('Hello, world!\n'.encode('utf-8'))
        python_version = f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"
        self.wfile.write(f'Python version {python_version}'.encode('utf-8'))

# variable required by Vercel
handler = GETHandler