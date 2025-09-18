#!/usr/bin/env python3
"""
Simple test to check if basic Python web server works
"""

import http.server
import socketserver
import webbrowser
import threading
import time

def test_server():
    """Test basic web server functionality"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Server</title>
    </head>
    <body>
        <h1>Python Web Server Test</h1>
        <p>If you see this, Python is working!</p>
        <p>Time: {}</p>
    </body>
    </html>
    """.format(time.strftime("%Y-%m-%d %H:%M:%S"))
    
    with open('test_index.html', 'w') as f:
        f.write(html_content)
    
    PORT = 8081
    Handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Server starting at http://localhost:{PORT}")
            print("Press Ctrl+C to stop")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")

if __name__ == "__main__":
    test_server()