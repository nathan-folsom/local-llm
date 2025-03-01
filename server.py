from http.server import BaseHTTPRequestHandler, HTTPServer
from mlx_lm import stream_generate, load
from mlx_lm.models.cache import make_prompt_cache
from time import time

import json

last = time()


def print_offset(msg):
    global last
    next_timestamp = time()
    print(f"+{int((next_timestamp - last) * 1000)}ms {msg}")
    last = next_timestamp


model_path = "/Users/nate/ai/Llama-3.3-70B-Instruct-8bit/"
print_offset(f"Starting up model {model_path}")
model, tokenizer = load(model_path)
prompt_cache = make_prompt_cache(model)
print_offset("Finished model initialization")


class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
            if 'messages' not in data:
                self.send_response(400)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b"Missing 'messages' field in request body")
                return
            self.chat(data)
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-type', 'text/plain')

    def chat(self, data):
        print_offset(f"Received: {data}")
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        messages = data['messages']
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        print_offset("Starting response")
        for data in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            prompt_cache=prompt_cache,
            max_tokens=-1,
        ):
            print_offset(f"Generated token {data.text}")
            response = {'message': {'content': data.text, 'role': "assistant"}}
            self.wfile.write(json.dumps(response).encode('utf-8') + b'\n')
        print_offset("Finished response")


def run_server():
    print_offset("Starting up HTTP server")
    port = 8000
    server_address = ('', port)
    httpd = HTTPServer(server_address, RequestHandler)
    print_offset(f"Server running on http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Error running server: {e}")


if __name__ == "__main__":
    run_server()
