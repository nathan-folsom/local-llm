from http.server import BaseHTTPRequestHandler, HTTPServer
from mlx_lm import stream_generate, load
from mlx_lm.models.cache import make_prompt_cache

import json

model_path = "/Users/nate/ai/CodeLlama-34b-Instruct-hf-4bit/"
print(f"Starting up model {model_path}")
model, tokenizer = load(model_path)
prompt_cache = make_prompt_cache(model)


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
        print(f"Received: {data}")
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        messages = data['messages']
        print(f"Tokenizing messages: {messages}")
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        print(f"Tokenized: {messages}")

        for data in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            prompt_cache=prompt_cache,
            max_tokens=-1,
        ):
            response = {'message': {'content': data.text, 'role': "assistant"}}
            self.wfile.write(json.dumps(response).encode('utf-8') + b'\n')


def run_server():
    print("Starting up HTTP server")
    port = 8000
    server_address = ('', port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"Server running on http://localhost:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
