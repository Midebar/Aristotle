# openai_compat_server.py
import json
import threading
from flask import Flask, request, jsonify
import time
import uuid

# we import backend here to allow main script to set backend before starting server
BACKEND = None  # will be set by run script

app = Flask(__name__)

def _make_choice_text(text: str):
    return {
        "index": 0,
        "message": {"role": "assistant", "content": text},
        "finish_reason": "stop"
    }

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    global BACKEND
    payload = request.get_json(force=True)
    # payload contains messages list; take last user content
    messages = payload.get("messages") or []
    prompt = ""
    if messages:
        # join content of messages to form prompt; this is a simple heuristic
        prompt = "\n".join([m.get("content", "") for m in messages])
    # parameters
    max_tokens = payload.get("max_tokens") or payload.get("max_new_tokens") or 256
    temperature = float(payload.get("temperature", 0.0))
    top_p = float(payload.get("top_p", 1.0))
    # call backend
    text = ""
    try:
        text = BACKEND.generate(prompt, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p, do_sample=temperature>0)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    choice = _make_choice_text(text)
    resp = {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "choices": [choice],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }
    return jsonify(resp)

@app.route("/v1/completions", methods=["POST"])
def completions():
    global BACKEND
    payload = request.get_json(force=True)
    prompt = payload.get("prompt", "")
    max_tokens = payload.get("max_tokens", 256)
    temperature = float(payload.get("temperature", 0.0))
    top_p = float(payload.get("top_p", 1.0))
    try:
        text = BACKEND.generate(prompt, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p, do_sample=temperature>0)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    resp = {
        "id": str(uuid.uuid4()),
        "object": "text_completion",
        "choices": [{"text": text, "index": 0}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }
    return jsonify(resp)

def run_server(bind_host="127.0.0.1", port=11434, backend=None):
    global BACKEND
    BACKEND = backend
    # run flask in a background thread (not blocking)
    def _run():
        app.run(host=bind_host, port=port, debug=False, threaded=True)
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    # give server a moment
    time.sleep(1.2)
    return t
