"""Simple Flask web app for testing FastAPI endpoints."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import requests
from flask import Flask, render_template_string, request


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

app = Flask(__name__)


HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>API Test</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    .card { border: 1px solid #ddd; padding: 16px; margin-bottom: 16px; }
    input { padding: 6px; }
    button { padding: 6px 10px; }
    pre { background: #f5f5f5; padding: 12px; }
  </style>
</head>
<body>
  <h1>FastAPI Test</h1>
  <p>API Base: {{ api_base }}</p>

  <div class="card">
    <h2>Register</h2>
    <form method="post" action="/register">
      <label>Class ID:</label>
      <input name="class_id" required />
      <label>Timeout (sec):</label>
      <input name="timeout_sec" value="30" />
      <button type="submit">Register</button>
    </form>
  </div>

  <div class="card">
    <h2>Update</h2>
    <form method="post" action="/update">
      <label>Class ID:</label>
      <input name="class_id" required />
      <label>Timeout (sec):</label>
      <input name="timeout_sec" value="30" />
      <button type="submit">Update</button>
    </form>
  </div>

  <div class="card">
    <h2>Verify</h2>
    <form method="post" action="/verify">
      <button type="submit">Verify</button>
    </form>
  </div>

  {% if result %}
    <h2>Result</h2>
    <pre>{{ result }}</pre>
  {% endif %}

  {% if image_data %}
    <h2>Image</h2>
    <img src="{{ image_data }}" style="max-width: 360px; border: 1px solid #ddd;" />
  {% endif %}
</body>
</html>
"""


def _post_json(path: str, payload: Dict[str, Any]) -> Any:
    url = f"{API_BASE_URL}{path}"
    res = requests.post(url, json=payload, timeout=300)
    res.raise_for_status()
    try:
        return res.json()
    except ValueError:
        return {"raw": res.text}


def _render_result(result: Any) -> str:
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=True, indent=2)
    return str(result)


def _extract_image_data(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None

    image = result.get("image")
    if not isinstance(image, dict):
        return None

    data = image.get("data")
    if not isinstance(data, str) or not data:
        return None

    return f"data:image/jpeg;base64,{data}"


@app.get("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        api_base=API_BASE_URL,
        result=None,
        image_data=None,
    )


@app.post("/register")
def register():
    class_id = request.form.get("class_id", "").strip()
    timeout_sec = float(request.form.get("timeout_sec", "30"))
    result = _post_json("/register", {"class_id": class_id, "timeout_sec": timeout_sec})
    return render_template_string(
        HTML_TEMPLATE,
        api_base=API_BASE_URL,
        result=_render_result(result),
        image_data=_extract_image_data(result),
    )


@app.post("/update")
def update():
    class_id = request.form.get("class_id", "").strip()
    timeout_sec = float(request.form.get("timeout_sec", "30"))
    result = _post_json("/update", {"class_id": class_id, "timeout_sec": timeout_sec})
    return render_template_string(
        HTML_TEMPLATE,
        api_base=API_BASE_URL,
        result=_render_result(result),
        image_data=_extract_image_data(result),
    )


@app.post("/verify")
def verify():
    result = _post_json("/verify", {})
    return render_template_string(
        HTML_TEMPLATE,
        api_base=API_BASE_URL,
        result=_render_result(result),
        image_data=_extract_image_data(result),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)