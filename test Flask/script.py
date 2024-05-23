
from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai

def get_gemini_response(prompt):
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    print("API Key:", api_key)
    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 2048,
        "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        generation_config=generation_config,
    )

    chat_session = model.start_chat()
    response = chat_session.send_message(prompt)
    print(response.text)
    return response.text

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/hello', methods=['POST'])
def hello():
    first_name = request.form['first_name']
    message = get_gemini_response(first_name)
    return jsonify(message=message)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500, debug=True)

