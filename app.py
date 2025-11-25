import os
import requests
import time
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai.errors import APIError

# --- GEMINI API SETUP ---
GEMINI_API_KEY = "AIzaSyAI3SgaNMciKOzrayNj9FEJit2datME84c"
client = genai.Client(api_key=GEMINI_API_KEY)

app = Flask(__name__)

# --- SEMANTIC SCHOLAR (Free Paper Search API) ---
def search_papers(topic, limit=5, max_retries=3):
    """
    Fetches papers from Semantic Scholar API with retry logic to handle 429 (rate limit) errors.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        'query': topic,
        'limit': limit,
        'fields': 'title,authors,abstract,url,year,venue'
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            if response.status_code == 429:
                wait_time = 2 ** attempt
                print(f"Rate limit hit. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            data = response.json()

            papers = []
            for paper in data.get("data", []):
                authors = ", ".join([a["name"] for a in paper.get("authors", [])])
                papers.append({
                    "title": paper.get("title"),
                    "authors": authors,
                    "abstract": paper.get("abstract") or "No abstract available.",
                    "url": paper.get("url"),
                    "year": paper.get("year") or "Unknown",
                    "venue": paper.get("venue") or "Unknown"
                })
            return papers
        except requests.RequestException as e:
            return {"error": str(e)}

    return {"error": "Failed to fetch papers after multiple retries."}

# --- GEMINI CHAT FUNCTION ---
def chat_query(prompt):
    """
    Sends prompt to Gemini API and returns a short natural response.
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction="You are a helpful and conversational research assistant. Keep replies short and natural.",
                temperature=0.6,
                max_output_tokens=120
            )
        )
        return response.text
    except APIError as e:
        return f"Gemini API Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# --- FLASK ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def api_search():
    data = request.json
    topic = data.get('topic')
    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    papers = search_papers(topic)
    if isinstance(papers, dict) and "error" in papers:
        return jsonify({"error": papers["error"]}), 500

    if not papers:
        return jsonify({"message": "No papers found for this topic."})

    for paper in papers:
        paper['summary'] = "(Summary is the full abstract)\n\n" + (paper.get('abstract') or "No summary available.")
    return jsonify({"papers": papers})

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json
    message = data.get('message')
    if not message:
        return jsonify({"error": "Message is required"}), 400

    reply = chat_query(message)
    if reply.startswith("Gemini API Error"):
        return jsonify({"error": reply}), 500

    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(debug=True)
