import os
import requests
import openai # Note: The openai library MUST be installed to run this version, even if the key is missing.
import time # For backoff/retry

def search_papers(topic, limit=3, max_retries=3):
    """Fetches papers from Semantic Scholar API."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        'query': topic,
        'limit': limit,
        'fields': 'title,authors,abstract,url'
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            
            # --- Rate Limit (429) Handling ---
            if response.status_code == 429:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limit hit (429). Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue # Go to the next attempt

            response.raise_for_status()
            data = response.json()
            
            papers = []
            for paper in data.get("data", []):
                authors = ", ".join([author["name"] for author in paper.get("authors", [])])
                papers.append({
                    "title": paper.get("title"),
                    "authors": authors,
                    "abstract": paper.get("abstract"),
                    "url": paper.get("url")
                })
            return papers
            
        except requests.RequestException as e:
            print(f"API error: {e}")
            return []
            
    # If all attempts failed
    print("Failed to fetch papers after multiple retries due to rate limits.")
    return []

def chatgpt_query(prompt, max_tokens=500):
    """Queries the OpenAI API for responses."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key is None:
        return None # Return None if API key is not set
    try:
        # Client needs to be initialized with the key
        client = openai.OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None # Return None on API failure

def research_chatbot():
    print("Welcome to Research Paper Chatbot (CLI)!")
    print("Enter research topics, or type 'exit' to quit.")
    
    openai_key_present = os.getenv("OPENAI_API_KEY") is not None
    if not openai_key_present:
        print("\n*** WARNING: OPENAI_API_KEY is NOT set. AI summarization and suggestions are disabled. ***")
    
    while True:
        topic = input("\nYour topic: ").strip()
        if topic.lower() == 'exit':
            print("Goodbye!")
            break

        papers = search_papers(topic, limit=3)
        
        if not papers:
            print("\nNo papers found.")
            if openai_key_present:
                print("Checking for related topics using AI...")
                suggestion_prompt = f"No exact research papers found for the topic '{topic}'. Suggest related topics or papers."
                suggestions = chatgpt_query(suggestion_prompt)
                if suggestions:
                     print(f"\nChatGPT Suggestions:\n{suggestions}")
            else:
                 print("Cannot provide suggestions. OPENAI_API_KEY not set.")
            continue

        print(f"\nTop {len(papers)} papers found on '{topic}':")
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"  Authors: {paper['authors']}")
            print(f"  URL: {paper['url']}")
            
            if paper['abstract']:
                if openai_key_present:
                    summary_prompt = f"Summarize this abstract simply:\n{paper['abstract']}"
                    summary = chatgpt_query(summary_prompt)
                    if summary:
                        print(f"  Summary: {summary}")
                    else:
                        print("  Summary: Could not generate AI summary due to an API error.")
                else:
                    print("  Summary: AI summarization skipped. OPENAI_API_KEY not set.")
                    print(f"  Abstract (Snippet): {paper['abstract'][:300]}...") # Print a snippet of the abstract
            else:
                print("  Summary: Abstract not available.")

if __name__ == "__main__":
    research_chatbot()