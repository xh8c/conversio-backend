import os
import re
import json
import threading
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
from typing import Optional
from fastembed import TextEmbedding

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# --- Scraping ---
def scrape_website(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    response = requests.get(url, timeout=5, headers=headers)
    response.encoding = response.apparent_encoding or "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "head", "header", "iframe", "noscript", "svg", "img", "button", "form", "input", "meta", "link"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r'[^\x20-\x7E]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_all_links(base_url: str) -> list:
    try:
        from urllib.parse import urlparse, urljoin
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(base_url, timeout=5, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        links = set()
        links.add(base_url)
        base_domain = urlparse(base_url).netloc
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            if parsed.netloc == base_domain and parsed.scheme in ("http", "https"):
                clean = parsed._replace(fragment="", query="").geturl()
                links.add(clean)
        return list(links)
    except:
        return [base_url]

def chunk_text(text: str, chunk_size: int = 500) -> list:
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# --- Vector Storage ---
def store_chunks(user_id: str, chunks: list):
    # Delete existing chunks for this user
    supabase.table("document_chunks").delete().eq("user_id", user_id).execute()
    
    # Generate embeddings and store in batches
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        embeddings = list(embedding_model.embed(batch))
        embeddings = [e.tolist() for e in embeddings]
        rows = [
            {
                "user_id": user_id,
                "content": chunk,
                "embedding": embedding
            }
            for chunk, embedding in zip(batch, embeddings)
        ]
        supabase.table("document_chunks").insert(rows).execute()
    print(f"Stored {len(chunks)} chunks for user {user_id}")

def search_chunks(user_id: str, query: str, n: int = 5) -> list:
    try:
        query_embedding = list(embedding_model.embed([query]))[0].tolist()
        result = supabase.rpc("match_chunks", {
            "query_embedding": query_embedding,
            "match_user_id": user_id,
            "match_count": n
        }).execute()
        if result.data:
            return [row["content"] for row in result.data]
    except Exception as e:
        print(f"Search error: {e}")
    return []

# --- Request Models ---
class TrainRequest(BaseModel):
    user_id: str
    url: str

class ChatRequest(BaseModel):
    user_id: str
    question: str
    bot_name: str = "Assistant"
    fallback_message: str = "I don't have that information, please contact us directly."
    conversation_history: list = []

class LeadRequest(BaseModel):
    user_id: str = ""
    name: str = ""
    email: str = ""
    intent: Optional[str] = None
    budget: Optional[str] = None
    timeline: Optional[str] = None
    urgency: Optional[str] = "low"
    conversation: Optional[str] = None

class UpdateLeadRequest(BaseModel):
    user_id: str
    email: str
    conversation_history: list = []

# --- Routes ---
@app.post("/train")
def train(req: TrainRequest):
    def run_training():
        try:
            print(f"Training started for {req.user_id} on {req.url}")
            links = get_all_links(req.url)
            print(f"Found {len(links)} pages")

            all_chunks = []
            # Filter out image/binary URLs
            skip_extensions = ('.webp', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.pdf', '.zip', '.mp4', '.mp3')
            links = [l for l in links if not any(l.lower().endswith(ext) for ext in skip_extensions)]
            print(f"Filtered to {len(links)} pages after removing images")

            for link in links:
                try:
                    print(f"Scraping: {link}")
                    text = scrape_website(link)
                    all_chunks.extend(chunk_text(text))
                    print(f"Done: {link} — {len(all_chunks)} chunks")
                except requests.exceptions.Timeout:
                    print(f"Timeout: {link}")
                except Exception as e:
                    print(f"Failed: {link} — {e}")

            if all_chunks:
                store_chunks(req.user_id, all_chunks)

            # Generate FAQ buttons
            faq_buttons = ["Our Services", "Pricing", "Location", "Contact Us"]
            try:
                sample_text = " ".join(all_chunks[:5])
                faq_prompt = f"""Based on this business website content, generate exactly 4 short FAQ button labels visitors would most likely click.

Content: {sample_text[:2000]}

Rules:
- Each label must be 2-4 words maximum
- Return ONLY a JSON array of 4 strings, nothing else
- Example: ["Our Services", "Pricing", "Book Now", "Contact Us"]"""

                faq_response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": faq_prompt}],
                    temperature=0.3,
                    max_tokens=100,
                )
                raw = faq_response.choices[0].message.content.strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    faq_buttons = parsed[:4]
            except Exception as e:
                print(f"FAQ error: {e}")

            # Save to Supabase
            try:
                existing = supabase.table("chatbot_settings").select("user_id").eq("user_id", req.user_id).execute()
                update_data = {
                    "faq_buttons": faq_buttons,
                    "training_status": "complete",
                }
                if existing.data:
                    supabase.table("chatbot_settings").update(update_data).eq("user_id", req.user_id).execute()
                else:
                    supabase.table("chatbot_settings").insert({"user_id": req.user_id, **update_data}).execute()
            except Exception as e:
                print(f"Settings save error: {e}")

            print(f"Training complete for {req.user_id}")

        except Exception as e:
            print(f"Training error: {e}")
            try:
                supabase.table("chatbot_settings").update({"training_status": "error"}).eq("user_id", req.user_id).execute()
            except:
                pass

    # Set status to training immediately
    try:
        existing = supabase.table("chatbot_settings").select("user_id").eq("user_id", req.user_id).execute()
        if existing.data:
            supabase.table("chatbot_settings").update({"training_status": "training"}).eq("user_id", req.user_id).execute()
        else:
            supabase.table("chatbot_settings").insert({"user_id": req.user_id, "training_status": "training"}).execute()
    except:
        pass

    thread = threading.Thread(target=run_training)
    thread.daemon = True
    thread.start()

    return {"status": "training_started", "message": "Training started in background."}

@app.post("/chat")
def chat(req: ChatRequest):
    context_chunks = search_chunks(req.user_id, req.question, n=3)
    context = " ".join(context_chunks)[:2000]

    contact_chunks = search_chunks(req.user_id, "contact phone email address booking location", n=2)
    contact_context = " ".join(contact_chunks)[:800]

    history_str = ""
    if req.conversation_history:
        for msg in req.conversation_history[-3:]:
            role = "Visitor" if msg["role"] == "user" else req.bot_name
            history_str += f"{role}: {msg['content']}\n"

    prompt = f"""You are {req.bot_name}, a smart friendly AI assistant for this business.

BUSINESS INFO:
{context}

CONTACT & BOOKING:
{contact_context}

CONVERSATION:
{history_str}

INSTRUCTIONS:
- Answer using only business info above
- Use bullet points for lists
- Be warm and conversational
- Guide toward booking/contact when appropriate
- Never make up information

VISITOR: {req.question}
{req.bot_name}:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return {"answer": response.choices[0].message.content}

@app.post("/lead")
def capture_lead(req: LeadRequest):
    try:
        supabase.table("leads").insert({
            "user_id": req.user_id,
            "name": req.name,
            "email": req.email,
            "intent": req.intent or None,
            "budget": req.budget or None,
            "timeline": req.timeline or None,
            "urgency": req.urgency or "low",
            "conversation": req.conversation or None,
        }).execute()
    except Exception as e:
        print(f"Lead error: {e}")
    return {"status": "success"}

@app.post("/update-lead")
def update_lead(req: UpdateLeadRequest):
    try:
        history_str = "\n".join([f"{'Visitor' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in req.conversation_history])
        extraction_prompt = f"""Extract lead information from this conversation. Return ONLY valid JSON.

Conversation:
{history_str}

Return exactly:
{{"intent": null, "budget": null, "timeline": null, "urgency": "low"}}"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.1,
            max_tokens=100,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        extracted = json.loads(raw)

        update_data = {"conversation": json.dumps(req.conversation_history)}
        if extracted.get("intent"): update_data["intent"] = extracted["intent"]
        if extracted.get("budget"): update_data["budget"] = extracted["budget"]
        if extracted.get("timeline"): update_data["timeline"] = extracted["timeline"]
        if extracted.get("urgency"): update_data["urgency"] = extracted["urgency"]

        supabase.table("leads").update(update_data).eq("user_id", req.user_id).eq("email", req.email).execute()
    except Exception as e:
        print(f"Update lead error: {e}")
    return {"status": "success"}

@app.post("/extract")
def extract_lead_info(req: ChatRequest):
    history_str = ""
    if req.conversation_history:
        for msg in req.conversation_history:
            role = "Visitor" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

    extraction_prompt = f"""Extract lead information from this conversation. Return ONLY valid JSON.

Conversation:
{history_str}

Return exactly:
{{"intent": null, "budget": null, "timeline": null, "urgency": "low"}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.1,
            max_tokens=100,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"Extraction error: {e}")
        return {"intent": None, "budget": None, "timeline": None, "urgency": "low"}

@app.get("/settings/{user_id}")
def get_settings(user_id: str):
    try:
        result = supabase.table("chatbot_settings").select("*").eq("user_id", user_id).single().execute()
        data = result.data
        if not data:
            return {}
        return {
            "botName": data.get("bot_name", "Assistant"),
            "welcomeMessage": data.get("welcome_message", "Hi! How can I help you today?"),
            "fallbackMessage": data.get("fallback_message", "I don't have that information."),
            "primaryColor": data.get("primary_color", "#6C63FF"),
            "position": data.get("position", "bottom-right"),
            "collectName": data.get("collect_name", True),
            "collectEmail": data.get("collect_email", True),
            "logoUrl": data.get("logo_url", ""),
            "faqButtons": data.get("faq_buttons", []),
            "trainingStatus": data.get("training_status", "idle"),
        }
    except Exception as e:
        print(f"Settings error: {e}")
        return {}

@app.get("/")
def root():
    return {"status": "Chatbot API is running"}