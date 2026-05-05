import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
from typing import Optional

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

def get_collection(user_id: str):
    return chroma_client.get_or_create_collection(name=f"user_{user_id}")

def scrape_website(url: str) -> str:
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)

def get_all_links(base_url: str) -> list:
    try:
        response = requests.get(base_url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        links = set()
        links.add(base_url)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/"):
                href = base_url.rstrip("/") + href
            if base_url in href:
                links.add(href.split("?")[0].split("#")[0])
        return list(links)
    except:
        return [base_url]

def chunk_text(text: str, chunk_size: int = 500) -> list:
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

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
# --- Routes ---
import json

@app.post("/train")
def train(req: TrainRequest):
    collection = get_collection(req.user_id)
    links = get_all_links(req.url)
    links = links[:20]
    all_chunks = []
    for link in links:
        try:
            text = scrape_website(link)
            all_chunks.extend(chunk_text(text))
        except:
            pass
    if all_chunks:
        for i, chunk in enumerate(all_chunks):
            try:
                collection.add(documents=[chunk], ids=[f"chunk_{i}_{req.user_id}"])
            except:
                collection.update(documents=[chunk], ids=[f"chunk_{i}_{req.user_id}"])

    # Auto-generate FAQ buttons
    faq_buttons = ["Our Services", "Pricing", "Location", "Contact Us"]
    try:
        sample_text = " ".join(all_chunks[:5])
        faq_prompt = f"""Based on this business website content, generate exactly 4 short FAQ button labels visitors would most likely click.

Content: {sample_text[:2000]}

Rules:
- Each label must be 2-4 words maximum
- Make them action-oriented or question-based  
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
        if isinstance(parsed, list) and len(parsed) > 0:
            faq_buttons = parsed[:4]
    except Exception as e:
        print(f"FAQ generation error: {e}")

    # Save to Supabase
    try:
        existing = supabase.table("chatbot_settings").select("user_id").eq("user_id", req.user_id).execute()
        if existing.data:
            supabase.table("chatbot_settings").update({"faq_buttons": faq_buttons}).eq("user_id", req.user_id).execute()
        else:
            supabase.table("chatbot_settings").insert({"user_id": req.user_id, "faq_buttons": faq_buttons}).execute()
    except Exception as e:
        print(f"FAQ save error: {e}")

    return {"status": "success", "pages_scraped": len(links), "chunks_stored": len(all_chunks), "faq_buttons": faq_buttons}

@app.post("/chat")
def chat(req: ChatRequest):
    collection = get_collection(req.user_id)
    results = collection.query(query_texts=[req.question], n_results=5)
    context = " ".join(results["documents"][0])
    contact_results = collection.query(query_texts=["contact phone email address booking"], n_results=3)
    contact_context = " ".join(contact_results["documents"][0])

    history_str = ""
    if req.conversation_history:
        for msg in req.conversation_history[-6:]:
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
- Use bullet points for lists, short paragraphs otherwise
- Be warm and conversational
- If visitor shows buying intent, guide toward booking/contact
- If you lack specific info, share contact details
- Never say "based on context" or make up info

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

@app.get("/")
def root():
    return {"status": "Chatbot API is running"}

@app.get("/settings/{user_id}")
def get_settings(user_id: str):
    try:
        result = supabase.table("chatbot_settings").select("*").eq("user_id", user_id).single().execute()
        data = result.data
        if not data:
            return {}
        return {
            "botName": data.get("bot_name", "Assistant"),
            "welcomeMessage": data.get("welcome_message", "Hi! How can I help you?"),
            "fallbackMessage": data.get("fallback_message", "I don't have that information."),
            "primaryColor": data.get("primary_color", "#6C63FF"),
            "position": data.get("position", "bottom-right"),
            "collectName": data.get("collect_name", True),
            "collectEmail": data.get("collect_email", True),
            "logoUrl": data.get("logo_url", ""),
            "faqButtons": data.get("faq_buttons", []),
        }
    except Exception as e:
        print(f"Settings error: {e}")
        return {}

@app.post("/extract")
def extract_lead_info(req: ChatRequest):
    history_str = ""
    if req.conversation_history:
        for msg in req.conversation_history:
            role = "Visitor" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

    extraction_prompt = f"""Extract lead information from this conversation. Return ONLY valid JSON, nothing else.

Conversation:
{history_str}

Return exactly this structure with null for missing fields:
{{"intent": null, "budget": null, "timeline": null, "urgency": "low"}}

urgency must be "high", "medium", or "low".
intent = what they want in a few words.
budget = amount mentioned or null.
timeline = when they need it or null."""

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

class UpdateLeadRequest(BaseModel):
    user_id: str
    email: str
    conversation_history: list = []

@app.post("/update-lead")
def update_lead(req: UpdateLeadRequest):
    try:
        history_str = "\n".join([f"{'Visitor' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in req.conversation_history])
        
        extraction_prompt = f"""Extract lead information from this conversation. Return ONLY valid JSON.

Conversation:
{history_str}

Return exactly this structure with null for missing fields:
{{"intent": null, "budget": null, "timeline": null, "urgency": "low"}}

urgency must be "high", "medium", or "low".
intent = what they want in a few words.
budget = amount mentioned or null.
timeline = when they need it or null."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.1,
            max_tokens=100,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        extracted = json.loads(raw)

        update_data = {
            "conversation": json.dumps(req.conversation_history),
        }
        if extracted.get("intent"): update_data["intent"] = extracted["intent"]
        if extracted.get("budget"): update_data["budget"] = extracted["budget"]
        if extracted.get("timeline"): update_data["timeline"] = extracted["timeline"]
        if extracted.get("urgency"): update_data["urgency"] = extracted["urgency"]

        supabase.table("leads").update(update_data).eq("user_id", req.user_id).eq("email", req.email).execute()

    except Exception as e:
        print(f"Update lead error: {e}")
    return {"status": "success"}