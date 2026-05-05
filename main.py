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
    user_id: str
    name: str
    email: str
    intent: str = None
    budget: str = None
    timeline: str = None
    urgency: str = "low"
    conversation: str = None

# --- Routes ---
import json

@app.post("/chat")
def chat(req: ChatRequest):
    collection = get_collection(req.user_id)

    # Get relevant chunks
    results = collection.query(query_texts=[req.question], n_results=5)
    context = " ".join(results["documents"][0])

    # Always fetch contact info
    contact_results = collection.query(query_texts=["contact phone email address location booking"], n_results=3)
    contact_context = " ".join(contact_results["documents"][0])

    # Build conversation history string
    history_str = ""
    if req.conversation_history:
        for msg in req.conversation_history[-6:]:  # last 6 messages for context
            role = "Visitor" if msg["role"] == "user" else req.bot_name
            history_str += f"{role}: {msg['content']}\n"

    # Main chat response
    prompt = f"""You are {req.bot_name}, a smart and friendly AI assistant for this business.

BUSINESS INFORMATION:
{context}

CONTACT & BOOKING INFO:
{contact_context}

CONVERSATION SO FAR:
{history_str}

INSTRUCTIONS:
- Answer helpfully using business information above
- Format responses neatly with bullet points for lists
- Be warm, conversational and professional
- Detect visitor intent from the conversation
- If visitor shows buying intent or urgency, naturally guide them toward booking/contact
- If visitor asks about pricing, give available info then ask about their specific needs
- If you don't have specific info, provide contact details from above
- Never say "based on context" or "according to the document"
- Never make up information

VISITOR: {req.question}
{req.bot_name}:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    answer = response.choices[0].message.content

    # Silent background extraction
    extraction_prompt = f"""Extract lead information from this conversation. Return ONLY a valid JSON object, nothing else.

Conversation:
{history_str}
Visitor: {req.question}
Assistant: {answer}

Return this exact JSON structure with null for missing fields:
{{"intent": null, "budget": null, "timeline": null, "urgency": "low"}}

urgency must be: "high", "medium", or "low"
Intent should describe what the visitor wants in a few words.
Budget should be the amount mentioned or null.
Timeline should be when they want it or null."""

    extracted = {"intent": None, "budget": None, "timeline": None, "urgency": "low"}
    try:
        extraction_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.1,
        )
        raw = extraction_response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        extracted = json.loads(raw)
    except Exception as e:
        print(f"Extraction error: {e}")

    return {
        "answer": answer,
        "extracted": extracted,
    }

@app.post("/lead")
def capture_lead(req: LeadRequest):
    try:
        supabase.table("leads").insert({
            "user_id": req.user_id,
            "name": req.name,
            "email": req.email,
            "intent": req.intent,
            "budget": req.budget,
            "timeline": req.timeline,
            "urgency": req.urgency,
            "conversation": req.conversation,
        }).execute()
    except Exception as e:
        print(f"Lead capture error: {e}")
    return {"status": "success", "message": "Lead captured"}

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
        }
    except Exception as e:
        print(f"Settings error: {e}")
        return {}