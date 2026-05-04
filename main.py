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

class LeadRequest(BaseModel):
    user_id: str
    name: str
    email: str

# --- Routes ---
@app.post("/train")
def train(req: TrainRequest):
    collection = get_collection(req.user_id)
    links = get_all_links(req.url)
    all_chunks = []
    for link in links:
        try:
            text = scrape_website(link)
            all_chunks.extend(chunk_text(text))
        except:
            pass
    for i, chunk in enumerate(all_chunks):
        collection.add(documents=[chunk], ids=[f"chunk_{i}"])
    return {"status": "success", "pages_scraped": len(links), "chunks_stored": len(all_chunks)}

@app.post("/chat")
def chat(req: ChatRequest):
    collection = get_collection(req.user_id)
    results = collection.query(query_texts=[req.question], n_results=5)
    chunks = results["documents"][0]
    context = " ".join(chunks)

    prompt = f"""You are {req.bot_name}, a smart and helpful AI assistant for this business.

Your job is to help website visitors by answering their questions clearly and in a well-organized way.

CONTEXT FROM BUSINESS WEBSITE:
{context}

RULES:
1. Answer using ONLY information from the context above
2. Format your responses clearly:
   - Use bullet points for lists
   - Use numbered steps for processes
   - Keep paragraphs short and readable
3. Be warm, professional and conversational
4. If the context contains partial info, use it and be helpful with what you know
5. If the question is completely outside the context, respond with:
   "I don't have specific information about that. To get the right answer, please reach out to the team directly:
   - [extract any email from context if available]
   - [extract any phone from context if available]
   - [extract any address from context if available]
   They'll be happy to help!"
6. NEVER make up information not in the context
7. NEVER say "based on the context" or "according to the document" — just answer naturally

VISITOR QUESTION: {req.question}

YOUR RESPONSE:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return {"answer": response.choices[0].message.content}

@app.post("/lead")
def capture_lead(req: LeadRequest):
    try:
        supabase.table("leads").insert({
            "user_id": req.user_id,
            "name": req.name,
            "email": req.email,
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