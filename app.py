import os
from dotenv import load_dotenv
from flask import Flask, request
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import requests

from supabase.client import create_client
# Switched to the Inference API to save server RAM
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_groq import ChatGroq
from langchain.agents import create_agent

# Integrate custom transaction tool
from tools import process_hardcopy_order

# Load credentials from the environment configuration
load_dotenv()
app = Flask(__name__)

# Added HF_TOKEN to the required variables list
REQUIRED_ENV_VARS = ["GROQ_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_KEY", "FB_VERIFY_TOKEN", "FB_PAGE_ACCESS_TOKEN", "HF_TOKEN"]
for var in REQUIRED_ENV_VARS:
    if not os.environ.get(var):
        raise ValueError(f"Missing required environment variable: {var}")

# 1. Initialize Supabase Connection
supabase_client = create_client(
    os.environ.get("SUPABASE_URL"), 
    os.environ.get("SUPABASE_SERVICE_KEY")
)

embeddings = HuggingFaceEndpointEmbeddings(
    api_key=os.environ.get("HF_TOKEN"),
    model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" 
)

# 3. Mount the Supabase pgvector instance
vector_store = SupabaseVectorStore(
    client=supabase_client,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

# 4. Bind the RAG Knowledge System as a tool for the agent
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
from langchain_core.tools import create_retriever_tool
rag_knowledge_tool = create_retriever_tool(
    retriever,
    "boikiit_knowledge_base",
    "Use this tool to find information about BoiKiit stories, customizations, and general policies."
)

# 5. Connect to the high-speed Llama-3 model on Groq
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.15)

# 6. Collate operational tools
tools = [rag_knowledge_tool, process_hardcopy_order]

# ---------------------------------------------------------------------------
# Strict System Instruction Ruleset (English Logic, Bengali Output)
# ---------------------------------------------------------------------------
system_rules = (
    "You are a professional customer support assistant for 'BoiKiit'. "
    "Your core mission is to assist customers with book inquiries and sales in Bengali.\n\n"
    
    "STRICT OPERATIONAL GUIDELINES:\n"
    "1. LANGUAGE: You must ONLY communicate in natural, fluent Bengali. Never show any internal technical details, "
    "tool names, function calls, or raw JSON strings to the customer. If you need to use a tool, do it silently in the background.\n"
    "2. PROFESSIONALISM: Your tone should be warm, polite, and helpful. If a user asks a question outside of your business scope "
    "(like random chat), handle it briefly and return to BoiKiit-related topics.\n"
    "3. TOOL USAGE: When you use the 'boikiit_knowledge_base' tool, summarize the findings into a human-readable, friendly Bengali response. "
    "NEVER mention the name of the tool to the customer.\n\n"
    "GREETING PROTOCOL:\n"

    "1. If the user greets in English (e.g., 'Hello', 'Hi'), reply naturally in Bengali with a warm, modern greeting like 'হ্যালো! কেমন আছেন?' or 'হ্যালো! আপনাকে বইকিটে স্বাগতম।' DO NOT use 'নমস্কার' if the input is in English, unless the user specifically prefers it.If user gives salam in Islamic ritual(আসসালামু আলাইকুম,assalamu alaikum) then reply with salam\n"
    "2. If the user greets in Bengali, reply appropriately in Bengali.\n"
    "3. NEVER display tool calls (like <function=...>) to the user. Always parse tool results silently.\n\n"
    
    "GENERAL RULES:\n"
    "1. Always maintain a professional, helpful, and friendly Bengali tone.\n"
    
    "SALES & PAYMENT PROTOCOL:\n"
    "1. Only initiate the sales process when the customer shows interest in purchasing a hardcopy.\n"
    "2. Collect: Child's Name, Custom Note, Delivery Address, and Phone Number.\n"
    "3. Total Cost: 150 (Book) + 80 (Delivery) = 230 BDT. Instruct them to send payment to bKash/Nagad: 01744492986.\n"
    "4. MANDATORY: Do not confirm the order or trigger the 'process_hardcopy_order' tool until the customer provides the Transaction ID (TrxID).\n"
    "5. If a customer provides details but not the TrxID, politely remind them that payment is required to finalize the order."

    "NAMING RULE: When referring to the brand name in Bengali, always write it as 'বইকীট' to maintain linguistic accuracy, instead of 'বইকিট'."
)

# 7. Initialize the agent using the modern unified harness architecture
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_rules
)

# ---------------------------------------------------------------------------
# Webhook Processing Routes
# ---------------------------------------------------------------------------

@app.route('/webhook', methods=['GET'])
def fb_verification_handshake():
    """Validates real-time verification status from the Meta developer platform."""
    if request.args.get("hub.verify_token") == os.environ.get("FB_VERIFY_TOKEN"):
        return request.args.get("hub.challenge"), 200
    return "Forbidden: Verification token mismatch", 403

@app.route('/webhook', methods=['POST'])
def handle_incoming_page_events():
    """Receives inbound live messaging payloads from Facebook servers."""
    payload = request.json
    if payload.get("object") == "page":
        for entry in payload.get("entry", []):
            for event in entry.get("messaging", []):
                if "message" in event and "text" in event["message"]:
                    sender_id = event["sender"]["id"]
                    user_query = event["message"]["text"]
                    
                    print(f"Incoming client query from {sender_id}: {user_query}")
                    
                    # Run invocation against the graph state machine
                    try:
                        response = agent.invoke({
                            "messages": [{"role": "user", "content": user_query}]
                        })
                        # Extract the final message content from the response thread state
                        ai_reply = response["messages"][-1].content
                    except Exception as error:
                        print(f"Internal Agent Exception: {error}")
                        ai_reply = "দুঃখিত, এই মুহূর্তে একটি কারিগরি ত্রুটি ঘটেছে। দয়া করে আবার চেষ্টা করুন।"
                    
                    # Dispatch the resulting text back to the platform user
                    dispatch_fb_response(sender_id, ai_reply)
                    
        return "EVENT_RECEIVED", 200
    return "Not Found", 404

def dispatch_fb_response(recipient_id: str, text_content: str):
    """Sends outbound communications via the Meta Graph API."""
    url = "https://graph.facebook.com/v19.0/me/messages"
    params = {"access_token": os.environ.get("FB_PAGE_ACCESS_TOKEN")}
    json_payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": text_content}
    }
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, params=params, json=json_payload, headers=headers)
    if response.status_code != 200:
        print(f"Meta Graph API error status code {response.status_code}: {response.text}")
    else:
        print("Response successfully dispatched to Messenger.")

if __name__ == "__main__":
    app.run(port=5000)