import os
import re
import requests
import threading
from dotenv import load_dotenv
from flask import Flask, request

from langchain_groq import ChatGroq
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.agents import create_agent

# Integrate custom transaction tools
from tools import process_hardcopy_order, fetch_realtime_books

load_dotenv()
app = Flask(__name__)

REQUIRED_ENV_VARS = [
    "GROQ_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_KEY", 
    "FB_VERIFY_TOKEN", "FB_PAGE_ACCESS_TOKEN", 
    "MISTRAL_API_KEY", "HUGGINGFACEHUB_API_TOKEN"
]
for var in REQUIRED_ENV_VARS:
    if not os.environ.get(var):
        raise ValueError(f"Missing required environment variable: {var}")

# --- 1. LLM & Fallback Setup ---
llm_groq = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.15)
llm_mistral = ChatMistralAI(model="mistral-large-latest", temperature=0.15)

hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-70B-Instruct", 
    task="text-generation",
    temperature=0.15
)
llm_hf = ChatHuggingFace(llm=hf_endpoint)

llm = llm_groq.with_fallbacks([llm_mistral, llm_hf])
tools = [fetch_realtime_books, process_hardcopy_order]

user_sessions = {}

# --- 2. System Prompt & Rules ---
system_rules = """
    You are a strict and professional customer support assistant for 'BoiKiit'. 
    Your core mission is to assist customers with book inquiries and sales in Bengali.

    COMPANY KNOWLEDGE:
    - BoiKiit (বইকীট) হলো বাচ্চাদের জন্য একটি কাস্টমাইজড বা পার্সোনালাইজড গল্পের বই তৈরির প্ল্যাটফর্ম।
    - বইকীটে বাচ্চারা ফ্রিতে বই পড়তে পারে। যেসব বাচ্চা পড়া জানে গঠন, তাদের জন্য 'ভয়েস অ্যাসিস্ট্যান্ট' আছে, যা দিয়ে তারা বইয়ের গল্প শুনতে পারে।
    - ডিজিটাল গল্প তৈরি একদম ফ্রি। তবে প্রিন্টেড হার্ডকপি বইয়ের দাম পেজ অনুযায়ী হয় এবং ডেলিভারি চার্জ আছে।

    STRICT OPERATIONAL GUIDELINES:
    1. LANGUAGE: You must ONLY communicate in natural, fluent Bengali.
    2. BOOK INQUIRIES & ANTI-HALLUCINATION (CRITICAL): 
    - ALWAYS execute the 'fetch_realtime_books' tool to check available books.
    - NEVER invent, guess, or make up book titles. ONLY mention the EXACT book names provided by the tool.
    - If the tool returns no books, simply say: 'বর্তমানে আমাদের নতুন বইয়ের স্টক আপডেট করা হচ্ছে।'
    3. PARTIAL NAMES: If a customer types a partial book name, intelligently match it to the correct full book name in the inventory.
    4. NO RAW CODE: NEVER output raw tool syntax like <function=...>. Keep tool usage completely hidden.

    ORDERING & CONFIRMATION PROTOCOL (CRITICAL):
    1. REQUIRED DETAILS: Collect: 1. Book ID (keep hidden), 2. Child's Name, 3. Delivery Address, 4. Phone Number, 5. Transaction ID (TrxID) for payment of (Book Price + 80 TK delivery) to bKash/Nagad 01744492986.
    2. PROGRESSIVE COLLECTION: If the user provides some details but not all, explicitly show them what you ALREADY have and politely ask ONLY for the remaining details.
    3. FINAL CONFIRMATION: When all details are collected, present a clear summary and ask for final confirmation ('সব তথ্য কি ঠিক আছে? কনফার্ম করলে আমরা অর্ডারটি সাবমিট করব।').
    4. TOOL EXECUTION: After the user confirms, you MUST execute the 'process_hardcopy_order' tool first. ONLY after the tool returns 'SUCCESS', you are allowed to send the 'POST-ORDER MESSAGE' to the user. Do not say order is submitted until the tool confirms success.
    5. DATABASE TOOL MAPPING (CRITICAL): When executing 'process_hardcopy_order':
       - 'user_id': Do not provide this argument (leave it empty).
       - 'delivery_fee': Always set to 80.
       - 'total_amount': Calculate (Book Price + 80).
       - 'payment_method': Detect from user input (e.g., 'bKash' or 'Nagad'). If unknown, use 'Online'.
       - 'custom_note': Set to 'N/A' if the user hasn't provided one.
    6. POST-ORDER MESSAGE: After submission, say: 'আপনার অর্ডারটি সফলভাবে সাবমিট হয়েছে এবং বর্তমানে পেন্ডিং অবস্থায় আছে। পেমেন্ট ভেরিফিকেশনের পর ডেলিভারি শুরু হবে। আরও ভালো এক্সপেরিয়েন্সের জন্য আমাদের BoiKiit অ্যাপটি ইন্সটল করুন: https://play.google.com/store/apps/details?id=com.shebokit.boikiit'

    NAMING RULE: Always write the brand name as 'বইকীট'.
"""

agent = create_agent(
    model=llm, 
    tools=tools,
    system_prompt=system_rules
)

# --- 3. Background Processing ---
def process_message_background(sender_id, user_query):
    print(f"Processing client query from {sender_id}: {user_query}")
    
    if sender_id not in user_sessions:
        user_sessions[sender_id] = []
    
    user_sessions[sender_id].append({"role": "user", "content": user_query})
    
    chat_history = user_sessions[sender_id][-20:] 
    
    try:
        response = agent.invoke({"messages": chat_history})
        ai_reply = response["messages"][-1].content

        # Regex Filter: Remove raw function calling syntax
        ai_reply = re.sub(r"<function=.*?</function>", "", ai_reply, flags=re.DOTALL).strip()
        
    except Exception as error:
        print(f"Internal Agent Exception: {error}")
        ai_reply = "আসসালামু আলাইকুম।অনুগ্রহ করে অপেক্ষা করুন।আমাদের প্রতিনিধি আপনার সাথে অতি দ্রুত যোগাযোগ করবেন।"
    
    user_sessions[sender_id].append({"role": "assistant", "content": ai_reply})
    dispatch_fb_response(sender_id, ai_reply)

# --- 4. Webhook Routes ---
@app.route('/webhook', methods=['GET'])
def fb_verification_handshake():
    if request.args.get("hub.verify_token") == os.environ.get("FB_VERIFY_TOKEN"):
        return request.args.get("hub.challenge"), 200
    return "Forbidden: Verification token mismatch", 403

@app.route('/webhook', methods=['POST'])
def handle_incoming_page_events():
    payload = request.json
    if payload.get("object") == "page":
        for entry in payload.get("entry", []):
            for event in entry.get("messaging", []):
                if "message" in event and "text" in event["message"]:
                    sender_id = event["sender"]["id"]
                    user_query = event["message"]["text"]
                    
                    # Start a background thread to process the AI reply
                    thread = threading.Thread(target=process_message_background, args=(sender_id, user_query))
                    thread.start()
                    
        # IMMEDIATELY return 200 OK to Facebook so they stop retrying
        return "EVENT_RECEIVED", 200
    return "Not Found", 404

def dispatch_fb_response(recipient_id: str, text_content: str):
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

if __name__ == "__main__":
    app.run(port=5000)