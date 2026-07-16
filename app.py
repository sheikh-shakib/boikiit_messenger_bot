import os
from dotenv import load_dotenv
from flask import Flask, request
import requests

from langchain_groq import ChatGroq
from langchain.agents import create_agent

# Integrate custom transaction tools
from tools import process_hardcopy_order, fetch_realtime_books

load_dotenv()
app = Flask(__name__)

REQUIRED_ENV_VARS = ["GROQ_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_KEY", "FB_VERIFY_TOKEN", "FB_PAGE_ACCESS_TOKEN"]
for var in REQUIRED_ENV_VARS:
    if not os.environ.get(var):
        raise ValueError(f"Missing required environment variable: {var}")

llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.15)
tools = [fetch_realtime_books, process_hardcopy_order]

# ---------------------------------------------------------------------------
# বোটের মেমোরি (Session Dictionary) তৈরি করা হলো
# ---------------------------------------------------------------------------
user_sessions = {}

# ---------------------------------------------------------------------------
# System Rules আপডেট (আংশিক নাম বোঝা এবং টুল লিক বন্ধ করার কড়া নির্দেশ)
# ---------------------------------------------------------------------------
system_rules = (
    "You are a strict and professional customer support assistant for 'BoiKiit'. "
    "Your core mission is to assist customers with book inquiries and sales in Bengali.\n\n"
    
    "COMPANY KNOWLEDGE:\n"
    "- BoiKiit (বইকীট) হলো বাচ্চাদের জন্য একটি কাস্টমাইজড বা পার্সোনালাইজড গল্পের বই তৈরির প্ল্যাটফর্ম।\n"
    "- ডিজিটাল গল্প তৈরি একদম ফ্রি। তবে প্রিন্টেড হার্ডকপি বইয়ের দাম পেজ অনুযায়ী হয় এবং ডেলিভারি চার্জ আছে।\n\n"
    
    "STRICT OPERATIONAL GUIDELINES:\n"
    "1. LANGUAGE: You must ONLY communicate in natural, fluent Bengali.\n"
    "2. BOOK INQUIRIES: ALWAYS use 'fetch_realtime_books' tool to get live data. ONLY show the Book Name and Price to the user. NEVER show Internal IDs to the user.\n"
    "3. PARTIAL NAMES: If a customer types a partial book name (e.g., 'সততার বাঁশি'), intelligently match it to the correct full book name in the inventory.\n"
    "4. NO RAW CODE: NEVER output raw tool syntax like <function=...>. Keep tool usage completely hidden from the user.\n\n"
    
    "SALES & PAYMENT PROTOCOL:\n"
    "1. If the user selects a book, ask for: Child's Name, Custom Note, Delivery Address, and Phone Number in a polite way.\n"
    "2. Total Cost: Book Price + 80 BDT (Delivery). Instruct them to send payment to bKash/Nagad: 01744492986.\n"
    "3. MANDATORY: Wait for the user to provide the Transaction ID (TrxID) before confirming the order via the 'process_hardcopy_order' tool.\n\n"

    "NAMING RULE: Always write the brand name as 'বইকীট'."
)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_rules
)

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
                    
                    print(f"Incoming client query from {sender_id}: {user_query}")
                    
                    # 1. মেমোরি চেক: নতুন ইউজার হলে তার জন্য হিস্ট্রি খাতা খোলা
                    if sender_id not in user_sessions:
                        user_sessions[sender_id] = []
                    
                    # 2. ইউজারের নতুন মেসেজটি হিস্ট্রিতে যোগ করা
                    user_sessions[sender_id].append({"role": "user", "content": user_query})
                    
                    # 3. শুধুমাত্র শেষের ৬টি মেসেজ পাঠানো (যাতে টোকেন লিমিট ক্রস না করে)
                    chat_history = user_sessions[sender_id][-6:]
                    
                    try:
                        # 4. পুরো চ্যাট হিস্ট্রি এআই-এর কাছে পাঠানো
                        response = agent.invoke({
                            "messages": chat_history
                        })
                        ai_reply = response["messages"][-1].content
                    except Exception as error:
                        print(f"Internal Agent Exception: {error}")
                        ai_reply = "দুঃখিত, এই মুহূর্তে একটি কারিগরি ত্রুটি ঘটেছে। দয়া করে আবার চেষ্টা করুন।"
                    
                    # 5. এআই-এর দেওয়া উত্তরটাও হিস্ট্রিতে সেভ করে রাখা
                    user_sessions[sender_id].append({"role": "assistant", "content": ai_reply})
                    
                    dispatch_fb_response(sender_id, ai_reply)
                    
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