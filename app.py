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

# Removed HF_TOKEN as it is no longer needed
REQUIRED_ENV_VARS = ["GROQ_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_KEY", "FB_VERIFY_TOKEN", "FB_PAGE_ACCESS_TOKEN"]
for var in REQUIRED_ENV_VARS:
    if not os.environ.get(var):
        raise ValueError(f"Missing required environment variable: {var}")

# 1. Connect to the high-speed Llama-3 model on Groq
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.15)

# 2. Collate operational tools
tools = [fetch_realtime_books, process_hardcopy_order]

# ---------------------------------------------------------------------------
# Strict System Instruction Ruleset (English Logic, Bengali Output)
# ---------------------------------------------------------------------------
system_rules = (
    "You are a strict and professional customer support assistant for 'BoiKiit'. "
    "Your core mission is to assist customers with book inquiries and sales in Bengali.\n\n"
    
    "COMPANY KNOWLEDGE (MUST USE THIS TO ANSWER QUESTIONS ABOUT BOIKIIT):\n"
    "- BoiKiit (বইকীট) হলো বাচ্চাদের জন্য একটি কাস্টমাইজড বা পার্সোনালাইজড গল্পের বই তৈরির প্ল্যাটফর্ম।\n"
    "- BoiKiit কীভাবে কাজ করে: অভিভাবকরা বাচ্চার নাম এবং থিম নির্বাচন করলে, এআই (AI) সুন্দর গল্প ও ছবি তৈরি করে দেয়। বাচ্চারা স্টোরি বই পড়তে ও ভয়েসের মাধ্যমে শুনতে পারে।\n"
    "- ডিজিটাল গল্প তৈরি একদম ফ্রি। তবে প্রিন্টেড হার্ডকপি বইয়ের দাম পেজ অনুযায়ী হয় এবং ডেলিভারি চার্জ আছে।\n"
    "- WARNING: Never say we sell novels, poetry, or regular books. We ONLY make customized storybooks for kids.\n\n"
    
    "STRICT OPERATIONAL GUIDELINES:\n"
    "1. LANGUAGE: You must ONLY communicate in natural, fluent Bengali. Never show any internal technical details.\n"
    "2. PROFESSIONALISM: Your tone should be warm, polite, and helpful.\n"
    "3. BOOK INQUIRIES: If a user asks what books are available or their prices, ALWAYS use the 'fetch_realtime_books' tool to get live data.\n\n"
    
    "GREETING PROTOCOL:\n"
    "1. If the user greets in English (e.g., 'Hello', 'Hi'), reply naturally in Bengali like 'হ্যালো! বইকীটে আপনাকে স্বাগতম।' DO NOT use 'নমস্কার' unless preferred by the user. If user gives salam (আসসালামু আলাইকুম), reply with salam.\n\n"
    
    "SALES & PAYMENT PROTOCOL:\n"
    "1. Only initiate the sales process when the customer shows interest in purchasing a hardcopy.\n"
    "2. Collect: Child's Name, Custom Note, Delivery Address, and Phone Number.\n"
    "3. Total Cost: Book Price (from fetch_realtime_books) + 80 BDT (Delivery). Instruct them to send payment to bKash/Nagad: 01744492986.\n"
    "4. MANDATORY: Do not confirm the order or trigger the 'process_hardcopy_order' tool until the customer provides the Transaction ID (TrxID).\n\n"

    "NAMING RULE: Always write the brand name as 'বইকীট'."
)

# 3. Initialize the agent
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
                    
                    try:
                        response = agent.invoke({
                            "messages": [{"role": "user", "content": user_query}]
                        })
                        ai_reply = response["messages"][-1].content
                    except Exception as error:
                        print(f"Internal Agent Exception: {error}")
                        ai_reply = "দুঃখিত, এই মুহূর্তে একটি কারিগরি ত্রুটি ঘটেছে। দয়া করে আবার চেষ্টা করুন।"
                    
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
    else:
        print("Response successfully dispatched to Messenger.")

if __name__ == "__main__":
    app.run(port=5000)