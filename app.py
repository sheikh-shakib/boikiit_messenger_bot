import os
from dotenv import load_dotenv
from flask import Flask, request
import re
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

user_sessions = {}

system_rules = (
    "You are a strict and professional customer support assistant for 'BoiKiit'. "
    "Your core mission is to assist customers with book inquiries and sales in Bengali.\n\n"
    
    "COMPANY KNOWLEDGE:\n"
    "- BoiKiit (বইকীট) হলো বাচ্চাদের জন্য একটি কাস্টমাইজড বা পার্সোনালাইজড গল্পের বই তৈরির প্ল্যাটফর্ম।\n"
    "- বইকীটে বাচ্চারা ফ্রিতে বই পড়তে পারে। যেসব বাচ্চা পড়া জানে না, তাদের জন্য 'ভয়েস অ্যাসিস্ট্যান্ট' (Voice Assistant) আছে, যা দিয়ে তারা বইয়ের গল্প শুনতে পারে।\n"
    "- ডিজিটাল গল্প তৈরি একদম ফ্রি। তবে প্রিন্টেড হার্ডকপি বইয়ের দাম পেজ অনুযায়ী হয় এবং ডেলিভারি চার্জ আছে।\n\n"
    
    ""STRICT OPERATIONAL GUIDELINES:\n"
    "1. LANGUAGE: You must ONLY communicate in natural, fluent Bengali.\n"
    "2. BOOK INQUIRIES & ANTI-HALLUCINATION (CRITICAL): \n"
    "   - ALWAYS execute the 'fetch_realtime_books' tool to check available books.\n"
    "   - NEVER invent, guess, hallucinate, or make up book titles (e.g., do not say 'হাসির গল্প', 'সাহসী মেয়েটি' unless the tool explicitly returns them).\n"
    "   - ONLY mention the EXACT book names provided by the tool's output.\n"
    "   - If the tool returns no books, simply say: 'বর্তমানে আমাদের নতুন বইয়ের স্টক আপডেট করা হচ্ছে।'\n"
    "3. PARTIAL NAMES: If a customer types a partial book name, intelligently match it to the correct full book name in the inventory.\n"
    "4. NO RAW CODE: NEVER output raw tool syntax like <function=...>. Keep tool usage completely hidden.\n\n"
    
    "ORDERING & CONFIRMATION PROTOCOL (CRITICAL):\n"
    "1. REQUIRED DETAILS: To place an order, you must collect: 1. Book ID (keep hidden), 2. Child's Name, 3. Delivery Address, 4. Phone Number, 5. Transaction ID (TrxID) for payment of (Book Price + 80 TK delivery) to bKash/Nagad 01744492986.\n"
    "2. PROGRESSIVE COLLECTION: If the user provides some details but not all, explicitly show them what you ALREADY have (e.g., 'আমরা আপনার ফোন নম্বর এবং TrxID পেয়েছি') and politely ask ONLY for the remaining missing details.\n"
    "3. FINAL CONFIRMATION (MANDATORY): When you have collected ALL required details, DO NOT trigger the 'process_hardcopy_order' tool yet. First, present a clear summary of all the provided details to the user and ask for their final confirmation (e.g., 'সব তথ্য কি ঠিক আছে? কনফার্ম করলে আমরা অর্ডারটি সাবমিট করব।').\n"
    "4. TOOL EXECUTION: ONLY trigger the 'process_hardcopy_order' tool AFTER the user explicitly says 'Yes', 'ঠিক আছে', 'ok', or confirms the summary.\n"
    "5. POST-ORDER MESSAGE: After the tool successfully saves the order, tell the user exactly this in Bengali:\n"
    "   - 'আপনার অর্ডারটি সফলভাবে আমাদের ডাটাবেজে সাবমিট হয়েছে এবং বর্তমানে পেন্ডিং অবস্থায় আছে। আমাদের টিম হিউম্যান ভেরিফিকেশন (Human Verification) সম্পন্ন করার পর আপনার ডেলিভারি প্রসেস শুরু হবে।'\n"
    "   - 'আরও ভালো এক্সপেরিয়েন্সের জন্য আমাদের BoiKiit অ্যাপটি ইন্সটল করতে পারেন। অ্যাপ থেকে বইয়ের প্রিভিউ পড়ে এবং কাস্টমাইজ করে খুব সহজেই অর্ডার করা যায়। অ্যাপ লিংক: [https://play.google.com/store/apps/details?id=com.shebokit.boikiit]'\n\n"

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
                    
                    if sender_id not in user_sessions:
                        user_sessions[sender_id] = []
                    
                    user_sessions[sender_id].append({"role": "user", "content": user_query})
                    
                    chat_history = user_sessions[sender_id][-6:]
                    
                    try:
                        response = agent.invoke({
                            "messages": chat_history
                        })
                        ai_reply = response["messages"][-1].content
                        
                        ai_reply = re.sub(r'<function=.*?</function>', '', ai_reply, flags=re.DOTALL).strip()
                        
                    except Exception as error:
                        print(f"Internal Agent Exception: {error}")
                        ai_reply = "দুঃখিত, এই মুহূর্তে একটি কারিগরি ত্রুটি ঘটেছে। দয়া করে আবার চেষ্টা করুন।"
                    
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