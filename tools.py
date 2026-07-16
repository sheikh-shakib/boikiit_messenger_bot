import os
from langchain_core.tools import tool
from supabase.client import create_client

@tool
def process_hardcopy_order(
    book_id: str,
    child_name: str,
    custom_note: str,
    total_amount: float,
    delivery_fee: float,
    payment_method: str,
    trx_id: str,
    delivery_phone: str,
    delivery_address: str,
    quantity: int = 1
) -> str:
    """
    Use this tool ONLY when a customer has provided their payment details (TrxID) 
    and all delivery information to complete their physical book purchase.
    """
    print(f"\n--- [DEBUG] TOOL EXECUTING: Order for {child_name} (Book: {book_id}) ---\n")
    
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    supabase = create_client(supabase_url, supabase_key)
    
    try:
        order_payload = {
            "total_amount": total_amount,
            "delivery_fee": delivery_fee,
            "status": "pending",
            "payment_method": payment_method,
            "trx_id": trx_id,
            "delivery_phone": delivery_phone,
            "delivery_address": delivery_address
        }
            
        order_response = supabase.table("orders").insert(order_payload).execute()
        inserted_order = order_response.data[0]
        new_order_id = inserted_order["id"]
        
        item_payload = {
            "order_id": new_order_id,
            "book_id": book_id,
            "package_index": 1,
            "package_type": "Standard",
            "wants_photo": False,
            "child_name": child_name,
            "custom_note": custom_note,
            "price": total_amount - delivery_fee,
            "quantity": quantity
        }
        
        supabase.table("order_items").insert(item_payload).execute()
        return "SUCCESS: Order and order_items records saved with status='pending'."
        
    except Exception as e:
        return f"Database insertion error encountered: {str(e)}"


@tool
def fetch_realtime_books(query: str = "") -> str:
    """
    Use this tool to fetch the LIVE list of books, themes, prices, and target ages from the database.
    Call this whenever a customer asks about available books, prices, or options.
    """
    try:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY")
        supabase = create_client(url, key)
        
        # Fetch data directly from the 'books' table
        response = supabase.table("books").select("id, title, author, theme, target_age, price_standard").execute()
        books = response.data
        
        if not books:
            return "বর্তমানে ডাটাবেসে কোনো বইয়ের তথ্য নেই।"
            
        result = "BoiKiit Live Book Inventory:\n"
        for b in books:
            result += f"- Book ID: {b.get('id')}, Title: {b.get('title')}, Theme: {b.get('theme')}, Age: {b.get('target_age')}+, Price: {b.get('price_standard')} TK\n"
        
        return result
        
    except Exception as e:
        return f"Database error: {str(e)}"