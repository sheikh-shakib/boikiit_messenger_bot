import os
import uuid
from langchain.tools import tool
from supabase.client import create_client

@tool
def process_hardcopy_order(
    user_id: str,
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
    
    All arguments are required string or numeric types mapping exactly to the DB schema.
    """
    # Initialize the Supabase client using environment variables
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    supabase = create_client(supabase_url, supabase_key)
    
    try:
        # Step 1: Create the parent record in the 'orders' table
        # We explicitly set status to 'pending' as requested
        order_payload = {
            "user_id": user_id,
            "total_amount": total_amount,
            "delivery_fee": delivery_fee,
            "status": "pending",
            "payment_method": payment_method,
            "trx_id": trx_id,
            "delivery_phone": delivery_phone,
            "delivery_address": delivery_address
        }
        
        order_response = supabase.table("orders").insert(order_payload).execute()
        
        # Extract the newly generated order UUID to link the child item
        inserted_order = order_response.data[0]
        new_order_id = inserted_order["id"]
        
        # Step 2: Create the child record in the 'order_items' table
        item_payload = {
            "order_id": new_order_id,
            "book_id": book_id,
            "package_index": 1,          # Default fallback index
            "package_type": "Standard",    # Default package fallback
            "wants_photo": False,
            "child_name": child_name,
            "custom_note": custom_note,
            "price": total_amount - delivery_fee, # Net book cost calculation
            "quantity": quantity
        }
        
        supabase.table("order_items").insert(item_payload).execute()
        
        # Return status confirmation token text back to the ReAct agent context loop
        return "SUCCESS: Order and order_items records saved with status='pending'."
        
    except Exception as e:
        return f"Database insertion error encountered: {str(e)}"