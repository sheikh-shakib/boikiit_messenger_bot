import os
from dotenv import load_dotenv
from supabase.client import create_client, Client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Load environment variables from the .env configuration file
load_dotenv()

def get_supabase_client() -> Client:
    """Initializes and returns the Supabase client using environment credentials."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    return create_client(url, key)

def fetch_and_ingest_all_knowledge():
    """
    Ingests static company FAQs AND
    queries the 'books' table to ingest dynamic product metadata using API embeddings.
    """
    # Check if HF_TOKEN is present
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("Missing HF_TOKEN in environment variables. Please add it to your .env file.")

    supabase = get_supabase_client()
    aggregated_texts = []
    metadata_list = []
    
    print("--- PART 1: Loading Static Company Knowledge ---")
    
    # 1. Core Company FAQs (Written in Bengali for the AI to memorize)
    # - Data extracted from user uploaded sync_stories.py file
    company_knowledge = [
        "BoiKiit (বইকীট) হলো বাচ্চাদের জন্য একটি কাস্টমাইজড বা পার্সোনালাইজড গল্পের বই তৈরির প্ল্যাটফর্ম।",
        "BoiKiit কীভাবে কাজ করে: অভিভাবকরা তাদের বাচ্চার নাম এবং পছন্দ অনুযায়ী থিম নির্বাচন করলে, এআই (AI) একটি সুন্দর গল্প ও ছবি তৈরি করে দেয়। এখানে বাচ্চারা স্টোরি বই পড়তে পারে। যেসব বাচ্চা পড়তে পারে নাহ তারা ভয়েস এর মাধ্যমে বইয়ের গল্প শুনতে পারে। নিজেদের নামে বই বানাতে পারে। এরপর চাইলে সেগুলা প্রিন্ট অর্ডার করতে পারে।",
        "আমাদের ডিজিটাল গল্প তৈরি করা একদম ফ্রি। তবে হার্ডকপি প্রিন্টেড বইয়ের দাম বইয়ের পেজ সংখ্যা অনুযায়ী হয় এবং ডেলিভারি চার্জ আছে।",
        "অর্ডার করার নিয়ম: বই অর্ডার করতে চাইলে বাচ্চার নাম, কাস্টম নোট, ডেলিভারি ঠিকানা এবং ফোন নম্বর দিতে হবে। পেমেন্ট (bKash/Nagad) কনফার্ম হওয়ার পর ৩-৫ কর্মদিবসের মধ্যে বই ডেলিভারি করা হয় ইনশা আল্লাহ।" 
    ]
    
    for text in company_knowledge:
        aggregated_texts.append(text)
        metadata_list.append({
            "source": "company_faq",
            "title": "General Knowledge"
        })
        
    print(f"Loaded {len(company_knowledge)} company FAQ entries.")

    print("--- PART 2: Fetching Dynamic Book Metadata ---")
    
    # 2. Extract the necessary columns based on the production schema
    try:
        response = supabase.table("books").select(
            "id, title, author, theme, target_age, description, price_standard"
        ).execute()
        books = response.data
    except Exception as e:
        print(f"Failed to fetch data from the 'books' table: {e}")
        books = []

    if books:
        # Iterate through the records to build a localized context string for the LLM
        for book in books:
            book_id = book.get("id")
            title = book.get("title", "Unknown Title")
            author = book.get("author", "BoiKiit Team")
            theme = book.get("theme", "General Story")
            target_age = book.get("target_age", "Children")
            price = book.get("price_standard", 150)
            
            # Formulated with Bengali keywords
            summary_text = (
                f"বইয়ের নাম (Book Title): {title}। "
                f"লেখক (Author): {author}। "
                f"বইয়ের ধরণ বা থিম (Theme): {theme}। "
                f"উপযোগী বয়স (Target Age): {target_age} বছর বয়সীদের জন্য। "
                f"হার্ডকপি প্রিন্টেড বইয়ের দাম (Price): {price} টাকা। "
                f"এটি একটি শিশুদের চমৎকার ইন্টারঅ্যাক্টিভ গল্পের বই।"
            )
                    
            aggregated_texts.append(summary_text)
            
            metadata_list.append({
                "source": "books_metadata_sync",
                "book_id": book_id,
                "title": title
            })
        print(f"Loaded {len(books)} books from the database.")
    else:
        print("No books found in the database to append.")

    print("--- PART 3: Initializing Embedding Pipeline via Hugging Face API ---")

    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=hf_token
    )

    # Push the combined texts (FAQs + Books) and vectors into the Supabase 'documents' table
    try:
        SupabaseVectorStore.from_texts(
            texts=aggregated_texts,
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents",
            metadatas=metadata_list
        )
        print("✅ Vector ingestion complete! The AI Agent now knows what BoiKiit is AND has the latest book data.")
    except Exception as e:
        print(f"❌ Failed to upload embeddings to the vector store: {e}")

if __name__ == "__main__":
    fetch_and_ingest_all_knowledge()