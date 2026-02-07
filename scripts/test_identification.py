"""
Test script for book identification.

Verifies that the IdentificationService can correctly find books
based on typical OCR output.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load env vars
load_dotenv()

from shelfsense.identification.google_books import GoogleBooksClient
from shelfsense.identification.service import IdentificationService

async def main():
    print("Initializing Identification Service...")
    client = GoogleBooksClient(api_key=os.getenv("GOOGLE_BOOKS_API_KEY"))
    service = IdentificationService(client)
    
    test_cases = [
        "JARRY POTTER 3",  # Typos + Series
        "Problems in Inorganic Chemistry", # Clean title
        "Balaji IN [ 4", # Fragmented OCR
        "GRB Mathematics for IIT-JEE", # Publisher + Title
    ]
    
    print("\n--- Running Tests ---\n")
    
    for query in test_cases:
        print(f"Query: '{query}'")
        try:
            result = await service.identify(query)
            if result:
                print(f"✅ Found: {result['title']} by {result['author']}")
                print(f"   ID: {result['book_id']}")
            else:
                print("❌ No match found")
        except Exception as e:
            print(f"⚠️ Error: {e}")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(main())
