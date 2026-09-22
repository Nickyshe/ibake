import os
import time
import pypdf
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from supabase import create_client

# Load secret keys from our .env file
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize Supabase and OpenRouter (via OpenAI SDK)
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

EMBEDDING_MODEL = "openai/text-embedding-3-small"
BATCH_SIZE = 50  # Process 50 chunks at a time


def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])
    return chunks


def process_pdf(file_path, discipline_name):
    print(f"\n Processing: {file_path}")
    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        return

    reader = pypdf.PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    chunks = chunk_text(full_text)
    print(f"Sliced into {len(chunks)} chunks. Uploading in batches...")

    # Process in batches
    total_batches = (len(chunks) - 1) // BATCH_SIZE + 1
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i : i + BATCH_SIZE]

        try:
            #  Generate embeddings in bulk using 'client'
            response = client.embeddings.create(
                model=EMBEDDING_MODEL, input=batch_chunks
            )
            embeddings = [item.embedding for item in response.data]

            #  Prepare rows
            rows_to_insert = []
            for chunk, embedding in zip(batch_chunks, embeddings):
                rows_to_insert.append({
                    "content": chunk,
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "discipline": discipline_name,
                    },
                    "embedding": embedding,
                })

            #  Bulk insert batch into Supabase
            supabase.table("course_documents").insert(rows_to_insert).execute()
            print(f"Uploaded batch {i // BATCH_SIZE + 1} / {total_batches}")

        except Exception as e:
            print(f" Error on batch starting at index {i}: {e}")
            break

    print(f" Finished uploading {os.path.basename(file_path)}!")


if __name__ == "__main__":
    process_pdf("curriculum_docs/OSHA3088.pdf", "Health and Safety")
    process_pdf("curriculum_docs/reservoir_engineering.pdf", "Petroleum Engineering")
    print("\n All files processed successfully!")