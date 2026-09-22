import os
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

# Load environment variables
load_dotenv()

# get the environment variables for Supabase and OpenRouter
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize clients
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

def search_database(query_text, discipline=None):
    """Step 1: Turn user query into an embedding and search Supabase vector store"""
    response = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=query_text
    )
    query_embedding = response.data[0].embedding

    # Call our Supabase SQL function to find the most similar text chunks
    result = supabase.rpc(
        "match_course_documents",
        {
            "query_embedding": query_embedding,
            "filter_discipline": discipline,
            "match_count": 3  # Retrieve the top 3 most relevant paragraphs
        }
    ).execute()
    
    return result.data

def ask_lms_tutor(student_question):
    print(f"\n🎓 Student Question: '{student_question}'")
    print("🔍 Searching LMS course materials...")
    
    # Retrieve relevant context from Supabase
    matches = search_database(student_question)
    
    if not matches:
        return "I'm sorry, but I couldn't find any relevant materials in the curriculum to answer this."

    # Combine the retrieved text chunks into a single context window
    context_text = "\n\n---\n\n".join([match["content"] for match in matches])
    sources = set([match["metadata"]["source"] for match in matches])

    print(f"📖 Found {len(matches)} relevant course chunks. Generating AI response...")

    # Construct the prompt for the LLM with strict grounding instructions
    system_prompt = (
        "You are an AI learning assistant for a university LMS. "
        "Answer the student's question strictly using the provided course context below. "
        "Do not make up facts. If the answer cannot be found in the context, say 'I cannot find this in the current course syllabus.' "
        "Always cite your sources."
    )

    user_prompt = f"Context:\n{context_text}\n\nQuestion: {student_question}"

    # Call OpenRouter LLM (using a fast, capable model like GPT-4o-mini or Claude 3 Haiku)
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1 # Low temperature keeps it factual and reduces hallucination
    )

    answer = response.choices[0].message.content
    print("\n-----------------------------------------")
    print(f"🤖 LMS Tutor Response:\n{answer}")
    print(f"\n📚 Cited Sources: {list(sources)}")
    print("-----------------------------------------")

if __name__ == "__main__":
    # Test the RAG system 
    ask_lms_tutor("Petroleum reservoirs are broadly classified?")