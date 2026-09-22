import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize clients
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Configure the Streamlit page layout
st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AI Learning Assistant")
st.markdown("Ask questions about your course materials. The AI will retrieve the exact curriculum sections to answer accurately.")

# Initialize chat history in session state so it remembers conversation turns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages from history when the app reruns
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input from the bottom chat box
if prompt := st.chat_input("Ask a question about the course syllabus..."):
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching LMS course materials..."):
            try:
                #  Generate query embedding
                embedding_response = client.embeddings.create(
                    model="openai/text-embedding-3-small",
                    input=prompt
                )
                query_embedding = embedding_response.data[0].embedding

                #  Search Supabase vector store
                result = supabase.rpc(
                    "match_course_documents",
                    {
                        "query_embedding": query_embedding,
                        "filter_discipline": None,
                        "match_count": 3
                    }
                ).execute()
                
                matches = result.data

                if not matches:
                    response_text = "I'm sorry, but I couldn't find any relevant materials in the curriculum to answer this."
                else:
                    # Compile context and sources
                    context_text = "\n\n---\n\n".join([m["content"] for m in matches])
                    sources = set([m["metadata"]["source"] for m in matches])

                    system_prompt = (
                        "You are an AI learning assistant for a university LMS. "
                        "Answer the student's question strictly using the provided course context below. "
                        "Do not make up facts. If the answer cannot be found in the context, say 'I cannot find this in the current course syllabus.' "
                        "Always cite your sources."
                    )
                    user_prompt = f"Context:\n{context_text}\n\nQuestion: {prompt}"

                    # Call OpenRouter LLM
                    completion = client.chat.completions.create(
                        model="openai/gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1
                    )
                    
                    answer = completion.choices[0].message.content
                    response_text = f"{answer}\n\n📚 *Cited Sources: {list(sources)}*"

            except Exception as e:
                response_text = f"⚠️ An error occurred while generating the response: {e}"

            st.markdown(response_text)
            # Save assistant response to session state history
            st.session_state.messages.append({"role": "assistant", "content": response_text})