import os
import datetime
import gradio as gr
from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Cache klasör ayarları
os.environ["HF_HOME"] = "./cache"
os.environ["HF_DATASETS_CACHE"] = "./cache/hf_datasets"
os.environ["TRANSFORMERS_CACHE"] = "./cache/transformers"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./cache/sentence_transformers"

# API anahtarları ortam değişkenlerinden alınmalı
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
MODEL_NAME = "models/gemini-2.5-pro"

# Veri hazırlama fonksiyonu (aynı)
def prepare_retriever():
    dataset_math_word = load_dataset("duxx/orca-math-word-problems-tr", split="train[:2000]")
    dataset_math_hard = load_dataset("Karayel-DDI/Turkce_Lighteval_MATH-Hard", split="train[:2000]")
    dataset_edu = load_dataset("korkmazemin1/turkish-education-dataset", split="train[:2000]")
    dataset_wiki_sum = load_dataset("musabg/wikipedia-tr-summarization", split="train[:2000]")

    documents = []
    for item in dataset_math_word:
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        if question and answer:
            documents.append(f"Soru: {question}\nCevap: {answer}")

    for item in dataset_math_hard:
        question = item.get("question", "").strip()
        answer = item.get("solution", "").strip()
        if question and answer:
            documents.append(f"Soru: {question}\nCevap: {answer}")

    for item in dataset_edu:
        question = item.get("soru", "").strip()
        answer = item.get("cevap", "").strip()
        if question and answer:
            documents.append(f"Soru: {question}\nCevap: {answer}")

    for item in dataset_wiki_sum:
        text = item.get("text", "").strip()
        summary = item.get("summary", "").strip()
        if text and summary:
            documents.append(f"Metin: {text}\nÖzet: {summary}")

    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    FAISS_PATH = "faiss_index"
    if os.path.exists(FAISS_PATH):
        vectorstore = FAISS.load_local(FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(docs, embedding_model)
        vectorstore.save_local(FAISS_PATH)

    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = prepare_retriever()

prompt_template = """
Sadece sorulan soruya net ve kısa cevap ver. Gereksiz ek açıklama yapma. 
Sadece yukarıdaki soruya cevap ver. Başka konulara girmeyin veya yeni sorular sormayın.

Bilgiler:
{context}

Soru:
{question}

Cevap:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GOOGLE_API_KEY
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT}
)

# Gradio UI düzeni
with gr.Blocks() as demo:
    gr.Markdown("# 📘 EğitBot - Eğitim Asistanı")
    
    with gr.Row():
        with gr.Column(scale=2):
            # Soru sorma alanı
            user_input = gr.Textbox(placeholder="Sorunuzu buraya yazınız...", label="Soru")
            send_btn = gr.Button("Gönder")

            # Toplam soru sayısı
            total_q = gr.Number(value=0, label="Toplam Sorulan Soru", interactive=False)

            # Sohbet geçmişi
            chatbox = gr.HTML(value="", label="Sohbet Geçmişi")

        with gr.Column(scale=1):
            # Sohbeti kaydetme ve temizleme butonları
            clear_btn = gr.Button("♻️ Geçmişi Temizle", variant="primary")
            save_btn = gr.Button("💾 Sohbeti Kaydet", variant="primary")
            download_file = gr.File(label="İndir")

    # Sohbeti temizleme fonksiyonu
    def clear_chat():
        return "", 0  # Sohbeti temizle ve soruları sıfırla

    clear_btn.click(fn=clear_chat, inputs=[], outputs=[chatbox, total_q])

    # Sohbeti kaydetme fonksiyonu
    def save_chat_to_file(chat_history):
        if not chat_history:
            return None
        
        # Chat geçmişini bir dosyaya kaydet
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.txt"
        
        with open(filename, "w") as file:
            file.write(chat_history)
        
        # Kullanıcıya dosya indirme bağlantısı gönder
        return filename

    save_btn.click(fn=save_chat_to_file, inputs=[chatbox], outputs=download_file)

    # Soru gönderme fonksiyonu
    def handle_question(user_question, total_questions, chat_history):
        # Soruyu kullanarak yanıt al
        result = qa_chain.run(user_question)
        
        # Geçmişi güncelle ve toplam soru sayısını arttır
        chat_history = f"""
        <div style="background-color:#e8f4f8; padding: 10px; border-radius: 15px; margin-bottom: 5px;">
            <b>Soru:</b> <span style="color: black;">{user_question}</span>
        </div>
        <div style="background-color:#f1f9f5; padding: 10px; border-radius: 15px; margin-bottom: 5px;">
            <b>Cevap:</b> <span style="color: black;">{result}</span>
        </div>
        """ + chat_history  # Yeni soru-cevap en üstte

        total_questions += 1
        return chat_history, total_questions

    # Gönder butonuna tıklanınca soru gönderme işlemi
    send_btn.click(fn=handle_question, inputs=[user_input, total_q, chatbox], 
                   outputs=[chatbox, total_q])

# Gradio UI'sini başlatma
demo.launch(share=True)
