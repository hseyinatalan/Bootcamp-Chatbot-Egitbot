# RAG Temelli Eğitim Chatbot (EğitBot)

# Gerekli kütüphaneleri ekliyoruz.
import os
import requests
import streamlit as st
from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import datetime

# Streamlit secrets üzerinden API anahtarını al
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
MODEL_NAME = "models/gemini-2.5-pro"

# -----------------------------
# 📄 VERİ SETİ & VEKTÖR VERİTABANI HAZIRLAMA
# -----------------------------
# Veri hazırlama işlemi de sadece bir kez yapılır.
# Burda 4 farklı veri seti ekliyoruz.
# Veri setlerinden math_word problem tarzı sorular için
                 # math_hard daha derin işlemler için
                 # edu eğitim temalı genel soru-cevap için
                 # wiki_sum bu veri setide tarih ve fen alanında daha verimli cevaplar için kullanılmıştır.
@st.cache_resource  
@st.cache_resource
def prepare_retriever():
    try:
        dataset_math_word = load_dataset("duxx/orca-math-word-problems-tr", split="train")
        dataset_math_hard = load_dataset("Karayel-DDI/Turkce_Lighteval_MATH-Hard", split="train")
        dataset_edu = load_dataset("korkmazemin1/turkish-education-dataset", split="train")
        dataset_wiki_sum = load_dataset("musabg/wikipedia-tr-summarization", split="train")

        documents = []
         # 1. Orca Math Word Problems
        for item in dataset_math_word:
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            if question and answer:
                documents.append(f"Soru: {question}\nCevap: {answer}")

        # 2. Karayel-DDI Math Hard
        for item in dataset_math_hard:
            question = item.get("question", "").strip()
            answer = item.get("solution", "").strip()
            if question and answer:
                documents.append(f"Soru: {question}\nCevap: {answer}")

        # 3. Korkmazemin1 Turkish Education Dataset
        for item in dataset_edu:
            question = item.get("soru", "").strip()
            answer = item.get("cevap", "").strip()
            if question and answer:
                documents.append(f"Soru: {question}\nCevap: {answer}")

         # 4. Musabg Wikipedia Turkish Summarization Dataset
        for item in dataset_wiki_sum:
            text = item.get("text", "").strip()
            summary = item.get("summary", "").strip()
            if text and summary:
                documents.append(f"Metin: {text}\nÖzet: {summary}")

         # Metinleri parçalara ayıralım
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100, length_function=len)
        docs = text_splitter.create_documents(documents)

        # Embedding modeli
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # FAISS dizini
        FAISS_PATH = "faiss_index"

        # Daha önce kayıtlı FAISS varsa onu yükle
        if os.path.exists(FAISS_PATH):
            vectorstore = FAISS.load_local(FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        else:
            vectorstore = FAISS.from_documents(docs, embedding_model)
            vectorstore.save_local(FAISS_PATH)

        return vectorstore.as_retriever(search_kwargs={"k": 3})

    except Exception as e:
        st.error(f"Veri seti hazırlanırken hata oluştu: {e}")
        return None
# Eğer retriever None dönerse uygulamayı durdurabilirsin
retriever = prepare_retriever()
if retriever is None:
    st.stop()

# -----------------------------
# 🔗 ÖZEL PROMPT OLUŞTURMA
# -----------------------------
# Burada modelden gelen bilgiyi nasıl kullanacağını söylüyoruz.
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

# -----------------------------
# 🔗 Gemini LLM (LangChain üzerinden)
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GOOGLE_API_KEY
)

# -----------------------------
# 🔗 LangChain QA Zinciri Kurulumu
# -----------------------------
# LLM ve retriever’ı bağlayarak "Soru-Cevap" zinciri oluşturuyoruz.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT}
)

# -----------------------------
# 🖥️ Streamlit Arayüzü (EğitBot)
# -----------------------------
# Sayfa başlığı, simgesi ve genişlik ayarlandı
st.set_page_config(page_title="📘 EğitBot - Eğitim Asistanı", page_icon="🎓", layout="wide")

# -----------------------------
# 👤 Oturum Durumu: Sohbet Geçmişi ve İstatistikler Başlatma
# -----------------------------
# Streamlit session_state ile kalıcı sohbet geçmişi ve sayaçları tutuyoruz.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "total_questions" not in st.session_state:
    st.session_state.total_questions = 0

if "total_answers" not in st.session_state:
    st.session_state.total_answers = 0

# Örnek sorular veri yapısı
EXAMPLE_QUESTIONS = {
    "İlkokul": {
        "Matematik": [
            "Bir çantada 5 kitap varsa, 3 çantada kaç kitap olur?",
            "10 - 4 işleminin sonucu kaçtır?",
            "2 ile 3'ün toplamı kaçtır?",
            "Bir elma 3 TL ise, 4 elma kaç TL eder?",
            "5 elma ve 2 armut kaç meyvedir?"
        ],
        "Türkçe": [
            "'Ev' kelimesi kaç harflidir?",
            "Cümlenin baş harfi nasıl yazılır?",
            "Bir cümleye örnek veriniz.",
            "Sesli harfler nelerdir?",
            "Hangi kelimede 'a' harfi vardır?"
        ],
        "Tarih": [
            "Türkiye'nin başkenti neresidir?",
            "Atatürk kimdir?",
            "Türkiye hangi kıtadadır?",
            "Bayrak neden önemlidir?",
            "Okulun ilk günü nasıl geçer?"
        ],
        "Fen Bilimleri": [
            "Bitkiler nasıl büyür?",
            "Güneş neden parlar?",
            "Su neden akar?",
            "Hayvanlar ne yer?",
            "Hangi nesne sıcak olur?"
        ]
    },
    "Ortaokul": {
        "Matematik": [
            "3x + 5 = 20 denkleminde x kaçtır?",
            "Bir dik üçgenin özellikleri nelerdir?",
            "5 ile 7'nin ortalaması kaçtır?",
            "Bir sayının karesi nedir?",
            "Alan ve çevre farkı nedir?"
        ],
        "Türkçe": [
            "Cümledeki yüklem nedir?",
            "Fiil nedir, örnek veriniz.",
            "Anlam bilgisi nedir?",
            "Noktalama işaretleri nelerdir?",
            "Cümlenin ögesi nedir?"
        ],
        "Tarih": [
            "Osmanlı Devleti ne zaman kurulmuştur?",
            "Türk tarihi önemli olayları nelerdir?",
            "Cumhuriyet ne zaman ilan edildi?",
            "Atatürk'ün hayatı hakkında bilgi verin.",
            "Tarih neden önemlidir?"
        ],
        "Fen Bilimleri": [
            "Fotosentez nedir?",
            "Maddenin halleri nelerdir?",
            "İnsan vücudundaki organlar nelerdir?",
            "Güç nedir?",
            "Enerji türleri nelerdir?"
        ]
    },
    "Lise": {
        "Matematik": [
            "Türev nedir ve nasıl hesaplanır?",
            "Bir üçgenin iç açılarının toplamı kaçtır?",
            "Kareköklü ifadeler nasıl sadeleştirilir?",
            "Fonksiyon nedir?",
            "Olasılık hesaplaması nasıl yapılır?"
        ],
        "Türkçe": [
            "Fiil çekimi nasıl yapılır?",
            "Anlatım bozukluğu nedir, örnek veriniz.",
            "Cümledeki ögeler nelerdir?",
            "Sözcük türleri nelerdir?",
            "Metin türleri nelerdir?"
        ],
        "Tarih": [
            "Birinci Dünya Savaşı'nın nedenleri nelerdir?",
            "Osmanlı Devleti'nin yıkılış süreci nasıl oldu?",
            "Cumhuriyet'in ilanı ne zaman gerçekleşti?",
            "Atatürk'ün inkılapları nelerdir?",
            "Soğuk Savaş dönemi hakkında bilgi verin."
        ],
        "Fen Bilimleri": [
            "Kimyasal reaksiyon nedir?",
            "Elektrik akımı nasıl oluşur?",
            "Fotosentezde ışığın rolü nedir?",
            "Biyoloji alanında DNA'nın önemi nedir?",
            "Newton'un hareket kanunları nelerdir?"
        ]
    }
}

# -----------------------------
# 📚 Sidebar - İstatistikler, Butonlar ve Örnek Sorular
# -----------------------------
with st.sidebar:
    st.title("📚 Örnek Sorular & Kontroller")

    # İstatistikler
    st.markdown(f"- Toplam Sorulan Soru: **{st.session_state.get('total_questions', 0)}**")
    st.markdown(f"- Toplam Alınan Cevap: **{st.session_state.get('total_answers', 0)}**")
    st.markdown("---")

    # "Geçmişi Temizle" butonu, tıklanınca tüm sohbeti ve sayacı sıfırlıyor
    if st.button("♻️ Geçmişi Temizle", key="clear_history"):
        st.session_state.chat_history = []
        st.session_state.total_questions = 0
        st.session_state.total_answers = 0

    # Sohbet geçmişini dosya olarak kaydetmek için fonksiyon
    def save_chat_history():
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"chat_history_{now}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            for sender, msg in st.session_state.get("chat_history", []):
                f.write(f"{'Kullanıcı' if sender == 'user' else 'Bot'}: {msg}\n")
        return filename
    
    # "Sohbeti Kaydet" butonuna basıldığında sohbet dosyasını oluşturup indirilebilir yapıyoruz
    if st.button("💾 Sohbeti Kaydet", key="save_chat"):
        filename = save_chat_history()
        with open(filename, "rb") as f:
            st.download_button(
                label="⬇️ Dosyayı İndir",
                data=f,
                file_name=filename,
                mime="text/plain"
            )

    # Örnek sorular için seçimler
    grade = st.selectbox("Sınıf Seviyesi Seçiniz:", options=list(EXAMPLE_QUESTIONS.keys()), key="grade_select")
    subjects = list(EXAMPLE_QUESTIONS[grade].keys())
    subject = st.selectbox("Konu Seçiniz:", options=subjects, key="subject_select")

    # Liste olarak örnek soruları göster
    example_questions = EXAMPLE_QUESTIONS[grade][subject]
    st.markdown("### Örnek Sorular:")
    for idx, question in enumerate(example_questions, 1):
        st.markdown(f"{idx}. {question}")

# Ana sayfa başlığı
st.title("📘 EğitBot - Eğitim Asistanı")

# Kullanıcının sorusunu al
user_question = st.text_input("Sorunuzu buraya yazınız veya yukarıdaki örnek sorulardan birini yazabilirsiniz:", key="user_question_input")

# Gönder butonu
if st.button("Gönder", key="send_question"):
    if user_question.strip() == "":
        st.warning("Lütfen bir soru yazınız veya örnek sorulardan birini kullanınız.")
    else:
        with st.spinner("Cevap aranıyor..."):
            try:
                answer = qa_chain.run(user_question)
                st.session_state.chat_history.append(("user", user_question))
                st.session_state.chat_history.append(("bot", answer))
                st.session_state.total_questions += 1
                st.session_state.total_answers += 1
            except Exception as e:
                st.error(f"Cevap alınırken hata oluştu: {e}")

# Sohbet geçmişi balonlar halinde gösterimi
def render_message(sender, message):
    if sender == "user":
        color = "#DCF8C6"  # Açık yeşil (Kullanıcı için)
        align = "flex-end"
        border_radius = "15px 15px 0 15px"
    else:
        color = "#EAEAEA"  # Açık gri (Bot için)
        align = "flex-start"
        border_radius = "15px 15px 15px 0"
    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: {align};
            margin: 5px 0;
        ">
            <div style="
                background-color: {color};
                padding: 10px 15px;
                border-radius: {border_radius};
                max-width: 70%;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                ">
                {message}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("💬 Sohbet Geçmişi")
    for sender, message in reversed(st.session_state.chat_history):
        if sender == "user":
            st.markdown(
                f"""
                <div style="
                    background-color: #A3C4F3;  /* Soft mavi */
                    padding: 12px;
                    border-radius: 15px;
                    margin: 5px 0px;
                    max-width: 80%;
                    color: black;   /* Yazı siyah */
                    ">
                    <b>Kullanıcı:</b> {message}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="
                    background-color: #8FBC8F;  /* Soft yeşil */
                    padding: 12px;
                    border-radius: 15px;
                    margin: 5px 0px;
                    max-width: 80%;
                    color: black;  /* Yazı siyah */
                    ">
                    <b>Bot:</b> {message}
                </div>
                """,
                unsafe_allow_html=True
            )
