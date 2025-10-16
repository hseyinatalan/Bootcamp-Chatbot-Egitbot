# 📘 EğitBot - RAG Tabanlı Eğitim Asistanı

EğitBot, Türkçe eğitim içerikleriyle eğitilmiş bir RAG (Retrieval-Augmented Generation) tabanlı akıllı eğitim asistanıdır. Öğrencilerin ders sorularını anlamlı, sade ve açıklayıcı şekilde yanıtlayarak öğrenme süreçlerine yardımcı olur.

Bu proje, LangChain, Google Gemini Pro, FAISS, Gradio ve çeşitli Türkçe eğitim veri setleri ile güçlendirilmiştir.

## 📚 Kullanılan Veri Setleri

Proje kapsamında aşağıdaki Türkçe veri setleri kullanılmıştır. Bu veri setleri, eğitim içeriklerinin zenginleştirilmesi ve sorulara doğru yanıtlar verebilmek için kullanılmıştır.

| Veri Seti                         | Açıklama                             | Hugging Face Linki                                                                                  |
|----------------------------------|------------------------------------|---------------------------------------------------------------------------------------------------|
| Orca Math Word Problems TR        | Türkçe matematik kelime problemleri| [duxx/orca-math-word-problems-tr](https://huggingface.co/datasets/duxx/orca-math-word-problems-tr) |
| Karayel-DDI Türkçe MATH-Hard     | Zorlayıcı Türkçe matematik soruları| [Karayel-DDI/Turkce_Lighteval_MATH-Hard](https://huggingface.co/datasets/Karayel-DDI/Turkce_Lighteval_MATH-Hard) |
| Korkmazemin1 Turkish Education Dataset | Genel Türkçe eğitim soruları ve cevapları | [korkmazemin1/turkish-education-dataset](https://huggingface.co/datasets/korkmazemin1/turkish-education-dataset?utm_source=chatgpt.com) |
| Musabg Wikipedia Turkish Summarization | Türkçe Wikipedia metin özetleri   | [musabg/wikipedia-tr-summarization](https://huggingface.co/datasets/musabg/wikipedia-tr-summarization) |

# 🚀 Özellikler

🔍 RAG tabanlı bilgi getirme (retrieval) desteği

🇹🇷 Türkçe eğitim verileriyle zenginleştirilmiş bilgi tabanı

🤖 Google Gemini 2.5 Pro (LLM) ile doğal dilde yanıt üretimi

🧠 FAISS vektör veritabanı ile hızlı sorgu yanıtı

📝 Soru geçmişi ve sohbet kaydetme özellikleri

💾 Sohbeti .txt dosyası olarak indirme

🎨 Kullanıcı dostu Gradio arayüzü

# 🧱 Kullanılan Kütüphaneler
gradio
python-dotenv
langchain
langchain-community
langchain-google-genai
langchain-huggingface
sentence-transformers
faiss-cpu
datasets==2.18.0
requests
torch
transformers
huggingface-hub
tiktoken

# 📦 Kurulum
1. Ortamı Hazırlama
```bash
git clone https://github.com/kullaniciAdi/egitbot-rag.git
cd egitbot-rag

python -m venv venv

# Sanal ortamı aktifleştir
# macOS / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

2. Gerekli Paketleri Yükleme
```bash
pip install -r requirements.txt
 ```
3. API Anahtarlarını Ayarlama
Aşağıdaki ortam değişkenlerini .env dosyası olarak veya terminalden tanımlayın:
```bash
export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
exportHUGGINGFACEHUB_API_TOKEN="YOUR_HUGGINGFACE_TOKEN"
```
