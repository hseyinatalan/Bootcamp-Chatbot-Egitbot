📘 EğitBot - RAG Tabanlı Eğitim Asistanı

EğitBot, Türkçe eğitim içerikleriyle eğitilmiş bir RAG (Retrieval-Augmented Generation) tabanlı akıllı eğitim asistanıdır. Öğrencilerin ders sorularını anlamlı, sade ve açıklayıcı şekilde yanıtlayarak öğrenme süreçlerine yardımcı olur.

Bu proje, LangChain, Google Gemini Pro, FAISS, Gradio ve çeşitli Türkçe eğitim veri setleri ile güçlendirilmiştir.

🚀 Özellikler

🔍 RAG tabanlı bilgi getirme (retrieval) desteği

🇹🇷 Türkçe eğitim verileriyle zenginleştirilmiş bilgi tabanı

🤖 Google Gemini 2.5 Pro (LLM) ile doğal dilde yanıt üretimi

🧠 FAISS vektör veritabanı ile hızlı sorgu yanıtı

📝 Soru geçmişi ve sohbet kaydetme özellikleri

💾 Sohbeti .txt dosyası olarak indirme

🎨 Kullanıcı dostu Gradio arayüzü

🧱 Kullanılan Kütüphaneler
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

📦 Kurulum
1. Ortamı Hazırlama
<pre> ```bash # Projeyi klonla git clone https://github.com/kullaniciAdi/egitbot-rag.git cd egitbot-rag # Sanal ortam oluştur python -m venv venv # Sanal ortamı aktifleştir # macOS / Linux: source venv/bin/activate # Windows (CMD): venv\Scripts\activate # Windows (PowerShell): venv\Scripts\Activate.ps1 ``` </pre>
2. Gerekli Paketleri Yükleme
<pre> ```bash pip install -r requirements.txt ``` </pre>
3. API Anahtarlarını Ayarlama
Aşağıdaki ortam değişkenlerini .env dosyası olarak veya terminalden tanımlayın:
<pre> ```bash export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY" export HUGGINGFACEHUB_API_TOKEN="YOUR_HUGGINGFACE_TOKEN" ``` </pre>
