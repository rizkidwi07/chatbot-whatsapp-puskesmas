"""
app.py — Flask Webhook untuk Chatbot Puskesmas Cijagra Baru
Logika inferensi sama dengan fungsi chatbot() di notebook:
  1. Preprocess input user (6 tahap)
  2. SVM OvO → prediksi intent
  3. Filter df_train ke intent yang diprediksi
  4. Cosine similarity HANYA dalam intent tersebut
  5. Ambil jawaban dari utterance paling mirip
"""

import os
import pickle
import numpy as np
import requests
from flask import Flask, request
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess_text

app = Flask(__name__)

# Load bundle saat server start 
print('[app] Memuat chatbot_bundle.pkl ...')
with open('chatbot_bundle.pkl', 'rb') as f:
    bundle = pickle.load(f)

pipeline      = bundle['pipeline']
X_train_tfidf = bundle['X_train_tfidf']
df_train      = bundle['df_train']

tfidf_step = pipeline.named_steps['tfidf']
svm_step   = pipeline.named_steps['svm']
print(f'[app] Bundle dimuat. Total data training: {len(df_train)} baris.')

# Konfigurasi WhatsApp (Diisi dari Meta Developer Dashboard) 
WA_TOKEN     = os.environ.get('WA_TOKEN', '')
PHONE_ID     = os.environ.get('PHONE_ID', '')
VERIFY_TOKEN = os.environ.get('VERIFY_TOKEN', 'puskesmas_cijagra_2026')


# Fungsi Inferensi Chatbot (identik dengan notebook)
def get_chatbot_response(user_input: str) -> str:
    clean = preprocess_text(user_input)

    if not clean.strip():
        return '🙏 Maaf kak, pesan tidak dapat dipahami. Coba tulis ulang ya!'

    # Step 1: Prediksi intent dengan model SVM OvO
    predicted_intent = svm_step.predict(tfidf_step.transform([clean]))[0]

    # Step 2: Filter training set ke intent yang sama
    intent_mask = (df_train['intent'] == predicted_intent).values
    df_intent   = df_train[intent_mask].reset_index(drop=True)
    X_intent    = X_train_tfidf[intent_mask]

    if len(df_intent) == 0:
        return '🙏 Maaf kak, belum ada informasi untuk pertanyaan tersebut.'

    # Step 3: Cosine similarity HANYA dalam intent yang diprediksi
    q_vec = tfidf_step.transform([clean])
    sims  = cosine_similarity(q_vec, X_intent).flatten()
    idx   = int(np.argmax(sims))

    return df_intent['jawaban'].iloc[idx]


# Fungsi Kirim Pesan ke WhatsApp
def send_whatsapp(to_number: str, message: str):
    url = f"https://graph.facebook.com/v18.0/{PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WA_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message}
    }
    print("PHONE_ID =", PHONE_ID)
    print("TO =", to_number)
    resp = requests.post(url, headers=headers, json=payload, timeout=10)
    if resp.status_code != 200:
        print(f'[app] Gagal kirim WA: {resp.status_code} | {resp.text}')
    return resp


# Endpoint utama: health check
@app.route('/', methods=['GET'])
def index():
    return 'Chatbot Puskesmas Cijagra Baru — Server Aktif ✅', 200


# Webhook WhatsApp
@app.route('/webhook', methods=['GET', 'POST'])
def webhook():

    # Verifikasi webhook (dipanggil Meta sekali saat setup)
    if request.method == 'GET':
        mode      = request.args.get('hub.mode')
        token     = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')
        print(f'[app] DEBUG verifikasi -> mode={mode!r}, token_diterima={token!r}, token_diharapkan={VERIFY_TOKEN!r}')
        if mode == 'subscribe' and token == VERIFY_TOKEN:
            print('[app] Webhook berhasil diverifikasi Meta.')
            return challenge, 200
        return 'Forbidden', 403

    # Terima pesan masuk dari user
    if request.method == 'POST':
        data = request.get_json()
        try:
            entry    = data['entry'][0]['changes'][0]['value']
            messages = entry.get('messages', [])

            for msg in messages:
                if msg.get('type') == 'text':
                    user_no  = msg['from']
                    user_msg = msg['text']['body']

                    print(f'[app] Pesan masuk dari {user_no}: {user_msg}')
                    response = get_chatbot_response(user_msg)
                    print(f'[app] Respons: {response[:80]}...')

                    send_whatsapp(user_no, response)

        except Exception as e:
            print(f'[app] Error saat memproses pesan: {e}')

        return 'OK', 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)