"""
preprocess.py
Fungsi preprocessing — HARUS identik dengan yang dipakai saat training
di notebook (6 tahap: case folding, cleaning, normalisasi, tokenisasi,
stopword removal, stemming).

Modul ini memuat kamus slangword, stopword, dan stemmer SEKALI saat
server start (bukan setiap request) supaya respons chatbot cepat.
"""

import re
import urllib.request
import nltk

# ── Auto-download NLTK punkt (server cloud tidak punya ini secara default) ──
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary
from Sastrawi.Stemmer.Stemmer import Stemmer

# ── 1. Korpus Slangword (KBBA) ───────────────────────────────────────────
SLANG_URL = 'https://raw.githubusercontent.com/ramaprakoso/analisis-sentimen/master/kamus/kbba.txt'
slang_dict = {}
try:
    with urllib.request.urlopen(SLANG_URL, timeout=10) as r:
        for line in r.read().decode('utf-8').splitlines():
            parts = line.strip().split('\t')
            if len(parts) == 2:
                slang_dict[parts[0].lower()] = parts[1].lower()
    print(f'[preprocess] Korpus slangword dimuat: {len(slang_dict):,} entri')
except Exception as e:
    print(f'[preprocess] Gagal load slangword online ({e}), pakai kamus minimal.')
    slang_dict = {
        'ga': 'tidak', 'gak': 'tidak', 'gk': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak',
        'yg': 'yang', 'dg': 'dengan', 'utk': 'untuk', 'dgn': 'dengan', 'krn': 'karena',
        'brp': 'berapa', 'brapa': 'berapa', 'gmn': 'bagaimana', 'gimana': 'bagaimana',
        'blm': 'belum', 'udh': 'sudah', 'udah': 'sudah', 'sdh': 'sudah',
        'mw': 'mau', 'bs': 'bisa', 'sm': 'sama', 'dr': 'dari', 'pd': 'pada',
        'klo': 'kalau', 'kalo': 'kalau', 'hrs': 'harus', 'hr': 'hari',
        'bsk': 'besok', 'skrg': 'sekarang', 'jm': 'jam', 'wkt': 'waktu',
        'smpe': 'sampai', 'sampe': 'sampai', 'lg': 'lagi', 'tp': 'tapi',
        'ttg': 'tentang', 'pk': 'pukul', 'pkm': 'puskesmas', 'pskesmas': 'puskesmas',
        'bkanya': 'bukanya', 'bka': 'buka', 'kak': 'kakak', 'bang': 'abang',
        'aja': 'saja', 'aj': 'saja', 'doang': 'saja', 'jg': 'juga', 'pls': 'please',
    }

# ── 2. Korpus Stopword Bahasa Indonesia ──────────────────────────────────
SW_URL = 'https://raw.githubusercontent.com/yasirutomo/python-sentianalysis-id/master/data/feature_list/stopwordsID.txt'
stopwords_id = set()
try:
    with urllib.request.urlopen(SW_URL, timeout=10) as r:
        for line in r.read().decode('utf-8').splitlines():
            w = line.strip().lower()
            if w:
                stopwords_id.add(w)
    print(f'[preprocess] Korpus stopword dimuat: {len(stopwords_id):,} kata')
except Exception as e:
    print(f'[preprocess] Gagal load stopword online ({e}), pakai NLTK.')
    from nltk.corpus import stopwords
    stopwords_id = set(stopwords.words('indonesian'))

# Kata domain kesehatan yang TIDAK boleh dihapus (sama seperti notebook)
DOMAIN_KEEP = {
    'poli', 'puskesmas', 'dokter', 'buka', 'tutup', 'jam', 'hari',
    'jadwal', 'daftar', 'syarat', 'periksa', 'berobat', 'konseling',
    'senin', 'selasa', 'rabu', 'kamis', 'jumat', 'sabtu', 'minggu',
    'bpjs', 'ktp', 'kk', 'gratis', 'biaya', 'rawat', 'jalan',
    'imunisasi', 'vaksin', 'hamil', 'bayi', 'anak', 'gigi', 'umum',
    'lab', 'farmasi', 'obat', 'sakit', 'demam', 'batuk', 'pilek',
    'mual', 'diare', 'pusing', 'gatal', 'luka', 'nyeri'
}
stopwords_id = stopwords_id - DOMAIN_KEEP

# ── 3. Stemmer PySastrawi dengan kamus kustom (sama seperti notebook) ───
factory = StemmerFactory()
base_words = factory.get_words()
kata_medis_tambahan = [
    'poli', 'bpjs', 'mtbs', 'kia', 'kb', 'gizi',
    'catin', 'puskesmas', 'imunisasi', 'vaksin'
]
base_words.extend(kata_medis_tambahan)
custom_dictionary = ArrayDictionary(base_words)
stemmer = Stemmer(custom_dictionary)

print('[preprocess] Stemmer PySastrawi kustom siap digunakan.')


# ── 4. Fungsi Preprocessing 6 Tahap (identik dengan notebook) ───────────
def step1_case_folding(text: str) -> str:
    return text.lower()


def step2_cleaning(text: str) -> str:
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    emoji_pat = re.compile(
        '[\U00010000-\U0010ffff\U0001F600-\U0001F64F'
        '\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        '\U0001F1E0-\U0001F1FF]+', flags=re.UNICODE)
    return emoji_pat.sub(' ', text).strip()


def step3_normalisasi(text: str) -> str:
    return ' '.join(slang_dict.get(w, w) for w in text.split())


def step4_tokenisasi(text: str) -> list:
    return word_tokenize(text)


def step5_stopword_removal(tokens: list) -> list:
    return [w for w in tokens if w not in stopwords_id and len(w) > 1]


def step6_stemming(tokens: list) -> str:
    return ' '.join(stemmer.stem(w) for w in tokens)


def preprocess_text(text: str) -> str:
    """Pipeline preprocessing lengkap 6 tahap — dipanggil dari app.py."""
    t = step1_case_folding(text)
    t = step2_cleaning(t)
    t = step3_normalisasi(t)
    tokens = step4_tokenisasi(t)
    tokens = step5_stopword_removal(tokens)
    return step6_stemming(tokens)
