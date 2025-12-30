from transformers import pipeline
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name)

SLANG_DICT = {
    "yg": "yang", "gak": "tidak", "ga": "tidak", "bgs": "bagus",
    "sy": "saya", "bgt": "banget", "sdh": "sudah", "krn": "karena",
    "mantul": "mantap", "recomended": "rekomen", "kecewa": "buruk"
}

# Lexicon sederhana (contoh, sebaiknya diperluas dari data kalian)
POS_WORDS = {"bagus", "mantap", "mantul", "suka", "puas", "cepat", "sesuai", "rekomen", "oke", "top"}
NEG_WORDS = {"buruk", "jelek", "parah", "rusak", "palsu", "penipu", "kecewa", "lama", "zonk", "refund"}

NEGATION = {"tidak", "gak", "ga", "nggak", "bukan"}
INTENSIFIERS = {"banget", "sekali", "parah", "super", "bgt"}

STRONG_NEG_PHRASES = {"barang palsu", "tidak sesuai", "ga sesuai", "gak sesuai", "penipu", "minta refund", "uang kembali"}
STRONG_POS_PHRASES = {"sesuai deskripsi", "cepat sampai", "pengiriman cepat"}

class SentimentAnalyzer:
    def init(self):
        logger.info("Memuat model IndoRoBERTa Sentiment...")
        self.pipeline = pipeline(
            "sentiment-analysis",
            model="w11wo/indonesian-roberta-base-sentiment-classifier",
            tokenizer="w11wo/indonesian-roberta-base-sentiment-classifier"
        )

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        # normalisasi slang per kata
        words = [SLANG_DICT.get(w, w) for w in text.split()]
        text = " ".join(words)
        # buang URL dll (contoh sederhana)
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        # pertahankan spasi
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def rule_score(self, text):
        """
        Return skor rule di rentang [-1, +1]
        """
        if not text:
            return 0.0

        # phrase override kuat
        for p in STRONG_NEG_PHRASES:
            if p in text:
                return -1.0
        for p in STRONG_POS_PHRASES:
            if p in text:
                return 1.0

        tokens = text.split()
        score = 0.0
        i = 0
        while i < len(tokens):
            w = tokens[i]

            # negasi: "tidak bagus" => negatif
            if w in NEGATION and i + 1 < len(tokens):
                nxt = tokens[i + 1]
                if nxt in POS_WORDS:
                    score -= 1.0
                    i += 2
                    continue
                if nxt in NEG_WORDS:
                    score += 0.8  # "tidak jelek" cenderung positif/lebih netral
                    i += 2
                    continue

            if w in POS_WORDS:
                score += 1.0
            elif w in NEG_WORDS:
                score -= 1.0

            # intensifier: kata sebelumnya diperkuat
            if w in INTENSIFIERS and i > 0:
                prev = tokens[i - 1]
                if prev in POS_WORDS:
                    score += 0.5
                elif prev in NEG_WORDS:
                    score -= 0.5

            i += 1

        # normalisasi kasar biar masuk [-1, 1]
        if score > 2: score = 2
        if score < -2: score = -2
        return score / 2.0

    def predict(self, text, alpha=0.75, conf_gate=0.60):
        clean = self.clean_text(text)
        if not clean:
            return "Netral", 0.0, {"mode": "empty"}

        # ---- MODEL (utama) ----
        # Catatan: truncation token-based lebih aman daripada clean[:512]
        result = self.pipeline(clean, truncation=True, max_length=512)[0]
        label = result["label"]
        score = float(result["score"])

        # map label (sesuaikan kalau label model ternyata LABEL_0 dkk)
        label_map = {"positive": "Positif", "neutral": "Netral", "negative": "Negatif"}
        model_sent = label_map.get(label, label)
        # ubah ke skor bertanda: +score/-score/0
        if model_sent == "Positif":
            model_signed = score
        elif model_sent == "Negatif":
            model_signed = -score
        else:
            model_signed = 0.0

        # ---- RULE ----
        rscore = self.rule_score(clean)

        # ---- HYBRID STRATEGY ----
        # 1) kalau model yakin, pakai model
        if score >= conf_gate:
            final_signed = model_signed
            mode = "model_confident"
        else:
            # 2) kalau ragu, ensemble weighted
            final_signed = alpha * model_signed + (1 - alpha) * rscore
            mode = "hybrid_ensemble"

        # final label dari skor
        if final_signed > 0.15:
            final_label = "Positif"
        elif final_signed < -0.15:
            final_label = "Negatif"
        else:
            final_label = "Netral"

        return final_label, abs(final_signed), {
            "mode": mode,
            "model": {"label": model_sent, "score": score},
            "rule_score": rscore,
            "final_signed": final_signed
        }
