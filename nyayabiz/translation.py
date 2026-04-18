# ─────────────────────────────────────────────────────────────────────────────
# MULTILINGUAL LAYER  (load once, reuse across queries)
# ─────────────────────────────────────────────────────────────────────────────

import re
from typing import Optional, Tuple

import torch
from langdetect import detect
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from nyayabiz.config import HF_TOKEN, IN2EN_MODEL, EN2IN_MODEL, DEVICE


LANG_MAP = {
    "hi": "hin_Deva", "mr": "mar_Deva", "gu": "guj_Gujr", "ta": "tam_Taml",
    "te": "tel_Telu", "bn": "ben_Beng", "kn": "kan_Knda", "pa": "pan_Guru",
    "ml": "mal_Mlym", "ur": "urd_Arab", "or": "ory_Orya", "as": "asm_Beng",
    "ne": "npi_Deva", "sa": "san_Deva", "sd": "snd_Arab", "ks": "kas_Arab",
    "mai": "mai_Deva", "brx": "brx_Deva", "doi": "doi_Deva", "kok": "gom_Deva",
    "sat": "sat_Olck", "fa": "urd_Arab",  "ar": "urd_Arab",
}


# ── Module-level state (initialized by load_translation_models) ──────────
_ip = None
_tok_in2en = None
_mdl_in2en = None
_tok_en2in = None
_mdl_en2in = None


def load_translation_models():
    """Load both Indic↔English translation models.

    Must be called once before translate_to_english / translate_to_indic.
    """
    global _ip, _tok_in2en, _mdl_in2en, _tok_en2in, _mdl_en2in

    from IndicTransToolkit.processor import IndicProcessor

    print(f"Loading translation models on {DEVICE} …")
    _ip = IndicProcessor(inference=True)

    _tok_in2en = AutoTokenizer.from_pretrained(IN2EN_MODEL, trust_remote_code=True, token=HF_TOKEN)
    _mdl_in2en = AutoModelForSeq2SeqLM.from_pretrained(IN2EN_MODEL, trust_remote_code=True, token=HF_TOKEN).to(DEVICE)

    _tok_en2in = AutoTokenizer.from_pretrained(EN2IN_MODEL, trust_remote_code=True, token=HF_TOKEN)
    _mdl_en2in = AutoModelForSeq2SeqLM.from_pretrained(EN2IN_MODEL, trust_remote_code=True, token=HF_TOKEN).to(DEVICE)

    print("Translation models ready.")


def detect_language(text: str) -> str:
    """Unicode-first Indian language detection. Returns AI4Bharat language code."""
    checks = [
        (r"[\u0B00-\u0B7F]", "ory_Orya"), (r"[\u0A80-\u0AFF]", "guj_Gujr"),
        (r"[\u0B80-\u0BFF]", "tam_Taml"), (r"[\u0C00-\u0C7F]", "tel_Telu"),
        (r"[\u0C80-\u0CFF]", "kan_Knda"), (r"[\u0D00-\u0D7F]", "mal_Mlym"),
        (r"[\u0A00-\u0A7F]", "pan_Guru"), (r"[\u0980-\u09FF]", "ben_Beng"),
    ]
    for pattern, code in checks:
        if re.search(pattern, text):
            return code
    if re.search(r"[\u0600-\u06FF]", text):
        return "snd_Arab" if re.search(r"[ڏڍڇڦڌٿڀٺڄڳ]", text) else "urd_Arab"
    if re.search(r"[\u0900-\u097F]", text):
        try:
            code = detect(text)
            if code in LANG_MAP:
                return LANG_MAP[code]
        except Exception:
            pass
        return "hin_Deva"
    return "eng_Latn"


def _translate(text: str, src_lang: str, tgt_lang: str, tokenizer, model) -> str:
    """Shared translation helper."""
    batch = _ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(batch, truncation=True, padding="longest",
                       return_tensors="pt", return_attention_mask=True).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, use_cache=True, min_length=0,
                             max_length=1024, num_beams=5, num_return_sequences=1)
    decoded = tokenizer.batch_decode(out.detach().cpu().tolist(),
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)
    # Explicit str() guard: some IndicTransToolkit versions return a pandas
    # TextAccessor instead of a plain string, which causes downstream TypeError.
    return str(_ip.postprocess_batch(decoded, lang=tgt_lang)[0])


def translate_to_english(text: str, src_lang: str) -> str:
    return _translate(text, src_lang, "eng_Latn", _tok_in2en, _mdl_in2en)


def translate_to_indic(text: str, tgt_lang: str) -> str:
    if tgt_lang == "eng_Latn":
        return text
    return _translate(text, "eng_Latn", tgt_lang, _tok_en2in, _mdl_en2in)
