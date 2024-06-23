from unidecode import unidecode
import re

from pandarallel import pandarallel
import pandas as pd
import spacy


pandarallel.initialize(progress_bar=True)

nlp_es = spacy.load("es_core_news_lg")
nlp_en = spacy.load("en_core_web_lg")


def preprocess(row):
    text, lang = row["review_es"].lower(), row["lang"]

    text_cleaned = re.sub(r"""(\w+)([,.¿?¡!;:()'"]+)(\w+)""", r"\1 \2 \3", text)
    text_cleaned = unidecode(text_cleaned)

    nlp = nlp_es if lang == "es" else nlp_en
    doc = nlp(text_cleaned)

    tokens = []

    for token in doc:
        if (not token.like_email and 
            not token.like_num and
            not token.like_url and 
            not token.is_stop and 
            not token.is_punct and
                token.is_alpha):
            tokens.append(token.lemma_)

    row["text_cleaned"] = " ".join(tokens)

    return row


df = pd.read_csv("train.csv", index_col=0)
df = df.parallel_apply(preprocess, axis=1)
df.to_csv("train-preprocesado.csv")

df = pd.read_csv("test.csv", index_col=0)
df = df.parallel_apply(preprocess, axis=1)
df.to_csv("test-preprocesado.csv")