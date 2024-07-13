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

    num_neg, num_adj_neg, num_exclm = 0, 0, len(re.findall(r"[¡!]", text))
    text_cleaned = re.sub(r"""(\w+)([,.¿?¡!;:()'"]+)(\w+)""", r"\1 \2 \3", text)
    text_cleaned = unidecode(text_cleaned)

    nlp = nlp_es if lang == "es" else nlp_en
    doc = nlp(text_cleaned)

    tokens, tokens_pos = [], []

    negating = False
    current_entity_name = ""

    for token in doc:
        if token.text == "no":
            num_neg += 1
            negating = True
            continue
        elif token.is_punct:
            negating = False
            continue

        if token.ent_type:
            if token.ent_iob_ == "B":
                current_entity_name = token.text
            elif token.ent_iob_ == "I":
                current_entity_name = f"{current_entity_name}_{token.text}"
        elif (not token.like_email and 
                not token.like_num and
                not token.like_url and 
                not token.is_stop and
                token.is_alpha and
                token.pos):
            if len(current_entity_name) > 0:
                tokens.append(current_entity_name)
                current_entity_name = ""

            token_text = f"NO_{token.lemma_}" if negating else token.lemma_
            if negating and token.pos_ == "ADJ":
                num_adj_neg += 1

            tokens.append(token_text)
            tokens_pos.append(f"{token_text}_{token.pos_}")

    row["text_cleaned"] = " ".join(tokens)
    row["text_cleaned_pos"] = " ".join(tokens_pos)
    row["num_neg"] = num_neg
    row["num_adj_neg"] = num_adj_neg
    row["num_exclm"] = num_exclm

    return row


df = pd.read_csv("train.csv", index_col=0)
df = df.parallel_apply(preprocess, axis=1)
df.to_csv("train-preprocesado.csv")

df = pd.read_csv("test.csv", index_col=0)
df = df.parallel_apply(preprocess, axis=1)
df.to_csv("test-preprocesado.csv")