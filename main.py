from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import json

es = Elasticsearch("http://localhost:9200")
index_name = "translations_with_vectors"

model = SentenceTransformer('all-MiniLM-L12-v2')

# Indice generalizzato ma compatibile con i nomi dei tuoi JSON
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body={
        "mappings": {
            "properties": {
                "source_language": {"type": "keyword"},
                "target_language": {"type": "keyword"},
                "sentence": {"type": "text"},       # frase sorgente
                "translation": {"type": "text"},    # frase tradotta
                "sentence_vector": {"type": "dense_vector", "dims": 384}  # embedding della frase sorgente
            }
        }
    })

def get_embedding(sentence: str):
    embedding = model.encode(sentence).tolist()
    if len(embedding) != 384:
        raise HTTPException(status_code=400, detail="Embedding length mismatch")
    return embedding

app = FastAPI()

class TranslationPair(BaseModel):
    source_language: str
    target_language: str
    sentence: str
    translation: str

class TranslationRequest(BaseModel):
    source_language: str
    target_language: str
    query_sentence: str

@app.post("/pairs")
def add_translation(pair: TranslationPair):
    """
    Inserisce una nuova coppia nel database.
    """
    vector = get_embedding(pair.sentence)

    es.index(index=index_name, body={
        "source_language": pair.source_language,
        "target_language": pair.target_language,
        "sentence": pair.sentence,
        "translation": pair.translation,
        "sentence_vector": vector
    })
    return {"status": "ok"}

@app.get("/prompt")
def get_translation_prompt(source_language: str, target_language: str, query_sentence: str):
    """
    Restituisce un prompt di traduzione basato su frasi parallele simili.
    """
    query_vector = get_embedding(query_sentence)

    query = {
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"source_language": source_language}},
                            {"term": {"target_language": target_language}}
                        ]
                    }
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, doc['sentence_vector'])",
                    "params": {"query_vector": query_vector}
                }
            }
        },
        "size": 4
    }

    res = es.search(index=index_name, body=query)

    hits = res["hits"]["hits"]
    if not hits:
        return {"prompt": f"No similar sentences found for '{query_sentence}'."}

    examples = "\n".join([
        f"{hit['_source']['sentence']} -> {hit['_source']['translation']}" for hit in hits
    ])

    prompt = (
        f"These are example translations from {source_language} to {target_language}:\n"
        f"{examples}\n\n"
        f"Translate the following sentence:\n{query_sentence}"
    )

    return {"prompt": prompt}

