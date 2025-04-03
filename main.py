from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import json

es = Elasticsearch("http://localhost:9200")
index_name = "translations_with_vectors"

if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body={
        "mappings": {
            "properties": {
                "source_language": {"type": "keyword"},
                "target_language": {"type": "keyword"},
                "sentence": {"type": "text"},
                "translation": {"type": "text"},
                "sentence_vector_en": {"type": "dense_vector", "dims": 384},
                "sentence_vector_it": {"type": "dense_vector", "dims": 384} 
            }
        }
    })

model = SentenceTransformer('all-MiniLM-L12-v2')

def get_embedding(sentence: str):
    embedding = model.encode(sentence).tolist()
    if len(embedding) != 384:
        raise HTTPException(status_code=400, detail="Embedding length is not 384")
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
    Aggiungi una nuova coppia di traduzioni al database Elasticsearch.
    """
    vector_en = get_embedding(pair.sentence)
    vector_it = get_embedding(pair.translation)

    es.index(index=index_name, body={
        "source_language": pair.source_language,
        "target_language": pair.target_language,
        "sentence": pair.sentence,
        "translation": pair.translation,
        "sentence_vector_en": vector_en,
        "sentence_vector_it": vector_it
    })

    return {"status": "ok"}


@app.get("/prompt")
def get_translation_prompt(source_language: str, target_language: str, query_sentence: str):
    """
    Restituisci il prompt di traduzione basato sulla query e sulle traduzioni esistenti.
    """
    query_vector = get_embedding(query_sentence)

    # Scegliamo il campo dei vettori e la lingua di origine in base alla lingua di input
    if source_language == "it":
        # Se la lingua di origine è italiano
        vector_field = "sentence_vector_it"
        source_field = "translation"  # La frase originale sarà in italiano
        target_field = "sentence"  # La traduzione sarà in inglese
    else:
        # Se la lingua di origine è inglese
        vector_field = "sentence_vector_en"
        source_field = "sentence"  # La frase originale sarà in inglese
        target_field = "translation"  # La traduzione sarà in italiano

    # Creiamo la query basata solo sulla similarità del coseno
    query = {
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": f"cosineSimilarity(params.query_vector, doc['{vector_field}']) + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        },
        "size": 4,  # Restituiamo i primi 4 risultati più simili
        "min_score": 0.5  # Impostiamo una soglia di similarità (facoltativo)
    }

    # Eseguiamo la ricerca su Elasticsearch
    res = es.search(index=index_name, body=query)

    # Estraiamo le traduzioni dai risultati
    suggestions = [hit["_source"][target_field] for hit in res["hits"]["hits"]]

    # Se non ci sono suggerimenti, restituiamo un messaggio di errore
    if not suggestions:
        return {"prompt": f"No similar sentences found for {query_sentence}."}

    # Restituiamo i suggerimenti di traduzione
    return {"prompt": f"Context: {query_sentence}. Suggested translations: {', '.join(suggestions)}"}
