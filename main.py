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
    
    if source_language == "it":
        vector_field = "sentence_vector_it"
        source_field = "sentence" 
        target_field = "translation"  
        
        # Invertiamo solo se la lingua di origine è italiano
        match_source_language = {"match": {"source_language": source_language}}
        match_target_language = {"match": {"target_language": target_language}}

    else:
        vector_field = "sentence_vector_en"
        source_field = "sentence"
        target_field = "translation" 
        
        match_source_language = {"match": {"source_language": source_language}}
        match_target_language = {"match": {"target_language": target_language}}

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
    "size": 4,
    "min_score": 0.5
}


    res = es.search(index=index_name, body=query)

    suggestions = [hit["_source"]["translation"] for hit in res["hits"]["hits"]]

    if not suggestions:
        return {"prompt": f"No similar sentences found for {query_sentence}."}

    return {"prompt": f"Context: {query_sentence}. Suggested translations: {', '.join(suggestions)}"}
