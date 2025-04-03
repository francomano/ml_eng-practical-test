from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import json

# Inizializzazione di Elasticsearch
es = Elasticsearch("http://localhost:9200")
index_name = "translations_with_vectors"

# Creazione dell'indice se non esiste
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body={
        "mappings": {
            "properties": {
                "source_language": {"type": "keyword"},
                "target_language": {"type": "keyword"},
                "sentence": {"type": "text"},
                "translation": {"type": "text"},
                "sentence_vector": {"type": "dense_vector", "dims": 384}
            }
        }
    })

# Inizializzazione del modello di SentenceTransformer
model = SentenceTransformer('all-MiniLM-L12-v2')

# Funzione per ottenere il vettore (embedding) della frase
def get_embedding(sentence: str):
    embedding = model.encode(sentence).tolist()
    if len(embedding) != 384:
        raise HTTPException(status_code=400, detail="Embedding length is not 384")
    return embedding

# Creazione dell'app FastAPI
app = FastAPI()

# Modello Pydantic per la coppia di traduzioni
class TranslationPair(BaseModel):
    source_language: str
    target_language: str
    sentence: str
    translation: str

# Modello Pydantic per la richiesta di traduzione
class TranslationRequest(BaseModel):
    source_language: str
    target_language: str
    query_sentence: str

# Endpoint per aggiungere una coppia di traduzioni
@app.post("/pairs")
def add_translation(pair: TranslationPair):
    """
    Aggiungi una nuova coppia di traduzioni al database Elasticsearch.
    """
    # Ottieni il vettore della frase
    sentence_vector = get_embedding(pair.sentence)
    
    # Tentiamo di indicizzare il documento su Elasticsearch
    es.index(index=index_name, body={
        "source_language": pair.source_language,
        "target_language": pair.target_language,
        "sentence": pair.sentence,
        "translation": pair.translation,
        "sentence_vector": sentence_vector
    })

    # Risposta "ok" al client
    return {"status": "ok"}

# Endpoint per ottenere suggerimenti di traduzione
@app.get("/prompt")
def get_translation_prompt(source_language: str, target_language: str, query_sentence: str):
    """
    Restituisci il prompt di traduzione basato sulla query e sulle traduzioni esistenti.
    """
    # Ottieni il vettore della query
    query_vector = get_embedding(query_sentence)
    
    # Crea la query Elasticsearch per il calcolo della similarità coseno
    query = {
        "query": {
            "bool": {
                "must": [
                    {"match": {"source_language": source_language}},
                    {"match": {"target_language": target_language}}
                ],
                "should": [
                    {
                        "script_score": {
                            "query": {
                                "match_all": {}
                            },
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, doc['sentence_vector']) + 1.0",
                                "params": {"query_vector": query_vector}
                            }
                        }
                    }
                ]
            }
        },
        "size": 4,  # Restituiamo 4 risultati come richiesto
        "min_score": 0.3
    }

    # Esegui la query di ricerca
    res = es.search(index=index_name, body=query)

    # Estrai le traduzioni dai risultati
    suggestions = [hit["_source"]["translation"] for hit in res["hits"]["hits"]]

    # Se non ci sono suggerimenti, restituiamo un messaggio che lo indica
    if not suggestions:
        return {"prompt": "No similar sentences found for" f" {query_sentence}."}

    # Restituiamo la risposta con le traduzioni suggerite
    return {"prompt": f"Context: {query_sentence}. Suggested translations: {', '.join(suggestions)}"}
