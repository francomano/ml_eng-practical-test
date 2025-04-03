# README

## Installazione

1. Installa le dipendenze:

   ```bash
   pip install fastapi uvicorn
   pip install elasticsearch==7.10.0

## Avvia Elasticsearch con Docker:

docker run --name elasticsearch -d -p 9200:9200 docker.elastic.co/elasticsearch/elasticsearch:7.10.0

## Avvia il server FastAPI:

uvicorn main:app --reload

# Scelte Tecniche

    Modello Embedding: all-MiniLM-L12-v2 (velocità e accuratezza).

    Metodologia: Similarità del coseno tra gli embedding per confrontare le traduzioni.

    Soglia: Similarità del 50% (min_score: 0.5).

    Lingua di Origine: Determinata dal campo source_language fornito dall'utente, per scegliere il vettore giusto.

# Struttura e Implementazione

    Elasticsearch: Utilizzato per archiviare e cercare traduzioni, ideale per scalabilità e prestazioni.

    Docker: Containerizza Elasticsearch per una configurazione semplice.

    FastAPI: Un'unica applicazione con due endpoint:

        POST /pairs: Aggiungi traduzioni.

        GET /prompt: Ottieni suggerimenti basati sulla query.
