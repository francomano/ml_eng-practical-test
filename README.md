# Progetto di Traduzione con Elasticsearch e Sentence Transformers

## Descrizione
Questo progetto utilizza **Elasticsearch** come motore di ricerca per archiviare frasi parallele e i loro vettori di embedding generati con il modello **Sentence-Transformer (all-MiniLM-L12-v2)**. L'applicazione permette di inserire coppie di frasi tradotte e di effettuare ricerche tramite query per ottenere un prompt di traduzione basato su esempi paralleli simili.

## Prerequisiti

1. **Docker**: Assicurati di avere **Docker** installato e in esecuzione sul tuo sistema.
2. **Python 3.8+**: Questo progetto è stato sviluppato utilizzando Python 3.8 o versioni successive.

## Installazione

1. **Clonare il repository**:

    ```bash
    git clone <url-del-repository>
    cd <nome-del-repository>
    ```

2. **Creare un ambiente virtuale** (opzionale ma consigliato):

    ```bash
    python -m venv venv
    source venv/bin/activate  # su Linux/macOS
    venv\Scripts\activate  # su Windows
    ```

3. **Installare le dipendenze**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Avviare Docker con Elasticsearch**:
   
   Assicurati che Docker sia installato e in esecuzione. Usa il seguente comando per avviare il container Elasticsearch come single node:

    ```bash
    docker run --name elasticsearch -d -p 9200:9200 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.10.0
    ```

## Comandi per Gestire Elasticsearch e Docker

### 1. **Eliminare tutti gli indici in Elasticsearch**

Per eliminare tutti gli indici nel tuo cluster Elasticsearch (attenzione, questo cancellerà permanentemente tutti i dati):

```bash
curl -X DELETE "http://localhost:9200/*"
```

## Comandi per Gestire Elasticsearch e Docker

### 2. **Fermare e Rimuovere il Container Docker**

- **Fermare il container**:

    Se il container di Elasticsearch è in esecuzione, fermalo con il comando:

    ```bash
    docker stop elasticsearch
    ```

- **Rimuovere il container**:

    Dopo aver fermato il container, puoi rimuoverlo con il comando:

    ```bash
    docker rm elasticsearch
    ```

### 3. **Rimuovere l'immagine di Elasticsearch (Opzionale)**

Se desideri rimuovere anche l'immagine di Elasticsearch per liberare spazio, esegui:

```bash
docker rmi docker.elastic.co/elasticsearch/elasticsearch:7.10.0
```
