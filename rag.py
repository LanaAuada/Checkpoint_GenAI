"""
Módulo de RAG local — usado pela Rota 2 e pelo bake-off.

Nada de framework: ~120 linhas mostram que RAG é recuperação + prompt.
LangChain e LlamaIndex resolvem escala e integração, não o conceito.
"""
from __future__ import annotations

import json
import pickle

import numpy as np

from config import DADOS, MODELO_EMB, RUNS, TOP_K

INDICE = RUNS / "rag"
INDICE.mkdir(parents=True, exist_ok=True)


def carregar_chunks() -> list[dict]:
    arquivo = DADOS / "chunks.jsonl"
    if not arquivo.exists():
        raise SystemExit("Rode prep_corpus.py primeiro.")
    return [json.loads(l) for l in arquivo.read_text(encoding="utf-8").splitlines()]


def carregar_encoder():
    from sentence_transformers import SentenceTransformer

    from config import DEVICE
    # POR QUE NÃO O SmolLM2 AQUI: ele é um modelo de GERAÇÃO, treinado para
    # prever o próximo token. Busca semântica pede um modelo treinado com
    # objetivo contrastivo. E o corpus é em português: precisa ser multilíngue.
    return SentenceTransformer(MODELO_EMB, device=DEVICE)


def construir_indice():
    import faiss

    chunks = carregar_chunks()
    enc = carregar_encoder()

    # Modelos da família E5 exigem os prefixos "passage: " e "query: ".
    textos = [f"passage: {c['texto']}" for c in chunks]
    vetores = enc.encode(textos, batch_size=64, normalize_embeddings=True,
                         show_progress_bar=True).astype("float32")

    # Vetores normalizados => produto interno == similaridade de cosseno.
    # IndexFlatIP é busca EXATA. Para milhares de chunks é o suficiente e
    # não tem hiperparâmetro para errar. HNSW/IVF só a partir de milhões.
    indice = faiss.IndexFlatIP(vetores.shape[1])
    indice.add(vetores)

    faiss.write_index(indice, str(INDICE / "faiss.index"))
    (INDICE / "meta.pkl").write_bytes(pickle.dumps(chunks))
    print(f"Índice com {indice.ntotal} vetores de dimensão {vetores.shape[1]}")
    return indice, chunks


def carregar_indice():
    import faiss

    caminho = INDICE / "faiss.index"
    if not caminho.exists():
        return construir_indice()
    indice = faiss.read_index(str(caminho))
    chunks = pickle.loads((INDICE / "meta.pkl").read_bytes())
    return indice, chunks


def buscar(pergunta: str, indice, chunks, enc, k: int = TOP_K) -> list[dict]:
    qv = enc.encode([f"query: {pergunta}"], normalize_embeddings=True).astype("float32")
    scores, idx = indice.search(qv, k)
    saida = []
    for score, i in zip(scores[0], idx[0]):
        c = dict(chunks[i])
        c["score"] = float(score)
        saida.append(c)
    return saida


SISTEMA = (
    "Você é um monitor da disciplina de IA Generativa para Engenharia. "
    "Responda SOMENTE com base nos trechos fornecidos. "
    "Se a resposta não estiver neles, responda exatamente: "
    "'Não encontrei essa informação no material.' "
    "Sempre cite a fonte no formato [arquivo, p.N]. Seja conciso."
)


def montar_prompt(pergunta: str, trechos: list[dict]) -> str:
    contexto = "\n\n---\n\n".join(
        f"[{t['arquivo']}, p.{t['pagina']}]\n{t['texto']}" for t in trechos
    )
    return f"Trechos do material:\n\n{contexto}\n\nPergunta: {pergunta}"


def responder(pergunta, modelo, tok, indice, chunks, enc, k: int = TOP_K):
    from utils import chat

    trechos = buscar(pergunta, indice, chunks, enc, k)
    resposta = chat(modelo, tok, montar_prompt(pergunta, trechos),
                    sistema=SISTEMA, max_new_tokens=220)
    return resposta, trechos


# ------------------------------------------------------------------ BM25
def construir_bm25(chunks):
    """Busca lexical: resolve siglas, códigos e nomes próprios, onde o
    vetorial costuma falhar. Combinada com o denso, é o 'híbrido'."""
    from rank_bm25 import BM25Okapi

    corpus = [c["texto"].lower().split() for c in chunks]
    return BM25Okapi(corpus)


def buscar_hibrido(pergunta, indice, chunks, enc, bm25, k=TOP_K, peso_denso=0.6):
    denso = buscar(pergunta, indice, chunks, enc, k=k * 3)
    scores_bm = bm25.get_scores(pergunta.lower().split())
    maximo = max(scores_bm.max(), 1e-9)

    combinado = {}
    for t in denso:
        combinado[t["id"]] = peso_denso * t["score"]
    for i, s in enumerate(scores_bm):
        if s > 0:
            combinado[i] = combinado.get(i, 0.0) + (1 - peso_denso) * (s / maximo)

    melhores = sorted(combinado.items(), key=lambda kv: -kv[1])[:k]
    return [dict(chunks[i], score=s) for i, s in melhores]
