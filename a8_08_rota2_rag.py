"""
AULA 8 — Rota 2: RAG local sobre os PDFs da disciplina

Se o corpus não cabe na janela, não coloque o corpus.
Coloque os 4 pedaços certos.

Pré-requisito: python prep_corpus.py
"""
# %% 1) Construir o índice
from config import MODELO_CHAT, MODELO_EMB, TOP_K
from rag import (buscar, buscar_hibrido, carregar_encoder, carregar_indice,
                 construir_bm25, montar_prompt, responder, SISTEMA)
from utils import carregar, chat, medir, salvar_resultado

with medir("indexação"):
    indice, chunks = carregar_indice()
enc = carregar_encoder()

print(f"{len(chunks)} chunks indexados com {MODELO_EMB}")
print(f"Dimensão dos vetores: {indice.d}")

# %% 2) Ver a RECUPERAÇÃO isolada, antes de envolver o LLM
# Este é o passo que quase todo mundo pula — e é onde o RAG falha.
consultas = [
    "o que é esquecimento catastrófico",
    "fórmula da atenção",
    "quantos parâmetros tem o modelo",
]
for q in consultas:
    print(f"\n### {q}")
    for t in buscar(q, indice, chunks, enc, k=3):
        print(f"  {t['score']:.3f}  [{t['arquivo']}, p.{t['pagina']}]  "
              f"{t['texto'][:90].replace(chr(10), ' ')}...")

print("""
PERGUNTA PARA A TURMA: o trecho certo apareceu no top-3?
Se não apareceu, NENHUM modelo de linguagem salva a resposta. A métrica
que importa aqui chama-se recall@k, e ela é do RETRIEVER, não do LLM.
""")

# %% 3) O pipeline completo
modelo, tok = carregar(MODELO_CHAT)

pergunta = "O que é esquecimento catastrófico e como reduzi-lo?"
resposta, trechos = responder(pergunta, modelo, tok, indice, chunks, enc)
print(f"\nP: {pergunta}\nR: {resposta}")
print("\nTrechos usados:")
for t in trechos:
    print(f"  - [{t['arquivo']}, p.{t['pagina']}] score={t['score']:.3f}")

# %% 4) O elo fraco — demonstração deliberada de falha
# Pergunte algo cujo termo não aparece literalmente no material.
pergunta_ruim = "Qual o consumo de energia elétrica do datacenter da FIAP?"
resp_ruim, trechos_ruim = responder(pergunta_ruim, modelo, tok, indice, chunks, enc)
print(f"\nP: {pergunta_ruim}\nR: {resp_ruim}")
print("Trechos recuperados (irrelevantes):")
for t in trechos_ruim:
    print(f"  - [{t['arquivo']}, p.{t['pagina']}] score={t['score']:.3f}")
print("""
Observe o score baixo. Uma salvaguarda barata e eficaz: se o melhor score
for menor que um limiar (~0.75 para o E5), NEM CHAME o LLM — responda
'não encontrei'. Recuperação ruim com resposta fluente é pior que silêncio.
""")

# %% 5) Efeito do tamanho do chunk e do k
from avaliacao import avaliar_rota

for k in [2, 4, 8]:
    print(f"\n===== TOP_K = {k} =====")
    avaliar_rota(f"RAG k={k}",
                 lambda p, k=k: responder(p, modelo, tok, indice, chunks, enc, k)[0],
                 verboso=False)

print("""
k pequeno: perde o trecho certo. k grande: dilui o contexto e o modelo
'se perde no meio' (Aula 3, slide 36). Não existe k universal — existe o
k que o SEU gabarito escolhe.
""")

# %% 6) Híbrido: denso + BM25
try:
    bm25 = construir_bm25(chunks)

    def responder_hibrido(p):
        trechos = buscar_hibrido(p, indice, chunks, enc, bm25, k=TOP_K)
        return chat(modelo, tok, montar_prompt(p, trechos),
                    sistema=SISTEMA, max_new_tokens=220)

    resultado_hibrido = avaliar_rota("RAG híbrido (denso+BM25)", responder_hibrido,
                                     verboso=False)
except ImportError:
    print("rank_bm25 não instalado — pule (pip install rank-bm25)")
    resultado_hibrido = None

# %% 7) Rodar o gabarito oficial e salvar
with medir("gabarito rota 2"):
    resultado = avaliar_rota("Rota 2 — RAG",
                             lambda p: responder(p, modelo, tok, indice, chunks, enc)[0])

if resultado_hibrido:
    resultado["hibrido"] = resultado_hibrido["resumo"]
salvar_resultado("rota2_rag", resultado)

print("""
PRÓXIMAS ALAVANCAS, em ordem de custo/benefício:
  1. híbrido BM25 + denso   -> resolve siglas, códigos, nomes próprios
  2. reranker cross-encoder -> recupera 20, reordena para 4
     (BAAI/bge-reranker-base, ~1 GB, roda na 3060)
  3. limiar de score        -> abstém em vez de alucinar
  4. citação verificada     -> checar programaticamente se o trecho citado existe
""")
