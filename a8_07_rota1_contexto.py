"""
AULA 8 — Rota 1: jogar o arquivo no prompt

A rota mais simples que existe. Sempre comece por ela: se resolve, você
economizou uma semana de RAG.

Este script mostra os três limites, em ordem:
  1. a janela de 8.192 tokens
  2. o custo quadrático da atenção
  3. o desempenho degradando antes do limite (lost in the middle)
"""
# %%
import json
import time

import torch

from avaliacao import avaliar_rota
from config import DADOS, MODELO_CHAT
from utils import carregar, chat, medir, salvar_resultado

paginas = [json.loads(l) for l in
           (DADOS / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
modelo, tok = carregar(MODELO_CHAT)

# %% 1) O corpus cabe na janela?
corpus_inteiro = "\n\n".join(f"[{p['arquivo']}, p.{p['pagina']}]\n{p['texto']}"
                             for p in paginas)
n_tokens = len(tok(corpus_inteiro)["input_ids"])
janela = modelo.config.max_position_embeddings

print(f"Corpus completo : {n_tokens:,} tokens")
print(f"Janela do modelo: {janela:,} tokens")
print(f"Excedente       : {n_tokens / janela:.1f}x a janela")
print("\nNão cabe. Este é o slide 15 — e é uma vivência, não uma afirmação.")

# %% 2) O custo quadrático, medido
print("\nLatência de uma pergunta em função do tamanho do contexto:")
print(f"{'tokens':>8s} {'tempo(s)':>10s} {'VRAM(GB)':>10s}")
custos = []
for alvo in [512, 1024, 2048, 4096]:
    ids = tok(corpus_inteiro)["input_ids"][:alvo]
    trecho = tok.decode(ids)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    chat(modelo, tok, f"{trecho}\n\nResuma em uma frase.", max_new_tokens=40)
    dt = time.perf_counter() - t0
    vram = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    custos.append({"tokens": alvo, "s": round(dt, 2), "gb": round(vram, 2)})
    print(f"{alvo:>8d} {dt:>10.2f} {vram:>10.2f}")

print("""
LEITURA: dobrar o contexto não dobra o custo — cresce mais rápido, porque
a matriz de atenção é n x n (Aula 3, slide 18). Some a isso o cache KV,
que cresce linearmente com o contexto e ocupa VRAM que você não tem.
""")

# %% 3) Rodar o gabarito com UM documento no contexto
# Escolha o arquivo mais relevante; é o melhor caso possível desta rota.
arquivo_alvo = sorted({p["arquivo"] for p in paginas})[0]
doc = "\n\n".join(f"[{p['arquivo']}, p.{p['pagina']}]\n{p['texto']}"
                  for p in paginas if p["arquivo"] == arquivo_alvo)
doc_ids = tok(doc)["input_ids"][:janela - 800]     # deixa espaço p/ pergunta+resposta
doc = tok.decode(doc_ids)
print(f"\nUsando {arquivo_alvo} ({len(doc_ids):,} tokens no contexto)")

SISTEMA = ("Responda SOMENTE com base no material abaixo. Se a resposta não "
           "estiver nele, responda exatamente: 'Não encontrei essa informação "
           "no material.' Cite a fonte no formato [arquivo, p.N].")


def responder(pergunta: str) -> str:
    return chat(modelo, tok, f"{doc}\n\nPergunta: {pergunta}",
                sistema=SISTEMA, max_new_tokens=200)


with medir("gabarito rota 1"):
    resultado = avaliar_rota("Rota 1 — contexto", responder)

resultado["custos_contexto"] = custos
resultado["tokens_corpus"] = n_tokens
salvar_resultado("rota1_contexto", resultado)

print("""
O QUE ESPERAR: acerto bom nas perguntas cobertas pelo documento carregado,
zero nas demais, e abstenção razoável nas armadilhas. É uma boa rota para
UM documento — e só.
""")
