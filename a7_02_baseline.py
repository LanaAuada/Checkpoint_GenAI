"""
AULA 7 — Passo 2: a linha de base

REGRA DA DISCIPLINA: nada de treinar antes de medir.
Sem estes números, qualquer resultado depois é opinião.

Este script guarda em runs/resultados/baseline.json:
  - gerações de referência (EN, PT, e fora do domínio)
  - perplexidade em 4 conjuntos held-out distintos
"""
# %%
from config import MODELO_BASE
from textos import (HELDOUT_DOMINIO_EN, HELDOUT_DOMINIO_PT, HELDOUT_EN,
                    HELDOUT_PT, PROMPTS_EN, PROMPTS_GERAL_EN, PROMPTS_PT)
from utils import (carregar, fixar_seed, gerar, medir, perplexidade,
                   salvar_resultado)

fixar_seed()
modelo, tok = carregar(MODELO_BASE)

# %% 1) Gerações de referência (greedy => reprodutível)
def bloco_geracoes(modelo, tok, titulo, prompts):
    print(f"\n### {titulo}")
    saidas = {}
    for p in prompts:
        texto = gerar(modelo, tok, p, max_new_tokens=40, temperatura=0.0)
        saidas[p] = texto
        print(f"  > {p}")
        print(f"    {texto[len(p):].strip()}")
    return saidas


geracoes = {
    "en": bloco_geracoes(modelo, tok, "Inglês (domínio do treino)", PROMPTS_EN),
    "pt": bloco_geracoes(modelo, tok, "Português", PROMPTS_PT),
    "geral": bloco_geracoes(modelo, tok, "Fora do domínio", PROMPTS_GERAL_EN),
}

# OBSERVE a saída em português. É comum o modelo escorregar para o espanhol:
# em ~2 trilhões de tokens majoritariamente ingleses, o espanhol é o vizinho
# mais provável do português no espaço de embeddings. Não é bug — é o corpus.

# %% 2) Perplexidade em 4 conjuntos held-out
conjuntos = {
    "heldout_en": HELDOUT_EN,               # fora do domínio, inglês
    "heldout_pt": HELDOUT_PT,               # fora do domínio, português
    "dominio_en": HELDOUT_DOMINIO_EN,       # tema de IA, inglês, não visto
    "dominio_pt": HELDOUT_DOMINIO_PT,       # tema de IA, português, não visto
}

with medir("perplexidade base"):
    ppl = {nome: perplexidade(modelo, tok, textos) for nome, textos in conjuntos.items()}

print("\n### Perplexidade do modelo BASE (menor = melhor)")
for nome, valor in ppl.items():
    print(f"  {nome:12s} {valor:8.2f}")

print("""
LEITURA: a perplexidade em português é várias vezes maior que em inglês.
Esse número é a linha de base contra a qual todo treino desta aula será
comparado. Guarde-o.
""")

# %% 3) Salvar
salvar_resultado("baseline", {
    "modelo": MODELO_BASE,
    "perplexidade": ppl,
    "geracoes": geracoes,
})
