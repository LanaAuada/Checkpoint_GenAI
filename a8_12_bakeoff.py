"""
AULA 8 — Bake-off: as cinco rotas no mesmo gabarito

Não roda nada novo: lê runs/resultados/*.json e monta a tabela comparativa
e o gráfico que fecham a aula.

Rode depois de a8_07 .. a8_11.
"""
# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RUNS
from utils import carregar_resultado

ROTAS = [
    ("rota1_contexto", "Rota 1\ncontexto"),
    ("rota2_rag", "Rota 2\nRAG"),
    ("rota3_sft", "Rota 3\nSFT"),
    ("rota4_dapt", "Rota 4\nDAPT"),
]

dados = {}
for chave, rotulo in ROTAS:
    r = carregar_resultado(chave)
    if r is None:
        print(f"[falta] {chave} — rode o script correspondente")
        continue
    dados[rotulo] = r.get("resumo") or r.get("depois") or {}

if not dados:
    raise SystemExit("Nenhum resultado encontrado em runs/resultados/.")

# %% Tabela
METRICAS = [
    ("acerto_total_%", "acerto geral"),
    ("acerto_factual_%", "acerto factual"),
    ("acerto_sintese_%", "acerto síntese"),
    ("abstencao_correta_%", "abstenção correta"),
    ("abstencao_indevida_%", "abstenção indevida"),
    ("citou_fonte_%", "citou fonte"),
    ("latencia_media_s", "latência (s)"),
]

largura = max(len(r) for r in dados) + 2
print(f"\n{'métrica':22s}" + "".join(f"{r.replace(chr(10),' '):>16s}" for r in dados))
print("-" * (22 + 16 * len(dados)))
for chave, rotulo in METRICAS:
    linha = f"{rotulo:22s}"
    for r in dados:
        linha += f"{dados[r].get(chave, float('nan')):>16.1f}"
    print(linha)

# %% Gráfico
fig, ax = plt.subplots(figsize=(9, 4.5))
rotulos = list(dados)
x = range(len(rotulos))
largura_barra = 0.26

series = [
    ("acerto_factual_%", "acerto factual", "#E11D48"),
    ("abstencao_correta_%", "abstenção correta", "#A5B4FC"),
    ("citou_fonte_%", "citou fonte", "#2DD4BF"),
]
for i, (chave, nome, cor) in enumerate(series):
    valores = [dados[r].get(chave, 0) for r in rotulos]
    ax.bar([p + (i - 1) * largura_barra for p in x], valores,
           largura_barra, label=nome, color=cor)

ax.set_xticks(list(x))
ax.set_xticklabels(rotulos, fontsize=9)
ax.set_ylabel("%")
ax.set_ylim(0, 105)
ax.set_title("Cinco rotas, o mesmo gabarito", fontsize=12)
ax.legend(frameon=False, fontsize=9, ncol=3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
destino = RUNS / "bakeoff.png"
fig.savefig(destino, dpi=160, bbox_inches="tight")
print(f"\nGráfico salvo em {destino}")

# %% A pergunta que fecha a aula
print("""
AS MÉTRICAS QUE NINGUÉM MEDE E QUE DECIDEM O PROJETO
----------------------------------------------------
  custo de ATUALIZAR quando o material mudar:
      contexto  -> trivial (troca o arquivo)
      RAG       -> reindexar, minutos
      SFT/DAPT  -> retreinar, horas, e revalidar tudo

  rastreabilidade: só contexto, RAG e ferramentas conseguem apontar
      a página que sustenta a resposta. SFT e DAPT não conseguem — e
      em auditoria isso sozinho elimina as duas.

ÁRVORE DE DECISÃO
-----------------
  Cabe na janela?               -> sim  -> CONTEXTO
  Precisa de fato verificável?  -> sim  -> RAG (+ ferramentas se for estruturado)
  O problema é tom/formato?     -> sim  -> SFT com LoRA
  Tem dezenas de milhões de
    tokens do domínio?          -> sim  -> DAPT
  caso contrário                       -> volte para RAG

A RECEITA QUE SE USA DE VERDADE
-------------------------------
Quase nenhum sistema em produção escolhe UMA rota:
      RAG para o fato
    + LoRA leve para a forma
    + ferramentas para o que é estruturado
E tudo isso roda numa RTX 3060 de 12 GB, offline, com os dados nunca
saindo da máquina.
""")
