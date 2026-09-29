"""
02 - Dataset de beleza / skincare (pedido -> resposta)
======================================================
Mesma estrutura do 02_dados_new.py, com outro tema. Gera exemplos como:

    ### Pedido:
    Qual hidratante você recomenda para pele oleosa?
    ### Resposta:
    Para pele oleosa, recomendo gel-creme oil-free com niacinamida. ...

Os campos JSON mantêm os nomes do projeto original (tarefa, codigo, testes)
para que o 04_treinar.py funcione quase sem alteração:
  tarefa = o pedido | codigo = a resposta esperada | testes = palavras-chave

Avaliação: a resposta do modelo "passa" se contiver TODAS as palavras-chave.

Saída:
  dados/treino.jsonl
  dados/teste.jsonl  -> nunca visto no treino:
      "variacao" = mesmo assunto, mas o pedido foi escrito de outro jeito
      "inedita"  = assunto que o modelo nunca viu (acne, manchas, linhas finas)

Rodar:  python 02_dados_beleza.py
"""
import json
import os
import random
from pathlib import Path

try:
    os.chdir(Path(__file__).resolve().parent)
except NameError:
    pass

random.seed(42)

# ---------------------------------------------------------------- conhecimento
# kw_* = trecho que a resposta precisa conter para contar como acerto.
PELES = {
    "oleosa": dict(
        limpeza="gel de limpeza com ácido salicílico", kw_lim="ácido salicílico",
        hidratante="gel-creme oil-free com niacinamida", kw_hid="oil-free",
        protetor="protetor solar em gel oil-free FPS 50", kw_prot="em gel",
        ativo="niacinamida",
        why="ela ajuda a equilibrar a oleosidade e a refinar os poros"),
    "seca": dict(
        limpeza="sabonete líquido cremoso sem sulfato", kw_lim="sem sulfato",
        hidratante="creme rico com ceramidas", kw_hid="ceramidas",
        protetor="protetor solar cremoso hidratante FPS 50", kw_prot="cremoso hidratante",
        ativo="ácido hialurônico",
        why="ele retém água na pele e reduz a sensação de repuxar"),
    "mista": dict(
        limpeza="espuma de limpeza suave", kw_lim="espuma",
        hidratante="emulsão hidratante leve", kw_hid="emulsão",
        protetor="protetor solar fluido de toque seco FPS 50", kw_prot="toque seco",
        ativo="niacinamida",
        why="ela equilibra a zona T sem ressecar as bochechas"),
    "sensível": dict(
        limpeza="sabonete líquido sem perfume", kw_lim="sem perfume",
        hidratante="creme calmante com pantenol", kw_hid="calmante",
        protetor="protetor solar mineral com óxido de zinco FPS 50", kw_prot="mineral",
        ativo="centella asiática",
        why="ela acalma a pele e reduz a vermelhidão"),
    "normal": dict(
        limpeza="gel de limpeza suave", kw_lim="gel de limpeza suave",
        hidratante="loção hidratante leve", kw_hid="loção hidratante",
        protetor="protetor solar em loção FPS 50", kw_prot="em loção",
        ativo="vitamina C",
        why="ela uniformiza o tom e ajuda a proteger a pele"),
}

# Como o usuário pode descrever o tipo de pele
DESCRICOES = ["pele {tipo}", "uma pele {tipo}", "quem tem pele {tipo}"]

FAMILIAS = {
    "hidratante": dict(
        pedidos=["Qual hidratante você recomenda para {p}?",
                 "Preciso de um hidratante para {p}. O que devo usar?",
                 "Indique um produto hidratante para {p}.",
                 "Que creme hidratante é bom para {p}?"],
        resposta="Para pele {t}, recomendo {hidratante}. Use de manhã e à noite.",
        chaves=["kw_hid"]),
    "limpeza": dict(
        pedidos=["Qual limpador facial você indica para {p}?",
                 "Preciso de um sabonete facial para {p}.",
                 "Recomende um produto de limpeza para {p}.",
                 "Qual a melhor forma de limpar o rosto para {p}?"],
        resposta="Para pele {t}, use {limpeza}, uma ou duas vezes ao dia, sem esfregar.",
        chaves=["kw_lim"]),
    "protetor": dict(
        pedidos=["Qual protetor solar você recomenda para {p}?",
                 "Indique um filtro solar para {p}.",
                 "Preciso de protetor solar para {p}.",
                 "Que protetor solar é bom para {p}?"],
        resposta="Para pele {t}, escolha {protetor}. Reaplique a cada 2 a 3 horas se ficar no sol.",
        chaves=["kw_prot"]),
    "ativo": dict(
        pedidos=["Qual ingrediente ativo é bom para {p}?",
                 "Que ativo devo procurar nos produtos para {p}?",
                 "Indique um ativo para {p}.",
                 "Qual ativo você recomenda para {p}?"],
        resposta="O ativo indicado para pele {t} é {ativo}: {why}.",
        chaves=["ativo"]),
    "rotina_manha": dict(
        pedidos=["Monte uma rotina de skincare para a manhã para {p}.",
                 "Qual rotina matinal você indica para {p}?",
                 "Crie minha rotina de manhã para {p}.",
                 "Me ajude com uma rotina de manhã para {p}."],
        resposta="Rotina da manhã para pele {t}: 1) {limpeza}; 2) {hidratante}; 3) {protetor}.",
        chaves=["kw_lim", "kw_hid", "kw_prot"]),
    "rotina_noite": dict(
        pedidos=["Monte uma rotina de skincare para a noite para {p}.",
                 "Qual rotina noturna você indica para {p}?",
                 "Crie minha rotina de noite para {p}.",
                 "Me ajude com uma rotina de noite para {p}."],
        resposta="Rotina da noite para pele {t}: 1) {limpeza}; 2) sérum com {ativo}; 3) {hidratante}.",
        chaves=["kw_lim", "ativo", "kw_hid"]),
}

# ---- Assuntos que NUNCA aparecem no treino (teste de generalização) ----
PEDIDOS_PROBLEMA = ["Qual rotina você indica para {n}?",
                    "Tenho {n}. O que devo usar?",
                    "Que produto ajuda com {n}?",
                    "Me recomende algo para {n}."]
PROBLEMAS = {
    "acne": dict(
        nomes=["acne", "espinhas"],
        resposta="Para {n}, use gel de limpeza com ácido salicílico, hidratante oil-free "
                 "e protetor solar sem óleo. Não esprema as espinhas.",
        chaves=["ácido salicílico", "oil-free"]),
    "manchas": dict(
        nomes=["manchas", "manchas no rosto"],
        resposta="Para {n}, use vitamina C de manhã, protetor solar FPS 50 todos os dias "
                 "e ácido azelaico à noite.",
        chaves=["vitamina C", "FPS 50"]),
    "linhas_finas": dict(
        nomes=["linhas finas", "rugas finas"],
        resposta="Para {n}, use retinol em baixa concentração à noite, hidratante com "
                 "peptídeos e protetor solar todos os dias.",
        chaves=["retinol", "peptídeos"]),
}


def exemplo(familia, pedido, resposta, chaves, tipo):
    return {"familia": familia, "tarefa": pedido, "codigo": resposta,
            "testes": chaves, "tipo": tipo}


treino, teste = [], []

for familia, fam in FAMILIAS.items():
    for tipo, d in PELES.items():
        resposta = fam["resposta"].format(t=tipo, **d)
        chaves = [d[c] for c in fam["chaves"]]
        combos = [(t, p) for t in fam["pedidos"] for p in DESCRICOES]
        random.shuffle(combos)
        for i, (t, p) in enumerate(combos):
            pedido = t.format(p=p.format(tipo=tipo))
            ex = exemplo(f"{familia}/{tipo}", pedido, resposta, chaves,
                         "variacao" if i == 0 else "treino")
            (teste if i == 0 else treino).append(ex)

for problema, pr in PROBLEMAS.items():
    combos = [(t, n) for t in PEDIDOS_PROBLEMA for n in pr["nomes"]]
    random.shuffle(combos)
    for t, n in combos[:4]:
        teste.append(exemplo(problema, t.format(n=n), pr["resposta"].format(n=n),
                             pr["chaves"], "inedita"))

random.shuffle(treino)
Path("dados").mkdir(exist_ok=True)
for nome, dados in [("treino", treino), ("teste", teste)]:
    with open(f"dados/{nome}.jsonl", "w", encoding="utf-8") as arq:
        for ex in dados:
            arq.write(json.dumps(ex, ensure_ascii=False) + "\n")

n_var = sum(e["tipo"] == "variacao" for e in teste)
print(f"Treino: {len(treino)} exemplos")
print(f"Teste:  {len(teste)} exemplos ({n_var} variações + {len(teste) - n_var} inéditas)")
print("\nExemplo de treino:\n")
print(f"### Pedido:\n{treino[0]['tarefa']}\n### Resposta:\n{treino[0]['codigo']}")
print("\nPalavras-chave:", treino[0]["testes"])
