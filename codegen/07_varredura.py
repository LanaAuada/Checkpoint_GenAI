"""
07 - Varredura de parâmetros
=============================
Roda várias combinações de learning rate / épocas / lote, treinando do zero
a cada rodada, e mede pass rate (variação e inédita), loss final e tempo.

Gera:
  resultados/varredura.json   -> todos os resultados, um por rodada
  varredura.png               -> gráfico comparando as rodadas

Rodar:  python 07_varredura.py
Tempo estimado: poucos minutos por rodada (modelo pequeno).
"""
import json
import os
import random
import time
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    os.chdir(Path(__file__).resolve().parent)
except NameError:
    pass

MODELO = "HuggingFaceTB/SmolLM2-135M"
MAX_LEN = 256
MAX_TOKENS_GERACAO = 150

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Hardware: {device}")

treino_txt = [json.loads(l) for l in open("dados/treino.jsonl", encoding="utf-8")]
teste_txt = [json.loads(l) for l in open("dados/teste.jsonl", encoding="utf-8")]

# ------------------------------------------------------- combinações a testar
# Ponto de partida (baseline já medida: LR=5e-5, epocas=3, lote=8 -> 74%)
CONFIGS = [
    {"nome": "baseline",        "lr": 5e-5, "epocas": 3, "lote": 8},
    {"nome": "lr_menor",        "lr": 2e-5, "epocas": 3, "lote": 8},
    {"nome": "lr_maior",        "lr": 1e-4, "epocas": 3, "lote": 8},
    {"nome": "mais_epocas",     "lr": 5e-5, "epocas": 6, "lote": 8},
    {"nome": "menos_epocas",    "lr": 5e-5, "epocas": 1, "lote": 8},
    {"nome": "lote_menor",      "lr": 5e-5, "epocas": 3, "lote": 4},
]


def normalizar(texto):
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def preparar_lote(exemplos, tokenizer):
    ids_lote, labels_lote = [], []
    for ex in exemplos:
        prompt = f"### Pedido:\n{ex['tarefa']}\n### Resposta:\n"
        resposta = ex["codigo"] + tokenizer.eos_token
        ids_p = tokenizer(prompt)["input_ids"]
        ids_r = tokenizer(resposta, add_special_tokens=False)["input_ids"]
        ids = (ids_p + ids_r)[:MAX_LEN]
        labels = ([-100] * len(ids_p) + ids_r)[:MAX_LEN]
        ids_lote.append(ids)
        labels_lote.append(labels)
    tam = max(len(x) for x in ids_lote)
    input_ids = [x + [tokenizer.pad_token_id] * (tam - len(x)) for x in ids_lote]
    mascara = [[1] * len(x) + [0] * (tam - len(x)) for x in ids_lote]
    labels = [x + [-100] * (tam - len(x)) for x in labels_lote]
    return {
        "input_ids": torch.tensor(input_ids, device=device),
        "attention_mask": torch.tensor(mascara, device=device),
        "labels": torch.tensor(labels, device=device),
    }


def treinar_uma_rodada(cfg):
    random.seed(0)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODELO)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODELO).to(device).float()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    otimizador = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    dados = treino_txt.copy()
    t0 = time.perf_counter()
    for _ in range(cfg["epocas"]):
        random.shuffle(dados)
        for i in range(0, len(dados), cfg["lote"]):
            lote = preparar_lote(dados[i:i + cfg["lote"]], tokenizer)
            loss = model(**lote).loss
            loss.backward()
            otimizador.step()
            otimizador.zero_grad()
    tempo = time.perf_counter() - t0

    # loss final no conjunto de teste
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(teste_txt), cfg["lote"]):
            total += model(**preparar_lote(teste_txt[i:i + cfg["lote"]], tokenizer)).loss.item()
    loss_teste = total / ((len(teste_txt) + cfg["lote"] - 1) // cfg["lote"])

    vram = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
    return model, tokenizer, tempo, loss_teste, vram


def avaliar(model, tokenizer):
    model.eval()
    acertos = {"variacao": 0, "inedita": 0}
    totais = {"variacao": 0, "inedita": 0}
    for ex in teste_txt:
        prompt = f"### Pedido:\n{ex['tarefa']}\n### Resposta:\n"
        entrada = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            saida = model.generate(**entrada, max_new_tokens=MAX_TOKENS_GERACAO,
                                   do_sample=False, pad_token_id=tokenizer.eos_token_id)
        novos = saida[0][entrada["input_ids"].shape[1]:]
        resposta = tokenizer.decode(novos, skip_special_tokens=True).split("###")[0]
        r = normalizar(resposta)
        ok = all(normalizar(k) in r for k in ex["testes"])
        totais[ex["tipo"]] += 1
        acertos[ex["tipo"]] += ok
    return acertos, totais


# ------------------------------------------------------------------- rodar
resultados = []
for cfg in CONFIGS:
    print(f"\n=== {cfg['nome']}: lr={cfg['lr']}, epocas={cfg['epocas']}, lote={cfg['lote']} ===")
    model, tokenizer, tempo, loss_teste, vram = treinar_uma_rodada(cfg)
    acertos, totais = avaliar(model, tokenizer)

    total_acertos = sum(acertos.values())
    total_geral = sum(totais.values())
    linha = {
        **cfg,
        "tempo_s": round(tempo, 1),
        "loss_teste": round(loss_teste, 3),
        "vram_gb": round(vram, 2),
        "variacao_pct": round(100 * acertos["variacao"] / totais["variacao"], 1),
        "inedita_pct": round(100 * acertos["inedita"] / totais["inedita"], 1),
        "total_pct": round(100 * total_acertos / total_geral, 1),
    }
    resultados.append(linha)
    print(f"  tempo={linha['tempo_s']}s  loss={linha['loss_teste']}  "
          f"vram={linha['vram_gb']}GB  variacao={linha['variacao_pct']}%  "
          f"inedita={linha['inedita_pct']}%  total={linha['total_pct']}%")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

# ------------------------------------------------------------------- salvar
Path("resultados").mkdir(exist_ok=True)
with open("resultados/varredura.json", "w", encoding="utf-8") as arq:
    json.dump(resultados, arq, ensure_ascii=False, indent=2)

print("\n" + "=" * 90)
print(f"{'config':<16}{'lr':>8}{'épocas':>8}{'lote':>6}{'tempo(s)':>10}"
      f"{'loss':>8}{'VRAM':>7}{'variação':>10}{'inédita':>9}{'total':>8}")
for r in resultados:
    print(f"{r['nome']:<16}{r['lr']:>8.0e}{r['epocas']:>8}{r['lote']:>6}"
          f"{r['tempo_s']:>10}{r['loss_teste']:>8}{r['vram_gb']:>7}"
          f"{r['variacao_pct']:>9}%{r['inedita_pct']:>8}%{r['total_pct']:>7}%")

# ------------------------------------------------------------------- gráfico
fig, ax = plt.subplots(figsize=(9, 4.5))
nomes = [r["nome"] for r in resultados]
x = range(len(nomes))
largura = 0.25
for i, (chave, rotulo, cor) in enumerate([
    ("variacao_pct", "variação", "#E11D48"),
    ("inedita_pct", "inédita", "#A5B4FC"),
    ("total_pct", "total", "#2DD4BF"),
]):
    valores = [r[chave] for r in resultados]
    ax.bar([p + (i - 1) * largura for p in x], valores, largura, label=rotulo, color=cor)
ax.set_xticks(list(x))
ax.set_xticklabels(nomes, rotation=15, fontsize=9)
ax.set_ylabel("% de acerto")
ax.set_ylim(0, 105)
ax.set_title("Varredura de hiperparâmetros — assistente de skincare")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("varredura.png", dpi=140)
print("\nGráfico salvo em varredura.png")