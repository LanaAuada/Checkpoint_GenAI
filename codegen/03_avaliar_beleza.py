"""
03 - Avaliando o modelo: a resposta contém as palavras-chave esperadas?
=======================================================================
Versão do 03_avaliar.py para o tema de beleza. Para cada pedido do teste:
  1. o modelo gera a resposta;
  2. confere se TODAS as palavras-chave esperadas aparecem nela;
  3. conta os acertos (variação e inédita).

Rodar:
  python 03_avaliar_beleza.py                  -> modelo base (ANTES do treino)
  python 03_avaliar_beleza.py modelo-treinado  -> modelo treinado (DEPOIS)
Salva em resultados/base.json ou resultados/modelo-treinado.json
"""
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "HuggingFaceTB/SmolLM2-135M"
MODELO = BASE
argumentos = sys.argv[1:]
if argumentos and not any(a.startswith("-") for a in argumentos):
    MODELO = argumentos[0]
MAX_TOKENS = 150

try:
    os.chdir(Path(__file__).resolve().parent)
except NameError:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"

# %% Carregar modelo e dados de teste
tokenizer = AutoTokenizer.from_pretrained(MODELO)
model = AutoModelForCausalLM.from_pretrained(MODELO).to(device)
model.eval()
teste = [json.loads(l) for l in open("dados/teste.jsonl", encoding="utf-8")]
gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
print(f"Modelo: {MODELO}  |  hardware: {device} ({gpu})  |  {len(teste)} pedidos\n")


def normalizar(texto):
    """minúsculas e sem acento, para comparar palavras-chave."""
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def gerar_resposta(pedido):
    prompt = f"### Pedido:\n{pedido}\n### Resposta:\n"      # mesmo formato do treino
    entrada = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        saida = model.generate(**entrada, max_new_tokens=MAX_TOKENS, do_sample=False,
                               pad_token_id=tokenizer.eos_token_id)
    novos = saida[0][entrada["input_ids"].shape[1]:]
    texto = tokenizer.decode(novos, skip_special_tokens=True)
    return texto.split("###")[0].strip()          # corta se começar outro pedido


def confere(resposta, chaves):
    r = normalizar(resposta)
    faltando = [k for k in chaves if normalizar(k) not in r]
    return (not faltando), ("faltou: " + ", ".join(faltando) if faltando else "")


# %% Avaliar todos os pedidos
resultados = []
inicio = time.time()
for i, ex in enumerate(teste, 1):
    resposta = gerar_resposta(ex["tarefa"])
    ok, erro = confere(resposta, ex["testes"])
    resultados.append({**ex, "gerado": resposta, "passou": ok, "erro": erro})
    curta = resposta.replace("\n", " ")[:50]
    print(f"[{i:2d}/{len(teste)}] {'PASSOU' if ok else 'falhou'}  {ex['familia']:<24} "
          f"{curta!r}  {erro}")

# %% Resumo
print(f"\nTempo: {time.time() - inicio:.0f} s")
if device == "cuda":
    print(f"VRAM de pico: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print("=" * 50)
for tipo in ["variacao", "inedita"]:
    grupo = [r for r in resultados if r["tipo"] == tipo]
    acertos = sum(r["passou"] for r in grupo)
    print(f"{tipo:<10}: {acertos:2d}/{len(grupo)}  ({acertos / len(grupo):.0%})")
total = sum(r["passou"] for r in resultados)
print(f"{'TOTAL':<10}: {total:2d}/{len(resultados)}  ({total / len(resultados):.0%})")
print("=" * 50)

for r in resultados[:3]:
    print(f"\nPEDIDO: {r['tarefa']}")
    print(f"--- esperado ---\n{r['codigo']}")
    print(f"--- gerado ({'passou' if r['passou'] else 'falhou'}) ---\n{r['gerado']}")

# %% Salvar para comparar depois
Path("resultados").mkdir(exist_ok=True)
nome = "base" if MODELO == BASE else Path(MODELO).name
with open(f"resultados/{nome}.json", "w", encoding="utf-8") as arq:
    json.dump(resultados, arq, ensure_ascii=False, indent=2)
print(f"\nResultados salvos em resultados/{nome}.json")
