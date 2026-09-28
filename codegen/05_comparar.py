"""
05 - Antes x depois: modelo base contra modelo treinado
=======================================================
1. Mostra o placar salvo pelo 03_avaliar.py (base e treinado).
2. Faz a MESMA pergunta aos dois modelos, lado a lado.
   Troque o texto de PERGUNTAS para testar ao vivo com a turma.

Rodar:  python 05_comparar.py
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Garante que dados/, resultados/ e modelo-treinado/ fiquem ao lado deste script
import os
try:
    os.chdir(Path(__file__).resolve().parent)
except NameError:                     # janela interativa sem __file__
    pass

BASE = "HuggingFaceTB/SmolLM2-135M"
TREINADO = "modelo-treinado"
PERGUNTAS = [
    "Escreva uma função em Python chamada `somar_pares(valores)` que retorna a soma dos números pares da lista.",
    "Crie a função `contar_letras_a(texto)`, que retorna quantas letras 'a' existem no texto.",
]

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# %% 1) Placar
print("PLACAR (taxa de código que passa nos testes)")
print(f"{'modelo':<18}{'variação':>10}{'inédita':>10}{'total':>10}")
for nome in ["base", TREINADO]:
    arq = Path(f"resultados/{nome}.json")
    if not arq.exists():
        print(f"{nome:<18}  (rode antes: python 03_avaliar.py"
              f"{'' if nome == 'base' else ' ' + TREINADO})")
        continue
    res = json.loads(arq.read_text(encoding="utf-8"))
    taxa = lambda g: f"{sum(r['passou'] for r in g) / len(g):.0%}"
    print(f"{nome:<18}{taxa([r for r in res if r['tipo'] == 'variacao']):>10}"
          f"{taxa([r for r in res if r['tipo'] == 'inedita']):>10}{taxa(res):>10}")


# %% 2) Mesma pergunta para os dois modelos
def responder(model, tokenizer, tarefa):
    prompt = f"### Tarefa:\n{tarefa}\n### Código:\n"
    entrada = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        saida = model.generate(**entrada, max_new_tokens=80, do_sample=False,
                               pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(saida[0][entrada["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


modelos = {}
for nome, caminho in [("BASE", BASE), ("TREINADO", TREINADO)]:
    tok = AutoTokenizer.from_pretrained(caminho)
    mod = AutoModelForCausalLM.from_pretrained(caminho).to(device).eval()
    modelos[nome] = (mod, tok)

for tarefa in PERGUNTAS:
    print("\n" + "=" * 70)
    print(f"TAREFA: {tarefa}")
    for nome, (mod, tok) in modelos.items():
        print(f"\n--- {nome} ---")
        print(responder(mod, tok, tarefa))
