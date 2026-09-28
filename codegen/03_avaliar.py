"""
03 - Avaliando o modelo: o código gerado PASSA nos testes?
==========================================================
Para cada tarefa do conjunto de teste:
  1. o modelo gera o código a partir do enunciado;
  2. o código é executado junto com os asserts (em um processo separado);
  3. conta-se quantas tarefas passaram  ->  taxa de acerto (pass@1).

Qual modelo avaliar? Duas formas:
  a) No VSCode (células # %%): mude USAR_MODELO_TREINADO abaixo.
     False = modelo base do Hugging Face (ANTES do treino)
     True  = pasta modelo-treinado/     (DEPOIS do treino)
  b) No terminal:
     python 03_avaliar.py                    -> modelo base
     python 03_avaliar.py modelo-treinado    -> modelo treinado
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

USAR_MODELO_TREINADO = False          # <-- troque para True depois do 04_treinar.py

BASE = "HuggingFaceTB/SmolLM2-135M"   # SmolLM2-135M original, baixado do Hugging Face
MODELO = "modelo-treinado" if USAR_MODELO_TREINADO else BASE
# Argumento de terminal (ignorado no VSCode/Jupyter, que passa "-f ...")
argumentos = sys.argv[1:]
if argumentos and not any(a.startswith("-") for a in argumentos):
    MODELO = argumentos[0]
MAX_TOKENS = 100

# Garante que as pastas dados/ e resultados/ sejam as ao lado deste script
try:
    os.chdir(Path(__file__).resolve().parent)
except NameError:                     # janela interativa sem __file__
    pass

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# %% Carregar modelo e dados de teste
tokenizer = AutoTokenizer.from_pretrained(MODELO)
model = AutoModelForCausalLM.from_pretrained(MODELO).to(device)
model.eval()
teste = [json.loads(l) for l in open("dados/teste.jsonl", encoding="utf-8")]
print(f"Modelo: {MODELO}  |  hardware: {device}  |  {len(teste)} tarefas\n")


def gerar_codigo(tarefa):
    """Usa exatamente o mesmo formato de prompt do treino."""
    prompt = f"### Tarefa:\n{tarefa}\n### Código:\n"
    entrada = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        saida = model.generate(**entrada, max_new_tokens=MAX_TOKENS, do_sample=False,
                               pad_token_id=tokenizer.eos_token_id)
    novos = saida[0][entrada["input_ids"].shape[1]:]
    return tokenizer.decode(novos, skip_special_tokens=True).lstrip("\n")


def extrair_funcao(texto):
    """Procura o primeiro 'def' (ignorando texto e ``` antes dele) e fica só
    com a função: para na 1ª linha não indentada depois do 'def'."""
    todas = [l for l in texto.split("\n") if not l.strip().startswith("```")]
    inicio = next((i for i, l in enumerate(todas) if l.startswith("def ")), None)
    if inicio is None:
        return texto.strip()          # não escreveu função nenhuma
    linhas = [todas[inicio]]
    for linha in todas[inicio + 1:]:
        if linha.strip() and not linha.startswith((" ", "\t")):
            break
        linhas.append(linha)
    return "\n".join(linhas).rstrip()


def resumo(codigo, n=45):
    """Primeira linha do que o modelo escreveu, para mostrar na tela."""
    primeira = codigo.strip().split("\n")[0] if codigo.strip() else "(vazio)"
    return primeira[:n] + ("..." if len(primeira) > n else "")


def passa_nos_testes(codigo, testes):
    """Executa código + asserts num processo separado (com tempo limite)."""
    programa = codigo + "\n\n" + "\n".join(f"assert {t}" for t in testes) + "\n"
    programa = programa.replace("\x00", "")          # lixo que o modelo às vezes gera
    try:
        r = subprocess.run([sys.executable, "-c", programa],
                           capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        return False, "tempo esgotado (loop infinito?)"
    except Exception as e:
        return False, f"não executou: {e}"
    if r.returncode == 0:
        return True, ""
    erro = r.stderr.strip().splitlines()
    return False, erro[-1] if erro else "erro"


# %% Avaliar todas as tarefas
resultados = []
inicio = time.time()
for i, ex in enumerate(teste, 1):
    codigo = extrair_funcao(gerar_codigo(ex["tarefa"]))
    ok, erro = passa_nos_testes(codigo, ex["testes"])
    resultados.append({**ex, "gerado": codigo, "passou": ok, "erro": erro})
    print(f"[{i:2d}/{len(teste)}] {'PASSOU' if ok else 'falhou'}  {ex['familia']:<20} "
          f"gerou: {resumo(codigo)!r}  {erro}")

# %% Resumo
print(f"\nTempo: {time.time() - inicio:.0f} s")
print("=" * 50)
for tipo in ["variacao", "inedita"]:
    grupo = [r for r in resultados if r["tipo"] == tipo]
    acertos = sum(r["passou"] for r in grupo)
    print(f"{tipo:<10}: {acertos:2d}/{len(grupo)}  ({acertos / len(grupo):.0%})")
total = sum(r["passou"] for r in resultados)
print(f"{'TOTAL':<10}: {total:2d}/{len(resultados)}  ({total / len(resultados):.0%})")
print("=" * 50)

# Mostra 3 exemplos completos lado a lado com a resposta esperada
for r in resultados[:3]:
    print(f"\nTAREFA: {r['tarefa']}")
    print(f"--- esperado ---\n{r['codigo']}")
    print(f"--- gerado ({'passou' if r['passou'] else 'falhou'}) ---\n{r['gerado']}")

# %% Salvar para comparar depois
Path("resultados").mkdir(exist_ok=True)
nome = "base" if MODELO == BASE else Path(MODELO).name
with open(f"resultados/{nome}.json", "w", encoding="utf-8") as arq:
    json.dump(resultados, arq, ensure_ascii=False, indent=2)
print(f"\nResultados salvos em resultados/{nome}.json")
