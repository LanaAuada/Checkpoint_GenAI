"""
Funções compartilhadas por todos os experimentos.

Objetivo pedagógico: toda medição do laboratório passa por aqui, então
'antes' e 'depois' são sempre medidos exatamente do mesmo jeito.
"""
from __future__ import annotations

import json
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ATTN, DEVICE, DTYPE, RUNS, SEED


# --------------------------------------------------------------- semente
def fixar_seed(seed: int = SEED) -> None:
    """Sem isto, dois treinos iguais dão resultados diferentes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------- carga
def carregar_tokenizer(nome: str):
    tok = AutoTokenizer.from_pretrained(nome)
    # SmolLM2 herda o tokenizer estilo GPT-2: não tem pad_token.
    # Reaproveitar o eos é seguro em treino causal, porque o collator
    # marca as posições de padding com -100 e elas não entram na loss.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def carregar_modelo(nome: str, **kwargs):
    """Compatível com transformers v4 (torch_dtype) e v5 (dtype)."""
    comuns = dict(attn_implementation=ATTN, **kwargs)
    try:
        modelo = AutoModelForCausalLM.from_pretrained(nome, dtype=DTYPE, **comuns)
    except TypeError:
        modelo = AutoModelForCausalLM.from_pretrained(nome, torch_dtype=DTYPE, **comuns)
    return modelo.to(DEVICE)


def carregar(nome: str):
    """Atalho: devolve (modelo, tokenizer) prontos."""
    tok = carregar_tokenizer(nome)
    modelo = carregar_modelo(nome)
    modelo.config.pad_token_id = tok.pad_token_id
    return modelo, tok


# --------------------------------------------------------------- geração
@torch.no_grad()
def gerar(modelo, tok, prompt: str, max_new_tokens: int = 60,
          temperatura: float = 0.0) -> str:
    """Continuação de texto (modelo BASE). temperatura=0 => greedy, determinístico."""
    entradas = tok(prompt, return_tensors="pt").to(modelo.device)
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    if temperatura > 0:
        kw.update(do_sample=True, temperature=temperatura, top_p=0.9)
    else:
        kw.update(do_sample=False)
    saida = modelo.generate(**entradas, **kw)
    return tok.decode(saida[0], skip_special_tokens=True)


@torch.no_grad()
def chat(modelo, tok, pergunta: str, sistema: str | None = None,
         max_new_tokens: int = 256, temperatura: float = 0.0) -> str:
    """Pergunta/resposta (modelo INSTRUCT). Devolve só o turno do assistente."""
    msgs = []
    if sistema:
        msgs.append({"role": "system", "content": sistema})
    msgs.append({"role": "user", "content": pergunta})

    texto = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    entradas = tok(texto, return_tensors="pt").to(modelo.device)

    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    if temperatura > 0:
        kw.update(do_sample=True, temperature=temperatura, top_p=0.9)
    else:
        kw.update(do_sample=False)

    saida = modelo.generate(**entradas, **kw)
    novos = saida[0][entradas["input_ids"].shape[1]:]
    return tok.decode(novos, skip_special_tokens=True).strip()


# --------------------------------------------------------------- métricas
@torch.no_grad()
def perplexidade(modelo, tok, textos: list[str]) -> float:
    """
    Perplexidade média PONDERADA POR TOKEN sobre um conjunto de textos.

    Diferente de exp(loss) de uma frase só: frases curtas pesariam igual às
    longas e o número viraria ruído. Use SEMPRE em conjunto held-out —
    medir no próprio dado de treino mede memorização, não generalização.
    """
    modelo.eval()
    nll_total, tokens_total = 0.0, 0
    for texto in textos:
        enc = tok(texto, return_tensors="pt", truncation=True, max_length=1024)
        enc = {k: v.to(modelo.device) for k, v in enc.items()}
        n = enc["input_ids"].numel() - 1
        if n <= 0:
            continue
        saida = modelo(**enc, labels=enc["input_ids"])
        nll_total += saida.loss.item() * n
        tokens_total += n
    return math.exp(nll_total / tokens_total)


# --------------------------------------------------------------- VRAM/tempo
def reset_vram() -> None:
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def warmup(proporcao: float) -> dict:
    """
    Devolve o argumento de warmup correto para a versão instalada.

    transformers v4 tinha DOIS argumentos: warmup_steps (int) e warmup_ratio (float).
    A v5 removeu warmup_ratio e fez warmup_steps aceitar float: valor < 1 é
    interpretado como proporção do total de passos.

    Uso:  TrainingArguments(..., **warmup(0.05), ...)
    """
    import inspect

    from transformers import TrainingArguments

    campos = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_ratio" in campos:
        return {"warmup_ratio": proporcao}
    return {"warmup_steps": proporcao}


def liberar(*objetos) -> None:
    """Descarrega modelos da VRAM. `del` sozinho não devolve a memória."""
    import gc

    for o in objetos:
        del o
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def vram_pico_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**3 if DEVICE == "cuda" else 0.0


@contextmanager
def medir(rotulo: str):
    """with medir('treino'): ...  -> imprime tempo e VRAM de pico."""
    reset_vram()
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"[{rotulo}] tempo={dt:.1f}s  VRAM_pico={vram_pico_gb():.2f} GB")


# --------------------------------------------------------------- resultados
def salvar_resultado(nome: str, dados: dict) -> Path:
    """Guarda toda medição em runs/resultados/. É o que alimenta o bake-off."""
    destino = RUNS / "resultados"
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / f"{nome}.json"
    caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> salvo em {caminho.relative_to(RUNS.parent)}")
    return caminho


def carregar_resultado(nome: str) -> dict | None:
    caminho = RUNS / "resultados" / f"{nome}.json"
    if not caminho.exists():
        return None
    return json.loads(caminho.read_text(encoding="utf-8"))


def contar_parametros(modelo) -> tuple[int, int]:
    total = sum(p.numel() for p in modelo.parameters())
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return total, treinaveis
