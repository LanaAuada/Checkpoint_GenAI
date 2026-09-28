"""
AULA 7 — Passo 0: o ambiente está de pé?

Rode ISTO antes de qualquer outra coisa. Se algum item falhar, não adianta
seguir: 90% dos problemas de "fine-tuning não funciona" são ambiente.

Como rodar:
    - No terminal:  python a7_00_smoke_test.py
    - No VSCode:    clique em "Run Cell" acima de cada  # %%
"""
# %% 1) PyTorch enxerga a GPU?
import torch

print("PyTorch      :", torch.__version__)
print("compilado c/ CUDA:", torch.version.cuda)
print("CUDA disponível  :", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("""
    >>> FALHOU. Causa nº 1: você instalou o wheel CPU do PyTorch.
        Confira com:  pip show torch     (se a versão termina em '+cpu', é isso)
        Corrija com:  pip uninstall -y torch
                      pip install torch --index-url https://download.pytorch.org/whl/cu128
        (confira o índice correto em https://pytorch.org/get-started/locally/)
    """)
else:
    p = torch.cuda.get_device_properties(0)
    livre, total = torch.cuda.mem_get_info()
    print("GPU          :", p.name)
    print("Compute cap. :", f"sm_{p.major}{p.minor}")
    print("VRAM total   :", f"{total / 1024**3:.2f} GB")
    print("VRAM livre   :", f"{livre / 1024**3:.2f} GB   <- este é o seu orçamento real")
    print("bf16         :", torch.cuda.is_bf16_supported())

# %% 2) A GPU realmente calcula? (não basta ser detectada)
if torch.cuda.is_available():
    import time

    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        c = a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tflops = 50 * 2 * 4096**3 / dt / 1e12
    print(f"matmul bf16 4096³ x50 : {dt:.2f}s  ~{tflops:.1f} TFLOP/s")
    print("(uma 3060 saudável fica na casa de dezenas de TFLOP/s em bf16)")

# %% 3) As bibliotecas do ecossistema estão instaladas?
import importlib

esperadas = [
    "transformers", "datasets", "accelerate", "peft", "trl",
    "sentence_transformers", "faiss", "fitz", "numpy",
]
for nome in esperadas:
    try:
        mod = importlib.import_module(nome)
        versao = getattr(mod, "__version__", "ok")
        print(f"{nome:24s} {versao}")
    except Exception as e:  # noqa: BLE001
        print(f"{nome:24s} FALTANDO  ({type(e).__name__})")

# bitsandbytes é opcional (só para QLoRA 4-bit) e é o que mais dá problema
# no Windows nativo. Se falhar aqui, use LoRA em bf16 ou rode via WSL2.
try:
    import bitsandbytes as bnb

    print(f"{'bitsandbytes':24s} {bnb.__version__} (QLoRA disponível)")
except Exception:  # noqa: BLE001
    print(f"{'bitsandbytes':24s} indisponível — QLoRA fica de fora, LoRA bf16 funciona")

# %% 4) Consigo baixar e rodar o SmolLM2?
from config import MODELO_BASE, resumo_ambiente
from utils import carregar, gerar

print(resumo_ambiente())
print("\nBaixando", MODELO_BASE, "(~270 MB na primeira vez)...")
modelo, tok = carregar(MODELO_BASE)
print("Geração de teste:")
print(" ", gerar(modelo, tok, "The capital of France is", max_new_tokens=12))

print("\n=== AMBIENTE OK — pode seguir para a7_01_anatomia.py ===")
