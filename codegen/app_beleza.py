"""
Frontend (Gradio) do assistente de skincare.
Mostra a resposta do modelo BASE e do modelo TREINADO lado a lado e o status da GPU.

Coloque na mesma pasta do 04_treinar.py (onde fica modelo-treinado/).
Rodar:  python app_beleza.py   ->  http://127.0.0.1:7860
"""
import os
import time
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    os.chdir(Path(__file__).resolve().parent)
except NameError:
    pass

BASE = "HuggingFaceTB/SmolLM2-135M"
TREINADO = "modelo-treinado"
device = "cuda" if torch.cuda.is_available() else "cpu"


def carregar(caminho):
    tok = AutoTokenizer.from_pretrained(caminho)
    mod = AutoModelForCausalLM.from_pretrained(caminho).to(device).eval()
    return mod, tok


modelos = {"BASE": carregar(BASE)}
if Path(TREINADO).exists():
    modelos["TREINADO"] = carregar(TREINADO)


def gerar(nome, pedido, temperatura, max_tokens):
    if nome not in modelos:
        return "(pasta modelo-treinado/ nao encontrada: rode 04_treinar.py)"
    mod, tok = modelos[nome]
    prompt = f"### Pedido:\n{pedido}\n### Resposta:\n"      # mesmo formato do treino
    entrada = tok(prompt, return_tensors="pt").to(device)
    opcoes = (dict(do_sample=True, temperature=float(temperatura))
              if temperatura > 0 else dict(do_sample=False))
    with torch.no_grad():
        saida = mod.generate(**entrada, max_new_tokens=int(max_tokens),
                             pad_token_id=tok.eos_token_id, **opcoes)
    novos = saida[0][entrada["input_ids"].shape[1]:]
    return tok.decode(novos, skip_special_tokens=True).split("###")[0].strip()


def status_gpu(segundos):
    if device != "cuda":
        return f"Rodando em CPU | {segundos:.1f}s"
    p = torch.cuda.get_device_properties(0)
    usada = torch.cuda.memory_allocated() / 1024**3
    return (f"GPU: {p.name} | VRAM alocada: {usada:.2f} de "
            f"{p.total_memory / 1024**3:.2f} GB | tempo: {segundos:.1f}s")


def responder(pedido, temperatura, max_tokens):
    if not pedido.strip():
        return "", "", "Digite um pedido."
    t0 = time.perf_counter()
    r_base = gerar("BASE", pedido, temperatura, max_tokens)
    r_tr = gerar("TREINADO", pedido, temperatura, max_tokens)
    return r_base, r_tr, status_gpu(time.perf_counter() - t0)


demo = gr.Interface(
    fn=responder,
    inputs=[
        gr.Textbox(label="Pedido", lines=3,
                   placeholder="Qual hidratante você recomenda para pele oleosa?"),
        gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperatura (0 = determinístico)"),
        gr.Slider(32, 300, value=160, step=8, label="Max tokens"),
    ],
    outputs=[
        gr.Textbox(label="Modelo BASE (antes do treino)", lines=6),
        gr.Textbox(label="Modelo TREINADO (depois do treino)", lines=6),
        gr.Textbox(label="Status"),
    ],
    examples=[
        ["Qual hidratante você recomenda para pele oleosa?", 0.0, 160],
        ["Monte uma rotina de skincare para a manhã para pele sensível.", 0.0, 160],
        ["Qual protetor solar você recomenda para pele seca?", 0.0, 160],
        ["Qual rotina você indica para acne?", 0.0, 160],
    ],
    title="Assistente de skincare (SmolLM2-135M)",
    description="Projeto acadêmico: sugere categorias de produto e rotinas por tipo de pele. "
                "Não substitui a orientação de um dermatologista.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
