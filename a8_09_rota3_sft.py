"""
AULA 8 — Rota 3: SFT de instrução sobre os arquivos (LoRA)

Objetivo declarado: ensinar o modelo a RESPONDER COMO UM MONITOR da
disciplina — tom, formato, vocabulário.

Objetivo NÃO declarado (e que a turma vai descobrir sozinha): isto não
ensina fatos. Ao final o modelo escreve muito melhor e continua errando
números. Esse é o slide 33.

Etapas: chunks -> pares pergunta/resposta -> LoRA -> avaliação.
"""
# %%
import json
import random

from datasets import Dataset
from peft import LoraConfig, TaskType

from avaliacao import avaliar_rota, carregar_gabarito
from config import DADOS, MODELO_CHAT, RUNS, SEED, USA_BF16
from rag import carregar_chunks
from utils import (carregar, chat, fixar_seed, liberar, medir,
                   salvar_resultado, warmup)

fixar_seed()
ARQUIVO_DATASET = DADOS / "sft_qa.jsonl"

# %% 1) De PDF para pares pergunta/resposta
#
# Três caminhos possíveis:
#   (a) escrever à mão — 100 a 300 pares. Caro, e o melhor resultado.
#   (b) gerar sinteticamente com um modelo — o que fazemos aqui.
#   (c) reaproveitar perguntas reais de alunos — o melhor custo/benefício.
#
# ATENÇÃO METODOLÓGICA: nenhuma pergunta do gabarito pode entrar no treino.
# É o mesmo erro do notebook da Aula 7, em escala maior.

if ARQUIVO_DATASET.exists():
    pares = [json.loads(l) for l in ARQUIVO_DATASET.read_text(encoding="utf-8").splitlines()]
    print(f"Reaproveitando {len(pares)} pares de {ARQUIVO_DATASET.name}")
else:
    modelo_gerador, tok_gerador = carregar(MODELO_CHAT)
    chunks = carregar_chunks()
    random.shuffle(chunks)
    alvo = chunks[:200]

    INSTRUCAO = (
        "A partir do trecho abaixo, escreva UMA pergunta de prova e a resposta "
        "correspondente, ambas em português. Formato exato:\n"
        "PERGUNTA: <pergunta>\nRESPOSTA: <resposta em até 3 frases>\n\nTrecho:\n"
    )

    pares = []
    with medir("geração sintética de pares"):
        for i, c in enumerate(alvo):
            bruto = chat(modelo_gerador, tok_gerador, INSTRUCAO + c["texto"],
                         max_new_tokens=180, temperatura=0.7)
            if "PERGUNTA:" not in bruto or "RESPOSTA:" not in bruto:
                continue
            q = bruto.split("PERGUNTA:")[1].split("RESPOSTA:")[0].strip()
            a = bruto.split("RESPOSTA:")[1].strip()
            if len(q) < 15 or len(a) < 20:
                continue
            pares.append({"pergunta": q, "resposta": f"{a}\n\n[{c['arquivo']}, p.{c['pagina']}]"})
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(alvo)} trechos -> {len(pares)} pares")

    ARQUIVO_DATASET.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in pares), encoding="utf-8")
    print(f"{len(pares)} pares salvos em {ARQUIVO_DATASET}")
    liberar(modelo_gerador, tok_gerador)   # devolve os ~3,4 GB de VRAM

# Higiene: remover qualquer par parecido demais com o gabarito
gabarito = carregar_gabarito()
perguntas_gab = {g["pergunta"].lower()[:40] for g in gabarito}
antes = len(pares)
pares = [p for p in pares if p["pergunta"].lower()[:40] not in perguntas_gab]
print(f"Removidos {antes - len(pares)} pares que vazavam o gabarito.")

# %% 2) Formato prompt/completion — a máscara de loss sai de graça
#
# O DataCollatorForLanguageModeling(mlm=False) do notebook da Aula 7 treina
# em TODOS os tokens, inclusive na pergunta. Para SFT queremos loss só na
# resposta. Usando o formato conversacional prompt/completion, a TRL aplica
# completion_only_loss automaticamente.
registros = [{
    "prompt": [{"role": "user", "content": p["pergunta"]}],
    "completion": [{"role": "assistant", "content": p["resposta"]}],
} for p in pares]

ds = Dataset.from_list(registros).train_test_split(test_size=0.1, seed=SEED)
print(ds)

# %% 3) Medir ANTES (o mesmo gabarito, modelo intocado)
modelo, tok = carregar(MODELO_CHAT)
antes_sft = avaliar_rota("Antes do SFT (modelo Instruct puro)",
                         lambda p: chat(modelo, tok, p, max_new_tokens=200),
                         verboso=False)
liberar(modelo, tok)   # o SFTTrainer vai recarregar o modelo do zero

# %% 4) Treinar com LoRA
from trl import SFTConfig, SFTTrainer

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

args = SFTConfig(
    output_dir=str(RUNS / "a8_09_sft"),
    max_length=768,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,     # batch efetivo 16 sem estourar a VRAM
    learning_rate=1e-4,
    **warmup(0.05),                # v4: warmup_ratio | v5: warmup_steps
    lr_scheduler_type="cosine",
    bf16=USA_BF16,
    gradient_checkpointing=True,       # troca ~30% de tempo por VRAM
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="best",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_total_limit=1,
    seed=SEED,
    report_to="none",
    dataloader_num_workers=0,
)

trainer = SFTTrainer(
    model=MODELO_CHAT,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    peft_config=lora,
)

with medir("SFT LoRA 1.7B"):
    trainer.train()

destino = RUNS / "a8_09_sft" / "adaptador"
trainer.save_model(str(destino))
tamanho = sum(f.stat().st_size for f in destino.rglob("*") if f.is_file()) / 1024**2
print(f"Adaptador: {tamanho:.1f} MB (o modelo base tem ~3.400 MB)")

# %% 5) Medir DEPOIS
modelo_ft, tok_ft = trainer.model, trainer.processing_class
depois_sft = avaliar_rota("Rota 3 — SFT de instrução",
                          lambda p: chat(modelo_ft, tok_ft, p, max_new_tokens=200))

salvar_resultado("rota3_sft", {
    "n_pares": len(pares), "mb_adaptador": tamanho,
    "antes": antes_sft["resumo"], "depois": depois_sft["resumo"],
    "detalhes": depois_sft["detalhes"],
})

# %% 6) A lição
print(f"""
COMPARE:
  acerto factual  antes: {antes_sft['resumo']['acerto_factual_%']}%
                  depois: {depois_sft['resumo']['acerto_factual_%']}%
  citou fonte     antes: {antes_sft['resumo']['citou_fonte_%']}%
                  depois: {depois_sft['resumo']['citou_fonte_%']}%
  abstenção       antes: {antes_sft['resumo']['abstencao_correta_%']}%
                  depois: {depois_sft['resumo']['abstencao_correta_%']}%

O QUE ESPERAR:
  MELHORA muito  -> tom, formato, o hábito de citar fonte no padrão treinado
  NÃO melhora    -> acerto factual
  PIORA          -> abstenção. O modelo aprendeu a SEMPRE responder no
                    formato de monitor, inclusive quando não sabe. Ele agora
                    inventa a citação [arquivo, p.N] junto com o erro.

Este último ponto é o mais importante da aula: o fine-tuning tornou o erro
MAIS convincente. Fine-tuning ensina forma, não fato.
""")
