"""
AULA 8 — Rota 5: o modelo não responde, ele consulta

Inverte o problema: em vez de o modelo SABER, o modelo CONSULTA.
O LLM vira a interface em linguagem natural; a fonte da verdade continua
sendo o arquivo, o banco ou a API.

Duas demonstrações:
  A) extração estruturada — 200 PDFs viram uma tabela
  B) function calling — o modelo decide chamar a busca no corpus
"""
# %%
import json
import re

from config import MODELO_CHAT
from rag import buscar, carregar_encoder, carregar_indice
from utils import carregar, chat, medir, salvar_resultado

modelo, tok = carregar(MODELO_CHAT)

# %% A) Extração estruturada: PDF -> JSON
ESQUEMA = {
    "titulo": "string",
    "tema_principal": "string",
    "conceitos": ["string"],
    "tem_formula_matematica": "boolean",
}

def extrair_json(texto: str) -> dict | None:
    """Reparo tolerante: pega o primeiro objeto JSON balanceado do texto."""
    inicio = texto.find("{")
    if inicio < 0:
        return None
    profundidade = 0
    for i, c in enumerate(texto[inicio:], inicio):
        profundidade += (c == "{") - (c == "}")
        if profundidade == 0:
            try:
                return json.loads(texto[inicio:i + 1])
            except json.JSONDecodeError:
                return None
    return None


PROMPT_EXTRACAO = (
    "Extraia as informações do trecho e responda APENAS com um objeto JSON "
    f"neste esquema, sem texto antes ou depois:\n{json.dumps(ESQUEMA, ensure_ascii=False)}\n\n"
    "Trecho:\n"
)

indice, chunks = carregar_indice()
amostra = chunks[:12]

linhas, falhas = [], 0
with medir("extração estruturada"):
    for c in amostra:
        bruto = chat(modelo, tok, PROMPT_EXTRACAO + c["texto"], max_new_tokens=200)
        obj = extrair_json(bruto)
        if obj is None:
            falhas += 1
            continue
        obj["_fonte"] = f"{c['arquivo']} p.{c['pagina']}"
        linhas.append(obj)

print(f"\n{len(linhas)} extrações válidas, {falhas} falhas de formato "
      f"({100*falhas/len(amostra):.0f}%)")
for l in linhas[:5]:
    print(" ", json.dumps(l, ensure_ascii=False)[:140])

print("""
AS FALHAS SÃO O PONTO. Um modelo de 1,7B em texto livre erra o formato uma
fração das vezes — e em um pipeline de 10.000 documentos isso é inaceitável.

A solução NÃO é um prompt melhor: é DECODIFICAÇÃO RESTRITA. Bibliotecas
como `outlines` ou `xgrammar` restringem, a cada passo, os tokens possíveis
aos que mantêm o JSON válido segundo uma gramática. A taxa de formato
inválido vai a ZERO por construção — não por sorte.

    pip install outlines
    generator = outlines.generate.json(modelo, EsquemaPydantic)

Esta é a diferença entre um demo e um sistema.
""")

# %% B) Function calling: o modelo decide quando buscar
enc = carregar_encoder()

FERRAMENTAS = [{
    "type": "function",
    "function": {
        "name": "buscar_no_material",
        "description": "Busca trechos no material da disciplina de IA Generativa.",
        "parameters": {
            "type": "object",
            "properties": {"consulta": {"type": "string",
                                        "description": "termos de busca"}},
            "required": ["consulta"],
        },
    },
}]


def turno_com_ferramentas(pergunta: str, max_passos: int = 3) -> str:
    msgs = [{"role": "user", "content": pergunta}]
    for _ in range(max_passos):
        texto = tok.apply_chat_template(msgs, tools=FERRAMENTAS, tokenize=False,
                                        add_generation_prompt=True)
        ent = tok(texto, return_tensors="pt").to(modelo.device)
        saida = modelo.generate(**ent, max_new_tokens=256, do_sample=False,
                                pad_token_id=tok.pad_token_id)
        resposta = tok.decode(saida[0][ent["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()

        chamada = extrair_json(resposta)
        if chamada and ("name" in chamada or "function" in chamada):
            nome = chamada.get("name") or chamada.get("function")
            args = chamada.get("arguments", chamada.get("parameters", {}))
            consulta = args.get("consulta", pergunta) if isinstance(args, dict) else pergunta
            if nome == "buscar_no_material":
                trechos = buscar(consulta, indice, chunks, enc, k=3)
                conteudo = "\n\n".join(
                    f"[{t['arquivo']}, p.{t['pagina']}] {t['texto'][:400]}" for t in trechos)
                msgs.append({"role": "assistant", "content": resposta})
                msgs.append({"role": "tool", "name": nome, "content": conteudo})
                continue
        return resposta
    return resposta


for p in ["O que é esquecimento catastrófico segundo o material?",
          "Quantos parâmetros tem o SmolLM2-135M?"]:
    print(f"\nP: {p}")
    print(f"R: {turno_com_ferramentas(p)}")

print("""
OBSERVE: às vezes o SmolLM2-1.7B emite a chamada corretamente, às vezes
responde direto ignorando a ferramenta. Function calling em modelos
pequenos é frágil — e essa fragilidade é a razão pela qual, em produção,
se usa RAG DETERMINÍSTICO (sempre buscar) em vez de deixar o modelo
decidir. Deixe o modelo decidir só quando ele for grande o bastante.
""")

salvar_resultado("rota5_tools", {
    "extracoes_validas": len(linhas), "falhas_formato": falhas,
    "amostra": len(amostra), "exemplos": linhas[:5],
})
