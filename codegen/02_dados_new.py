"""
02 - Criando o dataset de treino (tarefa -> código Python)
==========================================================
Gera exemplos no formato:

    ### Tarefa:
    Escreva uma função em Python chamada `somar_pares(numeros)` que ...
    ### Código:
    def somar_pares(numeros):
        return sum(x for x in numeros if x % 2 == 0)

Cada exemplo vem com TESTES (asserts). É isso que permite avaliar o modelo
de forma objetiva: o código gerado passa nos testes ou não passa.

Saída:
  dados/treino.jsonl  -> usado no treino
  dados/teste.jsonl   -> NUNCA visto no treino. Dois tipos:
      "variacao" = mesma família de tarefa, mas nome/enunciado diferentes
      "inedita"  = família de tarefa que o modelo nunca viu

Rodar:  python 02_dados.py
"""
import json
import random
from pathlib import Path

# Garante que dados/, resultados/ e modelo-treinado/ fiquem ao lado deste script
import os
try:
    os.chdir(Path(__file__).resolve().parent)
except NameError:                     # janela interativa sem __file__
    pass

random.seed(42)

# Cada família: nomes possíveis da função, nomes do argumento,
# frases que descrevem o que ela faz, corpo da função e testes.
# {a} = nome do argumento, {f} = nome da função
FAMILIAS = {
    "soma_pares": dict(
        nomes=["somar_pares", "soma_pares", "total_pares"],
        args=["numeros", "lista", "valores"],
        oque=["retorna a soma dos números pares da lista",
              "soma apenas os valores pares de uma lista"],
        corpo="    return sum(x for x in {a} if x % 2 == 0)",
        testes=["{f}([1, 2, 3, 4]) == 6", "{f}([]) == 0", "{f}([5, 7]) == 0"]),
    "soma_impares": dict(
        nomes=["somar_impares", "soma_impares", "total_impares"],
        args=["numeros", "lista", "valores"],
        oque=["retorna a soma dos números ímpares da lista",
              "soma apenas os valores ímpares de uma lista"],
        corpo="    return sum(x for x in {a} if x % 2 != 0)",
        testes=["{f}([1, 2, 3, 4]) == 4", "{f}([]) == 0", "{f}([2, 8]) == 0"]),
    "maior": dict(
        nomes=["maior", "maior_valor", "maximo"],
        args=["numeros", "lista", "valores"],
        oque=["retorna o maior número da lista",
              "devolve o maior valor de uma lista"],
        corpo="    return max({a})",
        testes=["{f}([3, 9, 2]) == 9", "{f}([-5, -1]) == -1"]),
    "menor": dict(
        nomes=["menor", "menor_valor", "minimo"],
        args=["numeros", "lista", "valores"],
        oque=["retorna o menor número da lista",
              "devolve o menor valor de uma lista"],
        corpo="    return min({a})",
        testes=["{f}([3, 9, 2]) == 2", "{f}([-5, -1]) == -5"]),
    "media": dict(
        nomes=["media", "calcular_media", "valor_medio"],
        args=["numeros", "lista", "notas"],
        oque=["retorna a média aritmética dos números da lista",
              "calcula a média dos valores de uma lista"],
        corpo="    return sum({a}) / len({a})",
        testes=["{f}([2, 4]) == 3", "{f}([5]) == 5"]),
    "soma_quadrados": dict(
        nomes=["soma_quadrados", "somar_quadrados", "total_quadrados"],
        args=["numeros", "lista", "valores"],
        oque=["retorna a soma dos quadrados dos números da lista",
              "soma o quadrado de cada valor de uma lista"],
        corpo="    return sum(x * x for x in {a})",
        testes=["{f}([1, 2, 3]) == 14", "{f}([]) == 0"]),
    "contar_positivos": dict(
        nomes=["contar_positivos", "qtd_positivos", "conta_positivos"],
        args=["numeros", "lista", "valores"],
        oque=["retorna quantos números positivos existem na lista",
              "conta os valores maiores que zero de uma lista"],
        corpo="    return sum(1 for x in {a} if x > 0)",
        testes=["{f}([1, -2, 3, 0]) == 2", "{f}([]) == 0"]),
    "dobrar_lista": dict(
        nomes=["dobrar", "dobrar_lista", "duplicar_valores"],
        args=["numeros", "lista", "valores"],
        oque=["retorna uma nova lista com cada número multiplicado por 2",
              "devolve a lista com todos os valores dobrados"],
        corpo="    return [x * 2 for x in {a}]",
        testes=["{f}([1, 2, 3]) == [2, 4, 6]", "{f}([]) == []"]),
    "remover_negativos": dict(
        nomes=["remover_negativos", "sem_negativos", "filtrar_negativos"],
        args=["numeros", "lista", "valores"],
        oque=["retorna uma nova lista sem os números negativos",
              "remove os valores negativos de uma lista"],
        corpo="    return [x for x in {a} if x >= 0]",
        testes=["{f}([1, -2, 3, 0]) == [1, 3, 0]", "{f}([-1]) == []"]),
    "remover_duplicados": dict(
        nomes=["remover_duplicados", "sem_repetidos", "valores_unicos"],
        args=["itens", "lista", "valores"],
        oque=["retorna a lista sem elementos repetidos, mantendo a ordem",
              "remove os itens duplicados de uma lista preservando a ordem"],
        corpo="    return list(dict.fromkeys({a}))",
        testes=["{f}([1, 2, 1, 3, 2]) == [1, 2, 3]", "{f}([]) == []"]),
    "ultimo": dict(
        nomes=["ultimo", "ultimo_elemento", "pegar_ultimo"],
        args=["itens", "lista", "valores"],
        oque=["retorna o último elemento da lista",
              "devolve o item que está na última posição de uma lista"],
        corpo="    return {a}[-1]",
        testes=["{f}([1, 2, 3]) == 3", "{f}(['a']) == 'a'"]),
    "inverter_texto": dict(
        nomes=["inverter", "inverter_texto", "texto_invertido"],
        args=["texto", "frase", "s"],
        oque=["retorna o texto de trás para frente",
              "inverte a ordem dos caracteres de uma string"],
        corpo="    return {a}[::-1]",
        testes=["{f}('abc') == 'cba'", "{f}('') == ''"]),
    "maiusculas": dict(
        nomes=["maiusculas", "para_maiusculas", "gritar"],
        args=["texto", "frase", "s"],
        oque=["retorna o texto todo em letras maiúsculas",
              "converte uma string para maiúsculas"],
        corpo="    return {a}.upper()",
        testes=["{f}('abc') == 'ABC'", "{f}('Oi') == 'OI'"]),
    "contar_vogais": dict(
        nomes=["contar_vogais", "qtd_vogais", "conta_vogais"],
        args=["texto", "frase", "s"],
        oque=["retorna quantas vogais existem no texto",
              "conta as vogais (a, e, i, o, u) de uma string"],
        corpo='    return sum(1 for c in {a}.lower() if c in "aeiou")',
        testes=["{f}('Banana') == 3", "{f}('xyz') == 0"]),
    "contar_palavras": dict(
        nomes=["contar_palavras", "qtd_palavras", "conta_palavras"],
        args=["texto", "frase", "s"],
        oque=["retorna quantas palavras existem no texto",
              "conta as palavras de uma string separadas por espaço"],
        corpo="    return len({a}.split())",
        testes=["{f}('ola mundo bom') == 3", "{f}('') == 0"]),
    "eh_palindromo": dict(
        nomes=["eh_palindromo", "palindromo", "e_palindromo"],
        args=["texto", "frase", "s"],
        oque=["retorna True se o texto for um palíndromo, ignorando espaços e maiúsculas",
              "verifica se uma string é palíndromo, sem considerar espaços e maiúsculas"],
        corpo='    t = {a}.lower().replace(" ", "")\n    return t == t[::-1]',
        testes=["{f}('Ame a ema') == True", "{f}('abc') == False"]),
    "eh_par": dict(
        nomes=["eh_par", "par", "e_par"],
        args=["n", "numero", "x"],
        oque=["retorna True se o número for par e False caso contrário",
              "verifica se um número inteiro é par"],
        corpo="    return {a} % 2 == 0",
        testes=["{f}(4) == True", "{f}(7) == False"]),
    "quadrado": dict(
        nomes=["quadrado", "ao_quadrado", "elevar_quadrado"],
        args=["n", "numero", "x"],
        oque=["retorna o número elevado ao quadrado",
              "calcula o quadrado de um número"],
        corpo="    return {a} * {a}",
        testes=["{f}(3) == 9", "{f}(-2) == 4"]),
    "fatorial": dict(
        nomes=["fatorial", "calcular_fatorial", "fat"],
        args=["n", "numero", "x"],
        oque=["retorna o fatorial de n usando um laço for",
              "calcula o fatorial de um inteiro não negativo"],
        corpo="    resultado = 1\n    for i in range(2, {a} + 1):\n        resultado *= i\n    return resultado",
        testes=["{f}(5) == 120", "{f}(0) == 1"]),
    "celsius_fahrenheit": dict(
        nomes=["celsius_para_fahrenheit", "c_para_f", "converter_temperatura"],
        args=["celsius", "graus", "c"],
        oque=["converte uma temperatura de Celsius para Fahrenheit",
              "retorna a temperatura em Fahrenheit a partir de graus Celsius"],
        corpo="    return {a} * 9 / 5 + 32",
        testes=["{f}(100) == 212", "{f}(0) == 32"]),

    "cubo": dict(
        nomes=["cubo", "ao_cubo", "elevar_cubo"],
        args=["n", "numero", "x"],
        oque=["retorna o número elevado ao cubo",
              "calcula o cubo de um número"],
        corpo="    return {a} ** 3",
        testes=["{f}(2) == 8", "{f}(-3) == -27", "{f}(0) == 0"]),

    # ---- Famílias que NUNCA aparecem no treino (teste de generalização) ----
    "produto": dict(
        inedita=True,
        nomes=["produto", "multiplicar_todos"],
        args=["numeros", "lista"],
        oque=["retorna o produto (multiplicação) de todos os números da lista"],
        corpo="    resultado = 1\n    for x in {a}:\n        resultado *= x\n    return resultado",
        testes=["{f}([2, 3, 4]) == 24", "{f}([]) == 1"]),
    "triplicar_lista": dict(
        inedita=True,
        nomes=["triplicar", "triplicar_lista"],
        args=["numeros", "lista"],
        oque=["retorna uma nova lista com cada número multiplicado por 3"],
        corpo="    return [x * 3 for x in {a}]",
        testes=["{f}([1, 2]) == [3, 6]", "{f}([]) == []"]),
    "minusculas": dict(
        inedita=True,
        nomes=["minusculas", "para_minusculas"],
        args=["texto", "frase"],
        oque=["retorna o texto todo em letras minúsculas"],
        corpo="    return {a}.lower()",
        testes=["{f}('ABC') == 'abc'", "{f}('Oi') == 'oi'"]),
    "soma_positivos": dict(
        inedita=True,
        nomes=["somar_positivos", "soma_positivos"],
        args=["numeros", "lista"],
        oque=["retorna a soma dos números positivos da lista"],
        corpo="    return sum(x for x in {a} if x > 0)",
        testes=["{f}([1, -2, 3]) == 4", "{f}([]) == 0"]),
}

ENUNCIADOS = [
    "Escreva uma função em Python chamada `{f}({a})` que {oque}.",
    "Crie a função `{f}({a})`, que {oque}.",
    "Implemente em Python a função `{f}({a})`: ela {oque}.",
]


def montar_exemplo(familia, f, a, oque, modelo_enunciado):
    fam = FAMILIAS[familia]
    tarefa = modelo_enunciado.format(f=f, a=a, oque=oque)
    codigo = f"def {f}({a}):\n" + fam["corpo"].format(a=a)
    return {
        "familia": familia,
        "tarefa": tarefa,
        "codigo": codigo,
        "testes": [t.format(f=f) for t in fam["testes"]],
    }


def todas_combinacoes(familia):
    fam = FAMILIAS[familia]
    return [(f, a, o, e) for f in fam["nomes"] for a in fam["args"]
            for o in fam["oque"] for e in ENUNCIADOS]


treino, teste = [], []
for familia, fam in FAMILIAS.items():
    combos = todas_combinacoes(familia)
    random.shuffle(combos)
    if fam.get("inedita"):
        # família inteira fica só no teste
        for c in combos[:4]:
            teste.append({**montar_exemplo(familia, *c), "tipo": "inedita"})
    else:
        # 3 combinações vão para o teste, o resto (até 25) para o treino
        for c in combos[:3]:
            teste.append({**montar_exemplo(familia, *c), "tipo": "variacao"})
        for c in combos[3:28]:
            treino.append(montar_exemplo(familia, *c))

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
print(f"### Tarefa:\n{treino[0]['tarefa']}\n### Código:\n{treino[0]['codigo']}")
print("\nTestes:", treino[0]["testes"])
