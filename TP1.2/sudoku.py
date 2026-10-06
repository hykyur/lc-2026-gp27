# /// script
# requires-python = ">=3.13"
# dependencies = ["marimo>=0.25.0", "ortools>=9.15.6755"]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

# Estas definições ficam disponíveis para todas as células do notebook.
with app.setup:
    import random
    from collections.abc import Mapping

    import marimo as mo
    from ortools.sat.python import cp_model

    def validar_n(n):
        """Verifica se n é inteiro positivo e devolve o lado da grelha: n².

        Exemplo: n=3 significa blocos 3X3 numa grelha 9X9.
        """
        if type(n) is not int:
            raise ValueError("n tem de ser um inteiro.")
        if n < 1:
            raise ValueError("n tem de ser positivo.")
        return n * n

    # R1: um grupo guarda células. As regras são aplicadas pelo modelo CSP.
    class box:
        """Grupo genérico de células, sem conhecer linhas, blocos ou diagonais.

        n: lado de cada bloco; size: lado da grelha completa.
        cells: dicionário {(linha, coluna): valor ou None}.
        None significa uma célula livre; um inteiro significa uma célula fixa.
        As coordenadas começam em zero.
        """

        def __init__(self, cells=None, n=3):
            """Cria um grupo vazio ou com células fornecidas.

            Aceita um dicionário com valores, ou uma sequência de coordenadas
            livres. Exemplo: box({(0, 0): 5, (0, 1): None}, n=3).
            """
            self.n = n
            self.size = validar_n(n)
            self.cells = {}

            if cells is not None:
                if isinstance(cells, Mapping):
                    for (linha, coluna), valor in cells.items():
                        self.add(linha, coluna, valor)
                else:
                    for linha, coluna in cells:
                        self.add(linha, coluna)

        def validar_coordenadas(self, linha, coluna):
            """Levanta ValueError se a posição não existir na grelha."""
            if type(linha) is not int or type(coluna) is not int:
                raise ValueError("A linha e a coluna têm de ser inteiros.")
            if linha < 0 or linha >= self.size:
                raise ValueError("Linha fora da grelha.")
            if coluna < 0 or coluna >= self.size:
                raise ValueError("Coluna fora da grelha.")

        def add(self, i, j, val=None):
            """Acrescenta a célula (i, j), opcionalmente fixa em val.

            i é a linha e j é a coluna. Valida a posição e o valor.
            Se a célula já existir, atualiza-a. Devolve o próprio grupo.
            """
            self.validar_coordenadas(i, j)

            if val is not None:
                if type(val) is not int:
                    raise ValueError("O valor tem de ser inteiro ou None.")
                if val < 1 or val > self.size:
                    raise ValueError(f"O valor deve estar entre 1 e {self.size}.")

            self.cells[i, j] = val
            return self

        def to_matrix(self):
            """Devolve uma matriz nova com os valores fixos e zeros no resto.

            Os zeros servem para apresentar as pistas; não entram na solução.
            Alterar esta matriz não altera as células guardadas no grupo.
            """
            matriz = []
            for _ in range(self.size):
                matriz.append([0] * self.size)

            for (linha, coluna), valor in self.cells.items():
                if valor is not None:
                    matriz[linha][coluna] = valor
            return matriz

    # R2: cube herda de box e escolhe as células de um bloco n×n.
    class cube(box):
        """Bloco regular identificado pelos índices de bloco (i, j)."""

        def __init__(self, i, j, n=3):
            """Cria o bloco que começa na célula (i*n, j*n).

            Para n=3, cube(0, 0) é o bloco superior esquerdo.
            Os índices dos blocos vão de 0 a n-1.
            """
            # super chama o construtor da classe mãe, box.
            super().__init__(n=n)

            if type(i) is not int or type(j) is not int:
                raise ValueError("Os índices do bloco têm de ser inteiros.")
            if i < 0 or i >= n or j < 0 or j >= n:
                raise ValueError("Índices de bloco fora do intervalo [0, n-1].")

            primeira_linha = i * n
            primeira_coluna = j * n
            for linha in range(primeira_linha, primeira_linha + n):
                for coluna in range(primeira_coluna, primeira_coluna + n):
                    self.add(linha, coluna)

    # R3: path herda de box e escolhe um percurso horizontal ou vertical.
    class path(box):
        """Grupo com todas as células entre dois extremos, inclusive."""

        def __init__(self, inicio, fim, n=3):
            """Cria um percurso reto, nos dois sentidos.

            Exemplo: path((0, 0), (0, 8), n=3) representa a primeira linha.
            Uma diagonal não é um path; será construída usando box.
            """
            super().__init__(n=n)
            linha_inicial, coluna_inicial = inicio
            linha_final, coluna_final = fim
            self.validar_coordenadas(linha_inicial, coluna_inicial)
            self.validar_coordenadas(linha_final, coluna_final)

            if linha_inicial == linha_final:
                passo = 1
                if coluna_final < coluna_inicial:
                    passo = -1

                # range exclui o limite final; somar passo inclui o extremo.
                for coluna in range(coluna_inicial, coluna_final + passo, passo):
                    self.add(linha_inicial, coluna)

            elif coluna_inicial == coluna_final:
                passo = 1
                if linha_final < linha_inicial:
                    passo = -1

                for linha in range(linha_inicial, linha_final + passo, passo):
                    self.add(linha, coluna_inicial)
            else:
                raise ValueError("Um path tem de ser horizontal ou vertical.")

    def pistas_aleatorias(n=3, k=None, seed=None):
        """R4: devolve um box com k posições distintas e valores aleatórios.

        k é o número de pistas; por omissão, usa n.
        seed fixa permite repetir as pistas; seed=None permite variar a geração.
        Os valores podem repetir-se e tornar o puzzle impossível.
        """
        tamanho = validar_n(n)
        if k is None:
            k = n
        if type(k) is not int:
            raise ValueError("k tem de ser um inteiro.")
        if k < 0 or k > tamanho * tamanho:
            raise ValueError("k está fora do número de células da grelha.")

        # Construir as coordenadas diretamente evita conversões de índices.
        posicoes = []
        for linha in range(tamanho):
            for coluna in range(tamanho):
                posicoes.append((linha, coluna))

        gerador = random.Random(seed)
        escolhidas = gerador.sample(posicoes, k)  # sample não repete posições.
        pistas = box(n=n)
        for linha, coluna in escolhidas:
            valor = gerador.randint(1, tamanho)
            pistas.add(linha, coluna, valor)
        return pistas

    class modelo_csp:
        """R5: variáveis da grelha e restrições provenientes de qualquer box.

        O modelo não distingue uma linha, um bloco ou uma diagonal.
        Cada célula tem uma única variável, partilhada por todos os seus grupos.
        """

        def __init__(self, n=3):
            """Cria uma variável por célula, com domínio inteiro de 1 a n²."""
            self.n = n
            self.size = validar_n(n)
            self.model = cp_model.CpModel()
            self.variables = {}
            self.status = None

            for linha in range(self.size):
                for coluna in range(self.size):
                    nome = f"x_{linha}_{coluna}"
                    variavel = self.model.new_int_var(1, self.size, nome)
                    self.variables[linha, coluna] = variavel

        def add(self, *grupos):
            """Acrescenta AllDifferent e os valores fixos de cada grupo.

            *grupos aceita vários argumentos: modelo.add(linha, bloco, pistas).
            Devolve o próprio modelo, permitindo chamar .solve() a seguir.
            """
            # Verificar os grupos antes de modificar o modelo.
            for grupo in grupos:
                if not isinstance(grupo, box):
                    raise ValueError("Cada grupo tem de ser um box ou uma subclasse.")
                if grupo.n != self.n:
                    raise ValueError("O grupo e o modelo têm de ter o mesmo n.")

            for grupo in grupos:
                variaveis_do_grupo = []
                for posicao in grupo.cells:
                    variaveis_do_grupo.append(self.variables[posicao])

                if len(variaveis_do_grupo) > 1:
                    self.model.add_all_different(variaveis_do_grupo)

                for posicao, valor in grupo.cells.items():
                    if valor is not None:
                        self.model.add(self.variables[posicao] == valor)

            self.status = None
            return self

        def solve(self):
            """Devolve a grelha preenchida ou None se não existir solução.

            Um estado sem conclusão levanta RuntimeError: não é tratado como
            uma prova de impossibilidade. status guarda o resultado do solver.
            """
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            estado = solver.solve(self.model)
            self.status = solver.status_name(estado)

            if estado == cp_model.INFEASIBLE:
                return None
            if estado not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                raise RuntimeError(f"O solver não concluiu: {self.status}.")

            grelha = []
            for linha in range(self.size):
                valores_da_linha = []
                for coluna in range(self.size):
                    variavel = self.variables[linha, coluna]
                    valores_da_linha.append(solver.value(variavel))
                grelha.append(valores_da_linha)
            return grelha

    def montar_sudoku(n=3, pistas=None, extras=()):
        """R6: devolve o modelo com linhas, colunas, blocos e pistas.

        Se pistas for None, gera um grupo aleatório.
        extras permite acrescentar grupos, como os das diagonais.
        Esta função prepara o modelo; a resolução é feita por solve().
        """
        tamanho = validar_n(n)
        modelo = modelo_csp(n=n)

        for linha in range(tamanho):
            grupo = path((linha, 0), (linha, tamanho - 1), n=n)
            modelo.add(grupo)

        for coluna in range(tamanho):
            grupo = path((0, coluna), (tamanho - 1, coluna), n=n)
            modelo.add(grupo)

        for linha_bloco in range(n):
            for coluna_bloco in range(n):
                modelo.add(cube(linha_bloco, coluna_bloco, n=n))

        if pistas is None:
            pistas = pistas_aleatorias(n=n)
        modelo.add(pistas)

        for grupo in extras:
            modelo.add(grupo)
        return modelo

    def diagonais(n=3):
        """Bónus: devolve os dois grupos das diagonais principais.

        Os grupos são box comuns, com células livres. Basta entregá-los ao
        modelo para aplicar AllDifferent também nas diagonais.
        """
        tamanho = validar_n(n)
        principal = box(n=n)
        secundaria = box(n=n)

        for linha in range(tamanho):
            principal.add(linha, linha)
            secundaria.add(linha, tamanho - 1 - linha)
        return principal, secundaria

    def validar_solucao(grelha, n=3, pistas=None, extras=()):
        """Confirma dimensões, linhas, colunas, blocos, pistas e grupos extras.

        Devolve True se a solução estiver correta; caso contrário, levanta
        ValueError. A verificação não depende do resultado indicado pelo solver.
        """
        tamanho = validar_n(n)
        if grelha is None:
            raise ValueError("Não foi fornecida uma solução.")
        if len(grelha) != tamanho:
            raise ValueError("Número de linhas incorreto.")
        for linha in grelha:
            if len(linha) != tamanho:
                raise ValueError("Número de colunas incorreto.")
            for valor in linha:
                if type(valor) is not int:
                    raise ValueError("Os valores da solução têm de ser inteiros.")

        # set cria um conjunto. Uma repetição elimina um dos valores esperados.
        esperados = set(range(1, tamanho + 1))
        for linha in grelha:
            if set(linha) != esperados:
                raise ValueError("Uma linha tem valores incorretos ou repetidos.")

        for coluna in range(tamanho):
            valores = set()
            for linha in range(tamanho):
                valores.add(grelha[linha][coluna])
            if valores != esperados:
                raise ValueError("Uma coluna tem valores incorretos ou repetidos.")

        # Percorrer os blocos diretamente para verificar a solução.
        for primeira_linha in range(0, tamanho, n):
            for primeira_coluna in range(0, tamanho, n):
                valores = set()
                for linha in range(primeira_linha, primeira_linha + n):
                    for coluna in range(primeira_coluna, primeira_coluna + n):
                        valores.add(grelha[linha][coluna])
                if valores != esperados:
                    raise ValueError("Um bloco tem valores incorretos ou repetidos.")

        grupos = list(extras)
        if pistas is not None:
            grupos.append(pistas)
        for grupo in grupos:
            if grupo.n != n:
                raise ValueError("Grupo com dimensão incompatível.")
            valores = set()
            for (linha, coluna), valor_fixo in grupo.cells.items():
                valor_encontrado = grelha[linha][coluna]
                if valor_encontrado in valores:
                    raise ValueError("Um grupo tem valores repetidos.")
                valores.add(valor_encontrado)
                if valor_fixo is not None and valor_encontrado != valor_fixo:
                    raise ValueError("Uma pista não foi respeitada.")
        return True

    def gerar_e_resolver(n=3, k=None, seed=27, max_tentativas=100):
        """Gera pistas, monta o modelo, resolve e valida a solução.

        Se as pistas forem incompatíveis, tenta novas pistas até ao limite.
        Devolve (grelha, pistas, número de tentativas).
        seed=27 repete a sequência de geração; seed=None permite variá-la.
        """
        tamanho = validar_n(n)
        if type(max_tentativas) is not int or max_tentativas < 1:
            raise ValueError("max_tentativas tem de ser um inteiro positivo.")
        if k is not None:
            if type(k) is not int or k < 0:
                raise ValueError("k tem de ser um inteiro não negativo.")
            # Todas as pistas pertencem a um único grupo AllDifferent.
            if k > tamanho:
                raise ValueError("Mais de n² pistas num único grupo são impossíveis.")

        gerador = random.Random(seed)
        for tentativa in range(1, max_tentativas + 1):
            # O gerador principal escolhe uma seed para cada tentativa.
            seed_tentativa = gerador.randint(0, 1_000_000)
            pistas = pistas_aleatorias(n=n, k=k, seed=seed_tentativa)
            modelo = montar_sudoku(n=n, pistas=pistas)
            grelha = modelo.solve()

            if grelha is not None:
                validar_solucao(grelha, n=n, pistas=pistas)
                return grelha, pistas, tentativa
        raise RuntimeError("Não foram encontradas pistas solúveis dentro do limite.")

    def mostrar_grelha(grelha, n=3):
        """Devolve texto com a grelha, separadores de blocos e pontos nos zeros."""
        tamanho = validar_n(n)
        largura = len(str(tamanho))  # Alinhar também números com vários dígitos.
        linhas_de_texto = []

        for linha in range(tamanho):
            partes = []
            for coluna in range(tamanho):
                if coluna > 0 and coluna % n == 0:
                    partes.append("|")

                valor = grelha[linha][coluna]
                if valor == 0:
                    texto = "."
                else:
                    texto = str(valor)
                partes.append(texto.rjust(largura))

            texto_da_linha = " ".join(partes)
            if linha > 0 and linha % n == 0:
                linhas_de_texto.append("-" * len(texto_da_linha))
            linhas_de_texto.append(texto_da_linha)
        return "\n".join(linhas_de_texto)

    def executar_testes():
        """Verifica os requisitos obrigatórios e devolve os exemplos resolvidos.

        assert interrompe os testes se uma condição for falsa.
        A validação do bónus é feita na célula que apresenta o Sudoku diagonal.
        """
        def rejeita(funcao, *argumentos, **opcoes):
            """Confirma que uma chamada com dados inválidos levanta ValueError."""
            try:
                funcao(*argumentos, **opcoes)
            except ValueError:
                return
            raise AssertionError("A chamada deveria ter levantado ValueError.")

        resultados = []
        for n in (3, 2):
            tamanho = n * n
            grupo = box({(0, 0): 1, (1, 1): None}, n=n)
            matriz = grupo.to_matrix()
            assert len(matriz) == tamanho
            assert matriz[0][0] == 1
            assert matriz[1][1] == 0
            assert matriz[0][1] == 0

            # Coordenadas e valores fora do domínio têm de ser rejeitados.
            rejeita(grupo.add, -1, 0)
            rejeita(grupo.add, tamanho, 0)
            rejeita(grupo.add, 0, -1)
            rejeita(grupo.add, 0, tamanho)
            rejeita(grupo.add, 0, 0, 0)
            rejeita(grupo.add, 0, 0, tamanho + 1)
            rejeita(grupo.add, 0.5, 0)
            rejeita(grupo.add, 0, 0, True)

            bloco = cube(0, 0, n=n)
            assert len(bloco.cells) == n * n
            assert (n - 1, n - 1) in bloco.cells
            rejeita(cube, n, 0, n=n)

            # Verificar horizontal e vertical, nos dois sentidos.
            percursos = [
                ((0, 0), (0, tamanho - 1)),
                ((0, tamanho - 1), (0, 0)),
                ((0, 0), (tamanho - 1, 0)),
                ((tamanho - 1, 0), (0, 0)),
            ]
            for inicio, fim in percursos:
                celulas = list(path(inicio, fim, n=n).cells)
                assert len(celulas) == tamanho
                assert celulas[0] == inicio
                assert celulas[-1] == fim
            assert len(path((1, 1), (1, 1), n=n).cells) == 1
            rejeita(path, (0, 0), (1, 1), n=n)
            rejeita(path, (0, 0), (0, tamanho), n=n)

            # Fluxo completo: gerar pistas -> montar -> resolver -> validar.
            grelha, pistas, tentativas = gerar_e_resolver(n=n, seed=27)
            assert len(pistas.cells) == n
            assert validar_solucao(grelha, n=n, pistas=pistas)
            resultados.append((n, grelha, pistas, tentativas))

            # Um puzzle contraditório tem de devolver None.
            pistas_erradas = box({(0, 0): 1, (0, 1): 1}, n=n)
            modelo = montar_sudoku(n=n, pistas=pistas_erradas)
            assert modelo.solve() is None
            assert modelo.status == "INFEASIBLE"

        rejeita(box, n=0)
        rejeita(modelo_csp(2).add, box(n=3))
        rejeita(pistas_aleatorias, n=2, k=17)
        rejeita(gerar_e_resolver, n=2, k=5)
        return resultados


@app.cell
def _():
    mo.md(r"""
    # Sudoku genérico como CSP

    O fluxo é: **criar grupos → gerar pistas → montar regras → resolver → validar**.
    `n` é o lado de cada bloco: `n=3` dá uma grelha 9×9; `n=2` dá uma grelha 4×4.
    Todas as coordenadas começam em zero.

    Um CSP tem **variáveis** (células), **domínios** (números de 1 a $n^2$)
    e **restrições** (valores diferentes em cada grupo e pistas fixas).
    Usamos CP-SAT do OR-Tools porque permite representar estas regras diretamente.
    Cada célula tem uma variável partilhada pelos grupos a que pertence.

    | Requisito | Definição no código | Papel |
    |---|---|---|
    | R1 | `box` | Guarda células num dicionário e apresenta-as como matriz |
    | R2 | `cube` | Escolhe as células de um bloco |
    | R3 | `path` | Escolhe um percurso horizontal ou vertical |
    | R4 | `pistas_aleatorias` | Escolhe posições e valores aleatórios |
    | R5 | `modelo_csp` | Aplica as restrições dos grupos e resolve |
    | R6 | `montar_sudoku` | Junta linhas, colunas, blocos e pistas |
    | Bónus | `diagonais` | Acrescenta as duas diagonais como grupos `box` |

    `cube` e `path` herdam de `box`: aproveitam os seus métodos e apenas
    escolhem posições diferentes. `self` representa o próprio objeto;
    `super().__init__` prepara os atributos herdados de `box`.
    Os nomes `box`, `cube`, `path`, `add`, `to_matrix` e `solve` foram mantidos
    para facilitar a correspondência com o enunciado e a versão anterior.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Regras e pistas

    `box.cells` guarda `(linha, coluna) → valor ou None`.
    `None` significa célula livre. `to_matrix()` usa zero para apresentar uma
    célula livre ou ausente; zero nunca é um valor permitido na solução.

    Para cada grupo, `modelo_csp.add` aplica **AllDifferent** e fixa as pistas.
    Uma linha com $n^2$ valores diferentes entre 1 e $n^2$ contém necessariamente
    todos os números desse intervalo. A mesma ideia aplica-se a colunas e blocos.

    Conforme o enunciado, as pistas também formam **um único grupo AllDifferent**.
    Por isso, duas pistas iguais tornam esse grupo impossível, mesmo afastadas,
    e mais de $n^2$ pistas num grupo nunca têm solução. A geração mantém os valores
    aleatórios e `gerar_e_resolver` tenta novas pistas quando não há solução,
    até ao limite indicado em `max_tentativas` (por omissão, 100).

    `solve()` devolve a matriz resolvida ou `None` se provar que não há solução.
    Outros estados sem conclusão levantam uma exceção. Poucas pistas podem
    admitir várias soluções: não é garantida a unicidade.

    **Parâmetros:** `k` é o número de pistas; `seed` inicia o gerador aleatório.
    `seed=27` permite repetir uma sequência de geração.
    Para variar as pistas, usa `gerar_e_resolver(n=3, seed=None)`.
    """)
    return


@app.cell
def _():
    # Ao executar o notebook, os testes correm automaticamente.
    resultados = executar_testes()
    mo.md("**Testes obrigatórios concluídos:** grelhas 9×9 e 4×4, pistas "
          "preservadas, percursos nos dois sentidos, entradas inválidas "
          "e puzzles sem solução.")
    return (resultados,)


@app.cell
def _(resultados):
    # Construir a apresentação passo a passo, com variáveis locais ao Marimo.
    _paineis = []
    for _n, _grelha, _pistas, _tentativas in resultados:
        _texto_pistas = mostrar_grelha(_pistas.to_matrix(), n=_n)
        _texto_solucao = mostrar_grelha(_grelha, n=_n)
        _texto = f"""
### Sudoku {_n * _n}×{_n * _n}

{len(_pistas.cells)} pistas; {_tentativas} tentativa(s).

Pistas (`.` = célula livre):

```text
{_texto_pistas}
```

Solução validada:

```text
{_texto_solucao}
```
"""
        _paineis.append(mo.md(_texto))
    mo.vstack(_paineis)
    return


@app.cell
def _():
    # Bónus: os dois grupos adicionais passam pelo mesmo modelo CSP.
    grupos_diagonais = diagonais(n=2)
    pistas_diagonais = pistas_aleatorias(n=2, k=1, seed=27)
    modelo_diagonal = montar_sudoku(n=2, pistas=pistas_diagonais, extras=grupos_diagonais)
    grelha_diagonal = modelo_diagonal.solve()
    validar_solucao(grelha_diagonal, n=2, pistas=pistas_diagonais, extras=grupos_diagonais)

    mo.md(f"""
## Bónus: Sudoku diagonal 4×4

As duas diagonais também têm de conter todos os números de 1 a 4.
`diagonais()` constrói dois grupos `box`, sem alterar o modelo CSP.

Pistas:

```text
{mostrar_grelha(pistas_diagonais.to_matrix(), n=2)}
```

Solução com diagonais validadas:

```text
{mostrar_grelha(grelha_diagonal, n=2)}
```
""")
    return (grelha_diagonal,)


@app.cell
def _():
    mo.md("""
    ## Experimentar

    ```python
    grelha, pistas, tentativas = gerar_e_resolver(n=3, k=3, seed=None)
    print(mostrar_grelha(grelha, n=3))
    ```

    Para construir um Sudoku diagonal:

    ```python
    grupos = diagonais(n=2)
    pistas = pistas_aleatorias(n=2, k=1)
    modelo = montar_sudoku(n=2, pistas=pistas, extras=grupos)
    grelha = modelo.solve()
    if grelha is not None:
        validar_solucao(grelha, n=2, pistas=pistas, extras=grupos)
        print(mostrar_grelha(grelha, n=2))
    ```

    Cada classe e função tem uma descrição entre aspas triplas (docstring).
    Os comentários explicam os passos principais. `for` percorre os elementos;
    `range` gera índices; `raise` comunica um erro; `return` devolve o resultado.
    Nas células de apresentação, nomes iniciados por `_` são locais ao Marimo.
    Os testes obrigatórios cobrem 9×9 e 4×4; o exemplo do bónus verifica 4×4.
    """)
    return


# Executar o notebook quando este ficheiro é iniciado diretamente.
if __name__ == "__main__":
    app.run()
