import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import random

    import marimo as mo
    from ortools.sat.python import cp_model
    from sudoku_schemas import Box, Cube, Path, SudokuConfig, SudokuSolution


@app.cell
def _():
    mo.md(r"""
    # Sudoku genérico como CSP (CP-SAT)

    - **Técnica**: CP-SAT (OR-Tools). A regra "todos diferentes" mapeia diretamente para
      `add_all_different`, e o estado `INFEASIBLE` dá-nos uma forma distinguível de sinalizar
      que o puzzle não tem solução (`UNKNOWN` = tempo esgotado, que não é o mesmo).
    - **Estrutura** (CP-SAT Primer): dados/configuração/solução em classes pydantic
      (`sudoku_schemas.py`) e um `SudokuSolver` que constrói o modelo e devolve
      uma `SudokuSolution`.
    - **`Box`** guarda um dicionário `(linha, coluna) → valor | None`, com as mesmas chaves que as
      variáveis do modelo (`SudokuSolver.x`), por isso o modelo aplica qualquer grupo sem saber a
      sua origem. `Cube`, `Path` e as pistas aleatórias são apenas `Box`es.
    - **Pistas aleatórias**: o grupo de pistas é, ele próprio, um `Box` "todos diferentes", por
      isso as pistas usam valores distintos (logo `k ≤ n²`); com valores repetidos o modelo
      rejeitaria puzzles que têm solução.
    - **Puzzle sem solução**: tenta novas pistas aleatórias até um número máximo de tentativas
      e reporta o insucesso se nenhuma tiver solução.
    - **Sem warm start**: testámos dar como *hint* uma grelha válida conhecida e não houve ganho
      mensurável (n = 3 a 5), por isso não o usamos.
    """)
    return


@app.function
def grupos_sudoku(n: int) -> list[Box]:
    """R6 - todas as linhas, colunas e blocos n x n de uma grelha n² x n²."""
    N = n * n
    linhas = [Path((k, 0), (k, N - 1), n) for k in range(N)]
    colunas = [Path((0, k), (N - 1, k), n) for k in range(N)]
    blocos = [Cube(i, j, n) for i in range(n) for j in range(n)]
    return [*linhas, *colunas, *blocos]


@app.function
def pistas_aleatorias(n: int = 3, k: int | None = None, seed: int | None = None) -> Box:
    """R4 - k células aleatórias (por omissão k = n), fixas a valores aleatórios distintos."""
    rng = random.Random(seed)
    N = n * n
    k = n if k is None else k
    celulas = rng.sample([(i, j) for i in range(N) for j in range(N)], k)
    valores = rng.sample(range(1, N + 1), k)
    return Box(block_size=n, cells=dict(zip(celulas, valores, strict=True)))


@app.class_definition
class SudokuSolver:
    def __init__(self, n: int = 3, config: SudokuConfig | None = None):
        self.N = n * n
        self.config = config or SudokuConfig()
        self.model = cp_model.CpModel()
        # R5 - uma variável inteira por célula, em [1, n²]
        self.x = {
            (i, j): self.model.new_int_var(1, self.N, f"x_{i}_{j}")
            for i in range(self.N)
            for j in range(self.N)
        }
        self.solver = cp_model.CpSolver()

    def add(self, *groups: Box) -> None:
        # R5 - para cada grupo, seja qual for a origem: todos diferentes + células fixas
        for g in groups:
            if g.N != self.N:
                raise ValueError(f"Grupo para grelha {g.N}x{g.N}, modelo é {self.N}x{self.N}")
            self.model.add_all_different(self.x[c] for c in g.cells)
            for c, val in g.cells.items():
                if val is not None:
                    self.model.add(self.x[c] == val)

    def solve(self, time_limit: float | None = None) -> SudokuSolution:
        self.solver.parameters.max_time_in_seconds = time_limit or self.config.time_limit
        self.solver.parameters.log_search_progress = self.config.log_search_progress
        status = self.solver.solve(self.model)
        sol = SudokuSolution(status=self.solver.status_name(status), tempo=self.solver.wall_time)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            sol.matrix = [
                [self.solver.value(self.x[i, j]) for j in range(self.N)] for i in range(self.N)
            ]
        return sol


@app.function
def sudoku_aleatorio(
    n: int = 3, k: int | None = None, seed: int | None = None, tentativas: int = 10
) -> tuple[Box, SudokuSolution]:
    """Gera pistas → monta linhas + colunas + blocos + pistas → resolve.

    Se as pistas não tiverem solução, tenta novas (até `tentativas`) e devolve a última."""
    rng = random.Random(seed)
    for _ in range(tentativas):
        pistas = pistas_aleatorias(n, k, rng.randrange(2**32))
        s = SudokuSolver(n)
        s.add(*grupos_sudoku(n), pistas)
        sol = s.solve()
        if sol.matrix is not None:
            break
    return pistas, sol


@app.function
def mostrar(sol: SudokuSolution) -> str:
    if sol.matrix is None:
        return f"Sem solução ({sol.status})"
    largura = len(str(len(sol.matrix)))
    return "\n".join(" ".join(f"{v:>{largura}}" for v in linha) for linha in sol.matrix)


@app.cell
def _():
    pistas_3, sol_3 = sudoku_aleatorio(n=3, seed=0)
    pistas_2, sol_2 = sudoku_aleatorio(n=2, seed=0)
    mo.vstack(
        [
            mo.md(f"**n = 3** ({sol_3.status}, {sol_3.tempo:.3f}s), pistas `{pistas_3.cells}`"),
            mo.plain_text(mostrar(sol_3)),
            mo.md(f"**n = 2** ({sol_2.status}, {sol_2.tempo:.3f}s), pistas `{pistas_2.cells}`"),
            mo.plain_text(mostrar(sol_2)),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
