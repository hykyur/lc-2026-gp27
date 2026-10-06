from pydantic import BaseModel, Field, model_validator


class Box(BaseModel):
    """Grupo genérico de células com restrição "todos diferentes" (R1).

    Só conhece o tamanho da grelha (N = block_size²), nada sobre linhas, colunas ou blocos.
    """

    block_size: int = Field(ge=1)
    cells: dict[tuple[int, int], int | None] = {}

    @model_validator(mode="after")
    def _validar_cells(self) -> "Box":
        for (i, j), val in self.cells.items():
            self._validar(i, j, val)
        return self

    @property
    def N(self) -> int:
        return self.block_size**2

    def _validar(self, i: int, j: int, val: int | None) -> None:
        if not (0 <= i < self.N and 0 <= j < self.N):
            raise ValueError(f"Célula ({i}, {j}) fora da grelha {self.N}x{self.N}")
        if val is not None and not 1 <= val <= self.N:
            raise ValueError(f"Valor {val} fora do intervalo [1, {self.N}]")

    def add(self, i: int, j: int, val: int | None = None) -> None:
        self._validar(i, j, val)
        self.cells[i, j] = val

    def representation(self) -> list[list[int]]:
        matrix = [[0] * self.N for _ in range(self.N)]
        for (i, j), val in self.cells.items():
            matrix[i][j] = val or 0
        return matrix


class Cube(Box):
    """Bloco n x n cujo canto superior esquerdo é (i*n, j*n) (R2)."""

    def __init__(self, i: int, j: int, n: int = 3):
        super().__init__(block_size=n)
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"Índices de bloco ({i}, {j}) fora de [0, {n - 1}]")
        for row in range(i * n, (i + 1) * n):
            for col in range(j * n, (j + 1) * n):
                self.add(row, col)


class Path(Box):
    """Troço reto (horizontal ou vertical) de `inicio` a `fim`, inclusive,
    em qualquer sentido (R3)."""

    def __init__(self, inicio: tuple[int, int], fim: tuple[int, int], n: int = 3):
        super().__init__(block_size=n)
        (i0, j0), (i1, j1) = inicio, fim
        if i0 != i1 and j0 != j1:
            raise ValueError("O troço tem de ser horizontal ou vertical")
        di = (i1 > i0) - (i1 < i0)
        dj = (j1 > j0) - (j1 < j0)
        for k in range(max(abs(i1 - i0), abs(j1 - j0)) + 1):
            self.add(i0 + k * di, j0 + k * dj)


class SudokuConfig(BaseModel):
    time_limit: float = 30.0
    log_search_progress: bool = False


class SudokuSolution(BaseModel):
    status: str
    tempo: float = 0.0  # wall time do solver, em segundos
    matrix: list[list[int]] | None = None  # None = sem solução (ou tempo esgotado)
