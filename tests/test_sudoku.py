import pytest
from conftest import VIAVEL
from sudoku import (
    SudokuSolver,
    grupos_sudoku,
    pistas_aleatorias,
    sudoku_aleatorio,
)
from sudoku_schemas import Box, Cube, Path


def _valida(grid, n):
    """Verificador independente (não usa Cube/Path): linhas, colunas e blocos são 1..n²."""
    N = n * n
    alvo = set(range(1, N + 1))
    assert len(grid) == N and all(len(linha) == N for linha in grid)
    for k in range(N):
        assert set(grid[k]) == alvo, f"linha {k}"
        assert {grid[i][k] for i in range(N)} == alvo, f"coluna {k}"
    for bi in range(n):
        for bj in range(n):
            bloco = {grid[bi * n + a][bj * n + b] for a in range(n) for b in range(n)}
            assert bloco == alvo, f"bloco {bi},{bj}"


def _resolver(n, *groups):
    s = SudokuSolver(n)
    s.add(*groups)
    return s, s.solve()


# R1 - Box


def test_r1_box_vazio_por_omissao():
    assert Box(block_size=3).cells == {}


def test_r1_add_celula_livre_e_fixa():
    b = Box(block_size=3)
    b.add(0, 0)
    b.add(8, 8, 9)
    assert b.cells == {(0, 0): None, (8, 8): 9}


def test_r1_add_repetido_sobrescreve():
    b = Box(block_size=3)
    b.add(1, 1, 5)
    b.add(1, 1, 7)
    assert b.cells == {(1, 1): 7}


@pytest.mark.parametrize(
    "i, j, val",
    [(-1, 0, None), (0, -1, None), (9, 0, None), (0, 9, None), (0, 0, 0), (0, 0, 10)],
    ids=["i-negativo", "j-negativo", "i-fora", "j-fora", "val-zero", "val-acima"],
)
def test_r1_add_rejeita_fora_da_grelha(i, j, val):
    with pytest.raises(ValueError):
        Box(block_size=3).add(i, j, val)


@pytest.mark.parametrize("cells", [{(9, 0): None}, {(0, 0): 10}], ids=["celula-fora", "valor-fora"])
def test_r1_construtor_rejeita_cells_invalidas(cells):
    with pytest.raises(ValueError):
        Box(block_size=3, cells=cells)


def test_r1_limites_dependem_de_n():
    Box(block_size=3).add(4, 0, 9)
    with pytest.raises(ValueError):
        Box(block_size=2).add(4, 0)
    with pytest.raises(ValueError):
        Box(block_size=2).add(0, 0, 5)


def test_r1_representation():
    b = Box(block_size=2, cells={(0, 1): 3, (2, 2): None})
    assert b.representation() == [[0, 3, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


# R2 - Cube


@pytest.mark.parametrize("n", [2, 3])
def test_r2_cubes_particionam_a_grelha(n):
    vistos = []
    for i in range(n):
        for j in range(n):
            c = Cube(i, j, n)
            esperado = {(r, k) for r in range(i * n, i * n + n) for k in range(j * n, j * n + n)}
            assert set(c.cells) == esperado
            assert all(v is None for v in c.cells.values())
            vistos += list(c.cells)
    assert len(vistos) == len(set(vistos)) == n**4


@pytest.mark.parametrize("i, j", [(3, 0), (0, 3), (-1, 0)])
def test_r2_indice_de_bloco_invalido(i, j):
    with pytest.raises(ValueError):
        Cube(i, j, 3)


# R3 - Path


@pytest.mark.parametrize(
    "inicio, fim, esperado",
    [
        ((2, 1), (2, 4), [(2, 1), (2, 2), (2, 3), (2, 4)]),
        ((2, 4), (2, 1), [(2, 4), (2, 3), (2, 2), (2, 1)]),
        ((0, 5), (3, 5), [(0, 5), (1, 5), (2, 5), (3, 5)]),
        ((3, 5), (0, 5), [(3, 5), (2, 5), (1, 5), (0, 5)]),
        ((4, 4), (4, 4), [(4, 4)]),
    ],
    ids=["horizontal", "horizontal-inverso", "vertical", "vertical-inverso", "uma-celula"],
)
def test_r3_path_em_qualquer_sentido(inicio, fim, esperado):
    assert list(Path(inicio, fim, 3).cells) == esperado


@pytest.mark.parametrize(
    "inicio, fim", [((0, 0), (2, 2)), ((0, 0), (0, 9))], ids=["diagonal", "fora-da-grelha"]
)
def test_r3_path_invalido(inicio, fim):
    with pytest.raises(ValueError):
        Path(inicio, fim, 3)


# R4 - pistas aleatórias


@pytest.mark.parametrize("n, k", [(2, None), (3, None), (3, 9)])
def test_r4_pistas(n, k):
    p = pistas_aleatorias(n, k, seed=42)
    assert isinstance(p, Box)
    assert len(p.cells) == (n if k is None else k)
    valores = list(p.cells.values())
    assert all(v is not None and 1 <= v <= n * n for v in valores)
    assert len(set(valores)) == len(valores)  # o Box das pistas também é "todos diferentes"


def test_r4_pistas_reprodutiveis_com_seed():
    assert pistas_aleatorias(3, seed=7).cells == pistas_aleatorias(3, seed=7).cells


# R5 - modelo


def test_r5_uma_variavel_por_celula_em_1_n2():
    s = SudokuSolver(3)
    assert len(s.x) == 81
    assert all(list(v.domain) == [1, 9] for v in s.model.proto.variables)


def test_r5_grupo_com_valor_repetido_e_impossivel():
    _, sol = _resolver(3, Box(block_size=3, cells={(0, 0): 5, (4, 7): 5}))
    assert sol.status == "INFEASIBLE"
    assert sol.matrix is None


def test_r5_celula_fixa_a_valores_diferentes_e_impossivel():
    a, b = Box(block_size=3, cells={(0, 0): 1}), Box(block_size=3, cells={(0, 0): 2})
    _, sol = _resolver(3, a, b)
    assert sol.status == "INFEASIBLE"


def test_r5_linha_e_coluna_em_conflito():
    # A linha 0 obriga (0, 8) = 9, mas a coluna 8 já tem o 9 em (1, 8)
    linha = Path((0, 0), (0, 8), 3)
    for j in range(8):
        linha.add(0, j, j + 1)
    coluna = Path((0, 8), (8, 8), 3)
    coluna.add(1, 8, 9)
    _, sol = _resolver(3, linha, coluna)
    assert sol.status == "INFEASIBLE"


def test_r5_aceita_grupos_de_qualquer_origem():
    box = Box(block_size=3, cells={(4, 4): 1})
    _, sol = _resolver(3, box, Cube(0, 0), Path((8, 0), (8, 8)), pistas_aleatorias())
    assert sol.status in VIAVEL
    assert sol.matrix[4][4] == 1


def test_r5_rejeita_grupo_de_outra_grelha():
    with pytest.raises(ValueError):
        SudokuSolver(3).add(Box(block_size=2))


# R6 - Sudoku completo


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_r6_fluxo_completo(n, seed):
    pistas, sol = sudoku_aleatorio(n, seed=seed)
    assert sol.status in VIAVEL
    _valida(sol.matrix, n)
    for (i, j), v in pistas.cells.items():
        assert sol.matrix[i][j] == v


def test_r6_sem_pistas_ainda_e_um_sudoku():
    _, sol = _resolver(3, *grupos_sudoku(3))
    _valida(sol.matrix, 3)
