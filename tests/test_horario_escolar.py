import pytest
from conftest import DADOS, VIAVEL
from horario_escolar import HorarioSolver, contar_buracos
from schemas import HorarioInstance


def test_variaveis_cobrem_todos_os_tempos(s):
    i = s.instance
    assert len(s.x) == len(i.turmas) * len(i.disciplinas) * len(i.dias) * len(i.periodos)


def test_horario_valido(s, resolver, verificar):
    sol = resolver(s)
    assert sol.status in VIAVEL
    verificar(s.instance, sol.aulas)


def test_duplo_periodo_em_blocos(s, resolver):
    sol = resolver(s)
    for t in s.instance.turmas:
        tempos = sorted((d, p) for tt, c, d, p in sol.aulas if tt == t and c == "Educação Física")
        assert len(tempos) == 2 and tempos[0][0] == tempos[1][0]
        assert tempos[1][1] == tempos[0][1] + 1


def test_r1_turma_duas_aulas_em_simultaneo(s, resolver):
    x = s.x
    sol = resolver(s, x["7ºA", "Matemática", "Seg", 1], x["7ºA", "Português", "Seg", 1])
    assert sol.status == "INFEASIBLE"


def test_r2_carga_excedida(s, resolver):
    lits = [s.x["7ºA", "Matemática", d, 1] for d in s.instance.dias]  # 5 tempos, carga 4
    assert resolver(s, *lits).status == "INFEASIBLE"


def test_r3_mesma_disciplina_duas_vezes_no_dia(s, resolver):
    x = s.x
    sol = resolver(s, x["7ºA", "Matemática", "Seg", 1], x["7ºA", "Matemática", "Seg", 3])
    assert sol.status == "INFEASIBLE"


def test_r4_tempo_isolado_de_duplo_periodo(s, resolver):
    ef = "Educação Física"
    sol = resolver(s, s.x["7ºA", ef, "Seg", 5], ~s.x["7ºA", ef, "Seg", 4])
    assert sol.status == "INFEASIBLE"


def test_r4_bloco_duplo_nao_consecutivo(s, resolver):
    ef = "Educação Física"
    sol = resolver(s, s.x["7ºA", ef, "Seg", 1], s.x["7ºA", ef, "Seg", 5])
    assert sol.status == "INFEASIBLE"


@pytest.mark.parametrize(
    "a, b",
    [("História", "Inglês"), ("Matemática", "Matemática")],
    ids=["disciplinas-diferentes", "turmas-diferentes"],
)
def test_r5_professor_em_simultaneo(s, a, b, resolver):
    sol = resolver(s, s.x["7ºA", a, "Seg", 1], s.x["7ºB", b, "Seg", 1])
    assert sol.status == "INFEASIBLE"


@pytest.mark.parametrize("periodo", [1, 2, 3])
def test_r6_professor_indisponivel(s, periodo, resolver):
    sol = resolver(s, s.x["7ºA", "Educação Física", "Ter", periodo])
    assert sol.status == "INFEASIBLE"


def test_r6_professor_disponivel(s, resolver):
    sol = resolver(s, s.x["7ºA", "Educação Física", "Ter", 4])
    assert sol.status in VIAVEL


def test_r6_dados_v2(carregar_com, resolver, verificar):
    v2 = carregar_com(base="dados_v2")
    sol = resolver(v2)
    assert sol.status in VIAVEL
    verificar(v2.instance, sol.aulas)
    assert resolver(v2, v2.x["7ºA", "Matemática", "Sex", 5]).status == "INFEASIBLE"


@pytest.mark.parametrize(
    "normais, a, b",
    [(6, "Ciências", "Física"), (1, "Matemática", "Português")],
    ids=["especial", "normal"],
)
def test_r7_capacidade_de_salas_entre_turmas(normais, a, b, dados, carregar_com, resolver):
    """Duas turmas, professores diferentes, mesma sala com quantidade=1 → impossível."""
    base = (dados / "disciplinas.csv").read_text(encoding="utf8")
    s = carregar_com(
        salas=f"sala,tipo,quantidade\nSala Normal,normal,{normais}\n"
        "Laboratório,especial,1\nGinásio,especial,1\n",
        disciplinas=base + "Física,Prof. Filipe,1,nao,Laboratório\n",
    )
    sol = resolver(s, s.x["7ºA", a, "Seg", 4], s.x["7ºB", b, "Seg", 4])
    assert sol.status == "INFEASIBLE"


def test_r8_dados_lidos_dos_csv(s):
    assert s.instance.turmas == ["7ºA", "7ºB"]
    assert {d.disciplina: d.carga_semanal for d in s.instance.disciplinas}["Ciências"] == 3


def test_r8_outro_conjunto_de_dados(dados, carregar_com, resolver, verificar):
    s = carregar_com(
        turmas="turma\n7ºA\n7ºB\n8ºA\n",
        disciplinas=(dados / "disciplinas.csv").read_text(encoding="utf8")
        + "Música,Prof. Gil,2,sim,\n",
        disponibilidade_excecoes=(dados / "disponibilidade_excecoes.csv").read_text(encoding="utf8")
        + "Prof. Bruno,Qua,1\n",
    )
    assert len(s.x) == 3 * 7 * 25
    sol = resolver(s)
    assert sol.status in VIAVEL
    verificar(s.instance, sol.aulas)
    assert {t for t, *_ in sol.aulas} == {"7ºA", "7ºB", "8ºA"}


def test_csv_invalido_rejeitado(dados, carregar_com):
    with pytest.raises(ValueError, match="duplo_periodo"):
        carregar_com(
            disciplinas=(dados / "disciplinas.csv").read_text(encoding="utf8")
            + "Música,Prof. Gil,2,talvez,\n"
        )


def test_r9_incremental_muda_so_o_necessario(s, carregar_com, verificar):
    h0 = s.solve()
    s1 = carregar_com(base="dados_v2")
    s1.incremental(h0)
    h1 = s1.solve()
    assert h1.status in VIAVEL
    verificar(s1.instance, h1.aulas)
    # Só mudam as aulas de H0 que os novos dados proíbem (Prof. Ana, sexta, tempos 4-5)
    prof = {d.disciplina: d.professor for d in s1.instance.disciplinas}
    indisp = {(e.professor, e.dia, e.periodo) for e in s1.instance.excecoes}
    proibidas = {(t, c, d, p) for t, c, d, p in h0.aulas if (prof[c], d, p) in indisp}
    assert set(h0.aulas) - set(h1.aulas) == proibidas


@pytest.fixture(scope="module")
def o1():
    """Solver próprio: minimize_buracos altera o modelo, não pode partilhar o `s`."""
    so1 = HorarioSolver(HorarioInstance.from_csv(DADOS))
    so1.minimize_buracos()
    return so1


@pytest.mark.parametrize(
    "tempos, buracos",
    [([1, 3, 4], 1), ([2, 5], 2), ([1, 2, 3], 0), ([4], 0), ([], 0)],
    ids=["meio", "dois-seguidos", "sem-buracos", "uma-aula", "dia-vazio"],
)
def test_o1_contar_buracos(s, tempos, buracos):
    """Só contam tempos livres entre a 1.ª e a última aula (Prof. Ana dá Matemática)."""
    turmas = ["7ºA", "7ºB"]
    aulas = [(turmas[i % 2], "Matemática", "Seg", p) for i, p in enumerate(tempos)]
    assert contar_buracos(s.instance, aulas) == buracos


def test_o1_contar_buracos_por_professor_e_dia(s):
    # Ana (Mat) em Seg 1 e Seg 3 → 1 buraco; Bruno (Port) em Seg 2 não o tapa; Ter conta à parte
    aulas = [
        ("7ºA", "Matemática", "Seg", 1),
        ("7ºB", "Matemática", "Seg", 3),
        ("7ºA", "Português", "Seg", 2),
        ("7ºA", "Matemática", "Ter", 5),
    ]
    assert contar_buracos(s.instance, aulas) == 1


def test_o1_objetivo_bate_com_contagem(o1, resolver, verificar):
    sol = resolver(o1)
    assert sol.status in VIAVEL
    verificar(o1.instance, sol.aulas)
    assert o1.solver.objective_value == contar_buracos(o1.instance, sol.aulas)


def test_o1_reduz_buracos(o1, s, resolver):
    sem_o1 = resolver(s)
    com_o1 = resolver(o1)
    assert contar_buracos(o1.instance, com_o1.aulas) <= contar_buracos(s.instance, sem_o1.aulas)
    if com_o1.status == "OPTIMAL":
        assert contar_buracos(o1.instance, com_o1.aulas) == 0  # possível com os dados fornecidos


def test_o1_buraco_forcado_e_contado(o1, resolver):
    """Ana só pode dar 2 aulas à segunda (uma por turma): em Seg 1 e 3 há sempre 1 buraco."""
    x = o1.x
    sol = resolver(o1, x["7ºA", "Matemática", "Seg", 1], x["7ºB", "Matemática", "Seg", 3])
    assert sol.status in VIAVEL
    assert o1.solver.objective_value == contar_buracos(o1.instance, sol.aulas) >= 1
