"""Fixtures e funções auxiliares partilhadas pelos testes do horário escolar."""

import shutil
from pathlib import Path

import pytest
from horario_escolar import HorarioSolver
from schemas import HorarioInstance

DADOS = Path(__file__).parents[1] / "TP1.1" / "TP1.1" / "dados"
VIAVEL = ("OPTIMAL", "FEASIBLE")


def _resolver(s, *literais):
    s.model.clear_assumptions()
    s.model.add_assumptions(literais)
    sol = s.solve()
    s.model.clear_assumptions()
    return sol


def _verificar(inst, aulas):
    """Verificador independente de R1-R7 sobre um horário {(turma, disc, dia, periodo)}."""
    sol = set(aulas)
    disc = {d.disciplina: d for d in inst.disciplinas}
    indisp = {(e.professor, e.dia, e.periodo) for e in inst.excecoes}
    normal = next(s.sala for s in inst.salas if s.tipo == "normal")
    qtd = {s.sala: s.quantidade for s in inst.salas}
    slots = [(d, p) for d in inst.dias for p in inst.periodos]

    for t in inst.turmas:
        for d, p in slots:
            assert sum((t, c, d, p) in sol for c in disc) <= 1, f"R1 {t} {d}{p}"
        for c, r in disc.items():
            tempos = [(d, p) for d, p in slots if (t, c, d, p) in sol]
            assert len(tempos) == r.carga_semanal, f"R2 {t} {c}"
            for d in inst.dias:
                ps = sorted(p for dd, p in tempos if dd == d)
                if r.duplo_periodo == "sim":
                    assert not ps or ps == [ps[0], ps[0] + 1], f"R3/R4 {t} {c} {d} {ps}"
                else:
                    assert len(ps) <= 1, f"R3 {t} {c} {d}"
    for d, p in slots:
        aulas = [(t, c) for t, c, dd, pp in sol if (dd, pp) == (d, p)]
        profs = [disc[c].professor for _, c in aulas]
        assert len(profs) == len(set(profs)), f"R5 {d}{p}"
        assert not any((pr, d, p) in indisp for pr in profs), f"R6 {d}{p}"
        for sala, q in qtd.items():
            uso = sum((disc[c].sala_especial or normal) == sala for _, c in aulas)
            assert uso <= q, f"R7 {sala} {d}{p}"


@pytest.fixture(scope="module")
def s():
    return HorarioSolver(HorarioInstance.from_csv(DADOS))


@pytest.fixture
def dados():
    return DADOS


@pytest.fixture
def carregar_com(tmp_path):
    """Solver com os CSVs de `base`, substituindo os dados em `csvs`."""

    def carregar(base="dados", **csvs):
        shutil.copytree(DADOS.parent / base, tmp_path / "dados")
        for nome, conteudo in csvs.items():
            (tmp_path / "dados" / f"{nome}.csv").write_text(conteudo, encoding="utf8")
        return HorarioSolver(HorarioInstance.from_csv(tmp_path / "dados"))

    return carregar


@pytest.fixture
def resolver():
    return _resolver


@pytest.fixture
def verificar():
    return _verificar
