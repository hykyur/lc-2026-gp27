from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel, Field

from lc_2026_gp27.constants import DIAS, PERIODOS

# Aqui, utiliza-se pydantic para typechecking das variáveis de cada instância da classe,
# encapsulando os dados em classes


class Disciplina(BaseModel):
    disciplina: str
    professor: str
    carga_semanal: int = Field(ge=0)
    duplo_periodo: Literal["sim", "nao"]
    sala_especial: str = ""


class Sala(BaseModel):
    sala: str
    tipo: Literal["normal", "especial"]
    quantidade: int = Field(ge=0)


class Indisponibilidade(BaseModel):
    professor: str
    dia: str
    periodo: int


class HorarioInstance(BaseModel):
    turmas: list[str]
    disciplinas: list[Disciplina]
    salas: list[Sala]
    excecoes: list[Indisponibilidade]
    dias: list[str] = DIAS
    periodos: list[int] = PERIODOS

    @classmethod
    def from_csv(cls, pasta: Path) -> "HorarioInstance":
        def ler(nome):
            return pl.read_csv(
                pasta / f"{nome}.csv", empty_string_is_null=False, encoding="utf8-lossy"
            ).to_dicts()

        return cls(
            turmas=[r["turma"] for r in ler("turmas")],
            disciplinas=ler("disciplinas"),
            salas=ler("salas"),
            excecoes=ler("disponibilidade_excecoes"),
        )


class HorarioConfig(BaseModel):
    time_limit: float = 30.0
    log_search_progress: bool = False


class HorarioSolution(BaseModel):
    status: str
    tempo: float = 0.0  # wall time do solver, em segundos
    aulas: list[tuple[str, str, str, int]] = []  # (turma, disciplina, dia, periodo)
