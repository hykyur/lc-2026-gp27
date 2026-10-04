import marimo
__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import marimo as mo
    import polars as pl
    from polars import DataFrame
    from ortools.sat.python import cp_model
    from lc_2026_gp27.constants import PERIODOS, DIAS

    def parse_csv(source) -> DataFrame:
        return pl.read_csv(
            source=source,
            empty_string_is_null=False,
            encoding='utf8-lossy'
        )

    DADOS = Path(__file__).parent / "dados"

    horario = cp_model.CpModel()

    D = parse_csv(source=DADOS / "disciplinas.csv")
    E = parse_csv(source=DADOS / "disponibilidade_excecoes.csv")
    S = parse_csv(source=DADOS / "salas.csv")
    T = parse_csv(source=DADOS / "turmas.csv")

    x = {
        (
            turma,
            disciplina,
            dia,
            periodo
        ):
            horario.new_bool_var(
                f"x_{turma}_{disciplina}_{dia}_{periodo}"
            )
        for turma in T["turma"]
        for disciplina in D["disciplina"]
        for dia in DIAS
        for periodo in PERIODOS
    }


if __name__ == "__main__":
    app.run()
