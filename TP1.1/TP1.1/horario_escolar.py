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


# R1 - Uma aula por turma em cada período
for turma in T["turma"]:
    for dia in DIAS:
        for periodo in PERIODOS:
            horario.add(sum(x[(turma, disciplina, dia, periodo)]for disciplina in D["disciplina"]) <= 1)

# R2 - Carga semanal exata
for turma in T["turma"]:
    for linha in D.iter_rows(named=True):
        disciplina = linha["disciplina"]
        carga = linha["carga_semanal"]
        horario.add(sum(x[(turma, disciplina, dia, periodo)]for dia in DIAS for periodo in PERIODOS) == carga)


for turma in T["turma"]:
    for linha in D.iter_rows(named=True):
        disciplina = linha["disciplina"]
        dupla = linha["duplo_periodo"] == "sim"
        for dia in DIAS:
            # R3 - No máximo uma aula da mesma disciplina por dia, por turma — exceto disciplinas de duplo período (ver R4), em que o bloco de 2 tempos conta como uma só ocorrência nesse dia.
            limite = 2 if dupla else 1
            horario.add(sum(x[(turma, disciplina, dia, periodo)]for periodo in PERIODOS) <= limite)
            # R4 - Disciplinas marcadas duplo_periodo=sim só podem ser dadas em blocos de 2 tempos consecutivos, no mesmo dia (nunca um tempo isolado).
            if dupla:
                for periodo in PERIODOS:
                    vizinhos = []
                    if periodo > 1:
                        vizinhos.append(x[(turma, disciplina, dia, periodo - 1)])

                    if periodo < 5:
                        vizinhos.append(x[(turma, disciplina, dia, periodo + 1)])

                    horario.add(x[(turma, disciplina, dia, periodo)]<= sum(vizinhos))


# Requesito 5 Um professor não pode dar duas aulas em simultâneo,
#  mesmo que sejam a turmas ou disciplinas diferentes

for professor in D["professor"].unique(): # .unique evita repetir professores
    for dia in DIAS:
        for periodo in PERIODOS:
            horario.add(sum(
                x[(turma, linha["disciplina"], dia, periodo)]
                    for turma in T["turma"]
                    for linha in D.iter_rows(named=True)
                    if linha["professor"] == professor ) <=1 )


# Um professor só pode dar aulas nos tempos em que está disponível

for excecao in E.iter_rows(named=True):
    professor = excecao["professor"]
    dia = excecao["dia"]
    periodo = excecao["periodo"]

    for linha in D.iter_rows(named=True):
        if linha["professor"] == professor:
            disciplina = linha["disciplina"]

            for turma in T["turma"]:
                horario.add(
                    x[(turma, disciplina, dia, periodo)] == 0 # Igual a 0 pois nao pode acontecer naquele instante
                )




    


if __name__ == "__main__":
    app.run()

