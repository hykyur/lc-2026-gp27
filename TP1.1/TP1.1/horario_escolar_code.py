import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    from itertools import product
    from pathlib import Path

    import polars as pl
    from ortools.sat.python import cp_model
    from schemas import HorarioConfig, HorarioInstance, HorarioSolution

    DADOS = Path(__file__).parent / "dados"
    DADOS_V2 = Path(__file__).parent / "dados_v2"


@app.class_definition
class HorarioSolver:
    def __init__(self, instance: HorarioInstance, config: HorarioConfig | None = None):
        self.instance = instance
        self.config = config or HorarioConfig()
        self.model = cp_model.CpModel()
        self.x = {
            (t, d.disciplina, dia, p): self.model.new_bool_var(f"x_{t}_{d.disciplina}_{dia}_{p}")
            for t in instance.turmas
            for d in instance.disciplinas
            for dia in instance.dias
            for p in instance.periodos
        }
        self._build_model()
        self.solver = cp_model.CpSolver()

    def _slots(self):
        return product(self.instance.dias, self.instance.periodos)

    def _build_model(self):
        self._r1_uma_aula_por_tempo()
        self._r2_carga_semanal()
        self._r3_r4_uma_por_dia_e_duplo_periodo()
        self._r5_professor_sem_sobreposicao()
        self._r6_disponibilidade_professor()
        self._r7_capacidade_salas()

    def _r1_uma_aula_por_tempo(self):
        # R1 - Uma aula por turma em cada período
        for t in self.instance.turmas:
            for dia, p in self._slots():
                self.model.add_at_most_one(
                    self.x[t, d.disciplina, dia, p] for d in self.instance.disciplinas
                )

    def _r2_carga_semanal(self):
        # R2 - Carga semanal exata
        for t in self.instance.turmas:
            for d in self.instance.disciplinas:
                self.model.add(
                    sum(self.x[t, d.disciplina, dia, p] for dia, p in self._slots())
                    == d.carga_semanal
                )

    def _r3_r4_uma_por_dia_e_duplo_periodo(self):
        for t in self.instance.turmas:
            for d in self.instance.disciplinas:
                dupla = d.duplo_periodo == "sim"
                for dia in self.instance.dias:
                    # R3 - No máximo uma aula da mesma disciplina por dia, por turma;
                    # o bloco de 2 tempos de duplo período conta como uma só ocorrência.
                    self.model.add(
                        sum(self.x[t, d.disciplina, dia, p] for p in self.instance.periodos)
                        <= (2 if dupla else 1)
                    )
                    # R4 - Duplo período só em blocos de 2 tempos consecutivos no mesmo dia.
                    if dupla:
                        for p in self.instance.periodos:
                            vizinhos = (self.x.get((t, d.disciplina, dia, p + k)) for k in (-1, 1))
                            self.model.add(
                                self.x[t, d.disciplina, dia, p]
                                <= sum(v for v in vizinhos if v is not None)
                            )

    def _r5_professor_sem_sobreposicao(self):
        # R5 - Um professor não pode dar duas aulas em simultâneo
        for prof in {d.professor for d in self.instance.disciplinas}:
            for dia, p in self._slots():
                self.model.add_at_most_one(
                    self.x[t, d.disciplina, dia, p]
                    for t in self.instance.turmas
                    for d in self.instance.disciplinas
                    if d.professor == prof
                )

    def _r6_disponibilidade_professor(self):
        # R6 - Um professor só pode dar aulas nos tempos em que está disponível
        for e in self.instance.excecoes:
            for d in self.instance.disciplinas:
                if d.professor == e.professor:
                    for t in self.instance.turmas:
                        self.model.add(self.x[t, d.disciplina, e.dia, e.periodo] == 0)

    def _r7_capacidade_salas(self):
        # R7 - Nº de aulas em simultâneo numa sala não excede a quantidade dessa sala
        normal = next(s.sala for s in self.instance.salas if s.tipo == "normal")
        for s in self.instance.salas:
            disciplinas = [
                d.disciplina
                for d in self.instance.disciplinas
                if (d.sala_especial or normal) == s.sala
            ]
            for dia, p in self._slots():
                self.model.add(
                    sum(self.x[t, c, dia, p] for t in self.instance.turmas for c in disciplinas)
                    <= s.quantidade
                )

    def incremental(self, h0: HorarioSolution):
        # R9 - parte de H0 (hint) e minimiza o nº de aulas que mudam de tempo
        anteriores = set(h0.aulas)
        for k, v in self.x.items():
            self.model.add_hint(v, k in anteriores)
        self.model.minimize(sum(1 - self.x[k] for k in anteriores if k in self.x))
        self.solver.parameters.repair_hint = True

    def minimize_buracos(self):
        # O1 - Minimiza a quantidade de aulas vazias entre a primeira e a última de cada professor
        gaps = []
        for prof in {d.professor for d in self.instance.disciplinas}:
            owned = [d.disciplina for d in self.instance.disciplinas if d.professor == prof]
            for dia in self.instance.dias:
                busy = [
                    sum(
                        self.x[turma, dis, dia, p]
                        for turma in self.instance.turmas
                        for dis in owned
                    )
                    for p in self.instance.periodos
                ]
                busy_size = len(busy)
                started = [self.model.new_bool_var(f"s_{prof}_{dia}_{k}") for k in range(busy_size)]
                continues = [
                    self.model.new_bool_var(f"c_{prof}_{dia}_{k}") for k in range(busy_size)
                ]

                for k in range(busy_size):
                    self.model.add_max_equality(
                        started[k], [busy[k]] + ([started[k - 1]] if k > 0 else [])
                    )
                    self.model.add_max_equality(
                        continues[k], [busy[k]] + ([continues[k + 1]] if k < busy_size - 1 else [])
                    )
                    gap = self.model.new_bool_var(f"g_{prof}_{dia}_{k}")
                    self.model.add(gap >= started[k] + continues[k] - 1 - busy[k])
                    gaps.append(gap)

        self.model.minimize(sum(gaps))

    def solve(self, time_limit: float | None = None) -> HorarioSolution:
        self.solver.parameters.max_time_in_seconds = time_limit or self.config.time_limit
        self.solver.parameters.log_search_progress = self.config.log_search_progress
        status = self.solver.solve(self.model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return HorarioSolution(
                status=self.solver.status_name(status), tempo=self.solver.wall_time
            )
        return HorarioSolution(
            status=self.solver.status_name(status),
            tempo=self.solver.wall_time,
            aulas=[k for k, v in self.x.items() if self.solver.value(v)],
        )


@app.cell
def _():
    # H0 - horário principal: R1-R8 + O1 (minimizar buracos)
    s0 = HorarioSolver(HorarioInstance.from_csv(DADOS))
    s0.minimize_buracos()
    h0 = s0.solve()
    pl.DataFrame(h0.aulas, schema=["turma", "disciplina", "dia", "periodo"], orient="row")
    return h0, s0


@app.cell
def _(h0):
    # R9 - H1 do zero (sem H0) vs. incremental (hint de H0 + minimizar mudanças)

    def mudancas(h1):
        return len(set(h0.aulas) - set(h1.aulas))

    zero = HorarioSolver(HorarioInstance.from_csv(DADOS_V2)).solve()
    s1 = HorarioSolver(HorarioInstance.from_csv(DADOS_V2))
    s1.incremental(h0)
    h1 = s1.solve()
    pl.DataFrame(
        {
            "abordagem": ["do zero", "incremental"],
            "estado": [zero.status, h1.status],
            "tempo (s)": [zero.tempo, h1.tempo],
            "aulas alteradas": [mudancas(zero), mudancas(h1)],
        }
    )
    return (h1,)


@app.cell
def _(h1):
    pl.DataFrame(h1.aulas, schema=["turma", "disciplina", "dia", "periodo"], orient="row")
    return


@app.function
def contar_buracos(instance, aulas):
    prof = {d.disciplina: d.professor for d in instance.disciplinas}
    tempos = {}
    for _, c, dia, p in aulas:
        tempos.setdefault((prof[c], dia), []).append(p)
    return sum(max(ps) - min(ps) + 1 - len(ps) for ps in tempos.values())


@app.cell
def _(h0, s0):
    # O1 - buracos sem objetivo vs. com O1 (H0)
    sem_o1 = HorarioSolver(s0.instance).solve()
    pl.DataFrame(
        {
            "abordagem": ["sem O1", "com O1"],
            "estado": [sem_o1.status, h0.status],
            "tempo (s)": [sem_o1.tempo, h0.tempo],
            "buracos": [
                contar_buracos(s0.instance, sem_o1.aulas),
                contar_buracos(s0.instance, h0.aulas),
            ],
        }
    )
    return


if __name__ == "__main__":
    app.run()
