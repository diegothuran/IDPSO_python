from PSOd import PSOd
from IDPSO import IDPSO
import deap.benchmarks as dp
from PSO import PSO
import pandas as pd
import numpy as np

functions = {'griwank': dp.griewank, 'sphere': dp.sphere,
             'rosenbrock': dp.rosenbrock, 'bohachevysck': dp.bohachevsky}
n_dimentions = [50]
n_iteracoes = 20000
stop_criteria = 2000
n_executions = 1
c1 = c2 = 2
w_inital = 0.8
w_final = 0.4

limites = {'griwank': (-600, 600), 'sphere': (-5.12, 5.12),
           'rosenbrock': (-5, 10), 'bohachevysck': [-100, 100]}

type_pso = ['pso', 'idpso', 'psod']

def initiate_pso(pso, limites, j,function):
    if pso == 'pso':
        return PSO(n_iteracoes, 50, w_inital, c1, c2, stop_criteria, limites[0],
                      limites[1], j, function)
    if pso == 'idpso':
        return IDPSO(n_iteracoes, 50, w_inital, w_final, c1, c2, stop_criteria, limites[0],
                      limites[1], j, function)
    if pso == 'psod':
        return PSOd(n_iteracoes, 50, stop_criteria, limites[0],
                      limites[1], j, function)

    def Criterio_parada(self, i):
        '''
        Metodo para parar o treinamento caso a melhor solucao nao mude
        :param: i: iteracao atual
        '''

        global contador, fitness, resposta
        resposta = False

        # print(contador)
        if (contador == self.crit_parada):
            print("        -Iteracao: %d" % (i) + " - Melhor Fitness ", self.gbest.fitness)
            resposta = True
            return self.iteracoes, resposta

        if (i == 0):
            fitness = copy.deepcopy(self.gbest.fitness)
            return i, resposta

        if (fitness == self.gbest.fitness):
            contador += 1
            return i, resposta

        else:
            fitness = copy.deepcopy(self.gbest.fitness)
            contador = 0
            return i, resposta
writer = pd.ExcelWriter('resultados50.xlsx')
for function in functions.keys():
    results = {'idpso': [], 'psod': [], 'pso': []}
    for i in range(n_executions):
        for j in n_dimentions:
            for swarm in type_pso:
                results[swarm].append(list(initiate_pso(swarm, limites[function], j, functions[function]).Executar(function, i)))

    results = pd.DataFrame.from_dict(results)
    results = pd.concat({
        k: pd.DataFrame(v.tolist())
        for k, v in results.items()
        }, axis=1)

    results.to_excel(writer, 'resultados-{0}'.format(function))
    writer.save()
