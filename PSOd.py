# -*- coding: utf-8 -*-
import random
from numpy import array
import numpy as np
import copy
import matplotlib.pyplot as plt
from numpy.random import uniform
import deap.benchmarks as dp
import time
import matplotlib.animation as anim
import math

# limites
Xmax = 10
Xmin = -Xmax
posMax = 600
posMin = -600
mi = 100

# variaveis auxiliares
contador = 0
fitness = 0
grafico = []
z = 0


class Particulas():
    pass


class PSOd():
    def __init__(self, iteracoes, numero_particulas, crit_parada, min, max, n_dimentions ,function):
        '''
        Construtor para criar um objeto do tipo IDPSO
        :param iteracoes: inteiro, contendo a quantidade de iteracoes do enxame
        :param numero_particulas: inteiro, contendo a quantidade de particulas
        :param inercia_inicial: float, contendo a inercia que o enxame inicial
        :param crit_parada: inteiro, contendo a quantidade de particulas
        '''

        self.iteracoes = iteracoes
        self.numero_particulas = numero_particulas
        self.numero_dimensoes = n_dimentions
        self.crit_parada = crit_parada
        self.particulas = []
        self.gbest = []
        self.min = min
        self.max = max
        self.function = function

    def Criar_Particula(self):
        '''
        Metodo para criar e inicializar todos os compenentes das particulas do enxame
        '''

        for i in range(self.numero_particulas):
            p = Particulas()
            p.dimensao = array([uniform(self.min, self.max) for i in range(self.numero_dimensoes)])
            p.fitness = self.Funcao(p.dimensao)
            p.best = p.dimensao
            p.fit_best = p.fitness
            self.particulas.append(p)

        self.gbest = self.particulas[0]

    def Funcao(self, posicao):
        '''
        metodo para computar o fitness de acordo com a posicao passada
        :param: posicao: vetor de float, contendo as posicoes das particulas
        :return: retorna o fitness da particula correspondente a posicao passada
        '''
        # return dp.kursawe(posicao)
        # return dp.sphere(posicao)
        # return dp.ackley(posicao)
        return self.function(posicao)
        #return dp.griewank(posicao)
        #return dp.kursawe(posicao)

    def Fitness(self):
        '''
        metodo para computar o fitness de todas as particulas do enxame
        '''

        # tempo = time.time()

        for i in self.particulas:
            i.fitness = self.Funcao(i.dimensao)

        # print("done in: ", time.time()-tempo)

    def Atualizar_particulas(self):
        '''
        metodo para computar a nova posicao de cada particula
        '''

        # calculo para computar a nova posicao das particulas

        for i in self.particulas:
            for j in range(len(i.dimensao)):

                mean = (self.gbest.dimensao[j] + i.best[j] + i.dimensao[j])/3
                std = math.sqrt(1/3 * (i.dimensao[j] - mean)**2 +(i.best[j] - mean)**2 + (self.gbest.dimensao[j] - mean)**2)

                k1 = random.random()
                k2 = random.random()
                z = (-2*math.log(k1))**0.5 * math.cos(2*math.pi*k2)

                i.dimensao[j] = mean + std*z

                # condicoes para limitar a posicao das particulas
                if (i.dimensao[j] >= posMax):
                    i.dimensao[j] = posMax
                elif (i.dimensao[j] <= posMin):
                    i.dimensao[j] = posMin

    def Pbest(self):
        '''
        Metodo para atualizar a melhor posicao atual de cada particula
        '''

        for i in self.particulas:
            if (i.fit_best >= i.fitness):
                i.best = i.dimensao
                i.fit_best = i.fitness

    def Gbest(self):
        '''
        Metodo para atualizar a melhor particula do enxame
        '''

        for i in self.particulas:
            if (i.fitness <= self.gbest.fitness):
                self.gbest = copy.deepcopy(i)

    def Criterio_parada(self, i):
        '''
        Metodo para parar o treinamento caso a melhor solucao nao mude
        :param: i: iteracao atual
        '''

        global contador, fitness

        # print(contador)
        if (contador == self.crit_parada):
            print("        -Iteracao: %d" % (i) + " - Melhor Fitness ", self.gbest.fitness)
            return self.iteracoes

        if (i == 0):
            fitness = copy.deepcopy(self.gbest.fitness)
            return i

        if (fitness == self.gbest.fitness):
            contador += 1
            return i

        else:
            fitness = copy.deepcopy(self.gbest.fitness)
            contador = 0
            return i

    def Grafico_Convergencia(self, fitness, i, function_name, execution):
        '''
        Metodo para plotar o grafico de convergencia apos a busca
        :param: fitness: float, com o fitness da melhor particula no tempo i
        :param: i: iteracao atual
        '''

        global grafico

        grafico.append(fitness)

        if (i == self.iteracoes):
            plt.plot(grafico)
            plt.savefig('images/PSO-{0}-{1}'.format(function_name, execution))

    def Executar(self, function_name, execution):
        '''
        metodo para executar o procedimento do IDPSO
        '''

        # variavel para computar o tempo de inicio
        tempo = time.time()

        # criando as particulas
        self.Criar_Particula()

        # movimentando o enxame
        i = 0
        while (i < self.iteracoes):
            i = i + 1

            self.Fitness()
            self.Gbest()
            self.Pbest()
            self.Atualizar_particulas()

            #print("Iteracao: %d" % (i) + " - Melhor Fitness ", self.gbest.fitness)

            i = self.Criterio_parada(i)
            #self.Grafico_Convergencia(self.gbest.fitness, i, function_name, execution)

            execution_time = time.time() - tempo
            # print("done in: ", time.time()-tempo)
        global contador
        contador = 0
        return self.gbest.fitness[0], self.iteracoes, execution_time

    def Executar_animacao(self):
        global z

        # movimentando o enxame
        z = z + 1

        self.Fitness()
        self.Gbest()
        self.Pbest()
        self.Atualizar_particulas()

        return self.gbest.fitness[0]

    def Animacao(self, it):
        '''
        metodo para chamar a animacao
        :param: fun: e uma funcao que retorna um numero inteiro
        :param: it: e a quantidade de vezes que a funcao fun sera chamada e que os frames serao gerados
        '''

        # criando uma lista para armazenar os valores gerados
        fitnesss = []

        # criando uma figura para conter os graficos
        fig = plt.figure()
        # plotando  o grafico em um lugar especifico
        convergencia = fig.add_subplot(1, 2, 1)
        # plotando  o grafico em um lugar especifico
        movimento = fig.add_subplot(1, 2, 2)

        # criando as particulas
        self.Criar_Particula()

        def update(i):
            '''
            metodo para chamar atualizar o frame a cada instante de tempo
            '''

            ################################ atualiando o frame de fitness ########################
            # recebendo o fitness para a iteracao atual
            yi = self.Executar_animacao()
            # salvando o novo dado no vetor fitnesss
            fitnesss.append(yi)
            # coletando a quantidade de dados
            x_fitness = range(len(fitnesss))
            # apagando o frame antigo
            convergencia.clear()
            # atualizando o novo frame
            convergencia.plot(x_fitness, fitnesss)
            # printando a iteracao atual
            convergencia.set_title("Convergence Graph")
            convergencia.set_ylabel('Fitness')
            convergencia.set_xlabel('Iteration')
            print(i, ': ', yi)
            #########################################################################################

            ################################ atualiando o frame de movimento ########################
            particulas_x = [z.dimensao[0] for z in self.particulas]
            particulas_y = [z.dimensao[1] for z in self.particulas]
            movimento.clear()

            colors = ["blue"] * len(particulas_x)
            best = 0
            for j in range(len(self.particulas)):
                if (self.particulas[j].fitness[0] == self.gbest.fitness[0]):
                    best = j
            colors[best] = "red"

            movimento.scatter(particulas_x, particulas_y, color=colors)
            movimento.set_title("Particles moviment")
            movimento.set_ylabel('y')
            movimento.set_xlabel('x')
            ax = 150
            movimento.axis([ax, -ax, ax, -ax])
            #########################################################################################

        # funcao que atualiza a animacao
        plt.legend()
        a = anim.FuncAnimation(fig, update, frames=it, repeat=False)
        plt.show()


def main():
    enxame = PSOd(1000, 50, 100, Xmin, Xmax, 2, dp.griewank)
    enxame.Animacao(100)
    #enxame.Executar()


if __name__ == "__main__":
    main()



