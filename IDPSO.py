#-*- coding: utf-8 -*-
import random
from numpy import array
import numpy as np
import copy
import matplotlib.pyplot as plt
from numpy.random import uniform
import deap.benchmarks as dp
import time
import matplotlib.animation as anim

#limites
Xmax = 10
Xmin = -Xmax
posMax = 600
posMin = -600
mi = 100    

#variaveis auxiliares
contador = 0
fitness = 0
grafico = []
z = 0


class Particulas():
    pass

class IDPSO():
    def __init__(self, iteracoes, numero_particulas, inercia_inicial, inercia_final, c1, c2, crit_parada, min, max,
                 n_dimentions, function):
        '''
        Construtor para criar um objeto do tipo IDPSO
        :param iteracoes: inteiro, contendo a quantidade de iteracoes do enxame
        :param numero_particulas: inteiro, contendo a quantidade de particulas
        :param inercia_inicial: float, contendo a inercia que o enxame inicia
        :param inercia_final: float, contendo a inercia que o enxame termina
        :param c1: float, referente ao coeficiente pessoal
        :param c2: float, referente ao coeficiente coletivo
        :param crit_parada: inteiro, contendo a quantidade de particulas
        '''
        
        self.iteracoes = iteracoes
        self.numero_particulas = numero_particulas
        self.numero_dimensoes = n_dimentions
        self.inercia_inicial = inercia_inicial
        self.inercia_final = inercia_final
        self.c1_fixo = c1
        self.c2_fixo = c2
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
            p.velocidade = array([0.0 for i in range(self.numero_dimensoes)])
            p.best = p.dimensao
            p.fit_best = p.fitness
            p.c1 = self.c1_fixo
            p.c2 = self.c2_fixo
            p.inercia = self.inercia_inicial
            p.phi = 0
            self.particulas.append(p)
        
        self.gbest = self.particulas[0]
    
    def Funcao(self, posicao):
        '''
        metodo para computar o fitness de acordo com a posicao passada
        :param: posicao: vetor de float, contendo as posicoes das particulas
        :return: retorna o fitness da particula correspondente a posicao passada
        '''
        #return dp.kursawe(posicao)
        #return dp.sphere(posicao)
        #return dp.ackley(posicao)
        #return dp.bohachevsky(posicao)
        return self.function(posicao)
        #return dp.zdt6(posicao)
        
    def Fitness(self):
        '''
        metodo para computar o fitness de todas as particulas do enxame
        '''
        
        #tempo = time.time()
        
        for i in self.particulas: 
            i.fitness = self.Funcao(i.dimensao)
        
        #print("done in: ", time.time()-tempo)
        
    def Velocidade(self):
        '''
        metodo para computar a velocidade de todas as particulas do enxame
        '''
        
        calculo_c1 = 0
        calculo_c2 = 0
        
        # calculo da funcao de velocidade
        for i in self.particulas:
            for j in range(len(i.dimensao)):
                calculo_c1 = (i.best[j] - i.dimensao[j])
                calculo_c2 = (self.gbest.dimensao[j] - i.dimensao[j])
                
                influecia_inercia = (i.inercia * i.velocidade[j])
                influencia_cognitiva = ((i.c1 * random.random()) * calculo_c1)
                influecia_social = ((i.c2 * random.random()) * calculo_c2)
              
                i.velocidade[j] = influecia_inercia + influencia_cognitiva + influecia_social
                
                
                # condicoes para limitar a velocidade
                if (i.velocidade[j] >= Xmax):
                    i.velocidade[j] = Xmax
                elif(i.velocidade[j] <= Xmin):
                    i.velocidade[j] = Xmin
              
    def Atualizar_particulas(self):
        '''
        metodo para computar a nova posicao de cada particula
        '''
        
        # calculo para computar a nova posicao das particulas
        for i in self.particulas:
            for j in range(len(i.dimensao)):
                i.dimensao[j] = i.dimensao[j] + i.velocidade[j]
                
                
                # condicoes para limitar a posicao das particulas
                if (i.dimensao[j] >= self.max):
                    i.dimensao[j] = self.max
                elif(i.dimensao[j] <= self.min):
                    i.dimensao[j] = self.min

    def Atualizar_parametros(self, iteracao):
        '''
        metodo para atualizar os parametros c1, c2 e inercia de cada particula
        '''
        
        # computando a equacao para obter o valor de phi de cada particula
        for i in self.particulas:
            parte1 = 0
            parte2 = 0
            for j in range(len(i.dimensao)):
                parte1 = parte1 + self.gbest.dimensao[j] - i.dimensao[j]
                parte2 = parte2 + i.best[j] - i.dimensao[j]
                if(parte1 == 0):
                    parte1 = 1
                if(parte2 == 0):
                    parte2 = 1
            i.phi = abs(parte1/parte2)
            
            
        # computando as equacoes para mudar os coeficientes de cada particula
        for i in self.particulas:
            ln = np.log(i.phi)
            calculo = i.phi * (iteracao - ((1 + ln) * self.iteracoes) / mi)
            i.inercia = ((self.inercia_inicial - self.inercia_final) / (1 + np.exp(calculo))) + self.inercia_final
            i.c1 = self.c1_fixo * (i.phi ** (-1))
            i.c2 = self.c2_fixo * i.phi
       
    def Pbest(self):
        '''
        Metodo para atualizar a melhor posicao atual de cada particula
        '''
        
        for i in self.particulas:
            if(i.fit_best >= i.fitness):
                i.best = i.dimensao
                i.fit_best = i.fitness

    def Gbest(self):
        '''
        Metodo para atualizar a melhor particula do enxame
        '''
        
        for i in self.particulas:
            if(i.fitness <= self.gbest.fitness):
                self.gbest = copy.deepcopy(i)
    
    def Criterio_parada(self, i):
        '''
        Metodo para parar o treinamento caso a melhor solucao nao mude
        :param: i: iteracao atual
        '''
        
        global contador, fitness, resposta
        resposta = False
        
        #print(contador)
        if(contador == self.crit_parada):
            print("        -Iteracao: %d" % (i) + " - Melhor Fitness ", self.gbest.fitness)
            resposta = True
            return self.iteracoes, resposta

        if(i == 0):
            fitness = copy.deepcopy(self.gbest.fitness)
            return i, resposta
            
        if(fitness == self.gbest.fitness):
            contador+=1
            return i, resposta
        
        else:
            fitness = copy.deepcopy(self.gbest.fitness)
            contador = 0
            return i, resposta
    
    def Grafico_Convergencia(self, fitness, i, function_name, time):
        '''
        Metodo para plotar o grafico de convergencia apos a busca
        :param: fitness: float, com o fitness da melhor particula no tempo i
        :param: i: iteracao atual
        :param: function_name: nome da função que está sendo executada
        :param: time: tempo de execução do algoritmo
        '''

        global grafico

        grafico.append(fitness)

        if (i == self.iteracoes):
            plt.plot(grafico)
            plt.savefig('images/IDPSO-{0}-{1}'.format(function_name, time))
    
    def Executar(self, function_name, execution):
        '''
        metodo para executar o procedimento do IDPSO
        '''
        
        #variavel para computar o tempo de inicio 
        tempo = time.time()
        
        # criando as particulas
        self.Criar_Particula()       
        
        # movimentando o enxame
        i = 0
        while(i < self.iteracoes):
            i = i + 1
            
            self.Fitness()
            self.Gbest()
            self.Pbest()
            self.Velocidade()
            self.Atualizar_parametros(i)
            self.Atualizar_particulas()
            
            #print("Iteracao: %d" % (i) + " - Melhor Fitness ", self.gbest.fitness)

            i, resp = self.Criterio_parada(i)
            #self.Grafico_Convergencia(self.gbest.fitness, i, function_name, execution)

            execution_time = time.time() - tempo
            if resp:
                break
            # print("done in: ", time.time()-tempo)
        global contador
        contador = 0
        return self.gbest.fitness[0], i, execution_time
    
    def Executar_animacao(self):
        global z
        
        # movimentando o enxame
        z = z + 1
           
        self.Fitness()
        self.Gbest()
        self.Pbest()
        self.Velocidade()
        self.Atualizar_parametros(z)
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
                if(self.particulas[j].fitness[0] == self.gbest.fitness[0]):
                    best = j
            colors[best] = "red"
            
            movimento.scatter(particulas_x, particulas_y, color = colors)
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

    enxame = IDPSO(100, 20, 0.8, 0.4, 2.07, 2.07, 20, posMin, posMax, 20, dp.griewank)
    enxame.Animacao(50)
    #enxame.Executar()
            


if __name__ == "__main__":
    main()
    


