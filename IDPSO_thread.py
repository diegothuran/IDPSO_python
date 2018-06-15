#-*- coding: utf-8 -*-
import random
from numpy import array
import numpy as np
import copy
import matplotlib.pyplot as plt
from numpy.random.mtrand import uniform
import deap.benchmarks as dp
import time
import threading 
import queue
import multiprocessing as mp

#limites
Xmax = 2
Xmin = -Xmax
posMax = 100
posMin = 0
mi = 100

#variaveis auxiliares
contador = 0
fitness = 0
grafico = []

class Particulas():
    pass

class IDPSO():
    def __init__(self, iteracoes, numero_particulas, inercia_inicial, inercia_final, c1, c2, crit_parada):
        self.iteracoes = iteracoes
        self.numero_particulas = numero_particulas
        self.numero_dimensoes = 2
        self.inercia_inicial = inercia_inicial
        self.inercia_final = inercia_final
        self.c1_fixo = c1
        self.c2_fixo = c2
        self.crit_parada = crit_parada
        self.particulas = []
        self.gbest = []
        
    def Criar_Particula(self):
        for i in range(self.numero_particulas):
            p = Particulas()
            p.dimensao = array([uniform(posMin,posMax) for i in range(self.numero_dimensoes)])
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
        return dp.sphere(posicao)
   
    def Funcao_thread(self, posicao, out_queue):
        return out_queue.put(dp.sphere(posicao))
    
    def Fitness(self):
        for i in self.particulas:   
            i.fitness = self.Funcao(i.dimensao)
    
    def Fitness_thread(self):
        
        tarefas = []
        saida_tarefas = []
        
        #tempo = time.time()
         
        for x, i in enumerate(self.particulas):
            
            saida = queue.Queue()
            
            tarefa = threading.Thread(target=self.Funcao_thread, args=(i.dimensao, saida))
            tarefa.start()
               
            tarefas.append(tarefa)
            saida_tarefas.append(saida)
            
        for i in tarefas:
            i.join()
        
        for x, i in enumerate(self.particulas):    
            i.fitness = saida_tarefas[x].get()
        
        #print("done in: ", time.time()-tempo)   
    
    def Fitness_processo(self):
        for i in self.particulas:  
            p = mp.Pool(mp.cpu_count())
            i.fitness = p.map(self.Funcao, [i.dimensao]) 
            i.fitness = i.fitness[0][0]
            
    def Velocidade(self):
        calculo_c1 = 0
        calculo_c2 = 0
        
        for i in self.particulas:
            for j in range(len(i.dimensao)):
                calculo_c1 = (i.best[j] - i.dimensao[j])
                calculo_c2 = (self.gbest.dimensao[j] - i.dimensao[j])
                
                influecia_inercia = (i.inercia * i.velocidade[j])
                influencia_cognitiva = ((i.c1 * random.random()) * calculo_c1)
                influecia_social = ((i.c2 * random.random()) * calculo_c2)
              
                i.velocidade[j] = influecia_inercia + influencia_cognitiva + influecia_social
                
                if (i.velocidade[j] >= Xmax):
                    i.velocidade[j] = Xmax
                elif(i.velocidade[j] <= Xmin):
                    i.velocidade[j] = Xmin
              
    def Atualizar_particulas(self):
        for i in self.particulas:
            for j in range(len(i.dimensao)):
                i.dimensao[j] = i.dimensao[j] + i.velocidade[j]
                
                if (i.dimensao[j] >= posMax):
                    i.dimensao[j] = posMax
                elif(i.dimensao[j] <= posMin):
                    i.dimensao[j] = posMin

    def Atualizar_parametros(self, iteracao):
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
            
        for i in self.particulas:
            ln = np.log(i.phi)
            calculo = i.phi * (iteracao - ((1 + ln) * self.iteracoes) / mi)
            i.inercia = ((self.inercia_inicial - self.inercia_final) / (1 + np.exp(calculo))) + self.inercia_final
            i.c1 = self.c1_fixo * (i.phi ** (-1))
            i.c2 = self.c2_fixo * i.phi
       
    def Pbest(self):
        for i in self.particulas:
            if(i.fit_best >= i.fitness):
                i.best = i.dimensao
                i.fit_best = i.fitness

    def Gbest(self):
        for i in self.particulas:
            if(i.fitness <= self.gbest.fitness):
                self.gbest = copy.deepcopy(i)
    
    def Criterio_parada(self, i):
        global contador, fitness
        
        #print(contador)
        if(contador == self.crit_parada):
            print("        -Iteracao: %d" % (i) + " - Melhor Fitness ", self.gbest.fitness)
            return self.iteracoes
        
        if(i == 0):
            fitness = copy.deepcopy(self.gbest.fitness)
            return i
            
        if(fitness == self.gbest.fitness):
            contador+=1
            return i
        
        else:
            fitness = copy.deepcopy(self.gbest.fitness)
            contador = 0
            return i
    
    def Grafico_Convergencia(self, fitness, i):
        global grafico
        
        grafico.append(fitness)
        
        if(i == self.iteracoes):
            plt.plot(grafico)
            plt.show()
    
    def Executar(self):
        
        tempo = time.time()
        
        self.Criar_Particula()       
        
        i = 0
        while(i < self.iteracoes):
            i = i + 1
            
            self.Fitness_thread()
            self.Gbest()
            self.Pbest()
            self.Velocidade()
            self.Atualizar_parametros(i)
            self.Atualizar_particulas()
            
            print("        -Iteracao: %d" % (i) + " - Melhor Fitness ", self.gbest.fitness)
            
            #i = self.Criterio_parada(i)
            #self.Grafico_Convergencia(self.gbest.fitness, i)
        
        print("done in: ", time.time()-tempo)
             
def main():
    enxame = IDPSO(1000, 30, 0.5, 0.3, 2.4, 1.3, 20)
    enxame.Executar()        


if __name__ == "__main__":
    main()
    


