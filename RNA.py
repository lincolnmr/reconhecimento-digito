# importação das bibliotecas necessárias

# pybrain
from pybrain.datasets.supervised import SupervisedDataSet 
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer


# gráficos 
import matplotlib.pyplot as plt
import numpy as np

# função para carregar os dados de treinamento
def getData( path ):
    #Open file
    file = open( path, "r" )
    
    data = []    
    
    for linha in file:        # obtem cada linha do arquivo
      linha = linha.rstrip()  # remove caracteres de controle, \n
      digitos = linha.split(" ")  # pega os dígitos
      for numero in digitos:   # para cada número da linha
        data.append( numero )  # add ao vetor de dados  
    
    file.close()
    return data


# configurando a rede neural artificial e o dataSet de treinamento
network = buildNetwork( 45, 500, 500, 1 )    # define network 
dataSet = SupervisedDataSet( 45, 1 )  # define dataSet


arquivos = [    '0a-test.txt', '0b-test.txt', '0c-test.txt',
                '1a-test.txt', '1b-test.txt', '1c-test.txt',
                '2a-test.txt', '2b-test.txt', '2c-test.txt',
                '3a-test.txt', '3b-test.txt', '3c-test.txt',
                '4a-test.txt', '4b-test.txt', '4c-test.txt',
                '5a-test.txt', '5b-test.txt', '5c-test.txt',
                '6a-test.txt', '6b-test.txt', '6c-test.txt',
                '7a-test.txt', '7b-test.txt', '7c-test.txt',
                '8a-test.txt', '8b-test.txt', '8c-test.txt',
                '9a-test.txt', '9b-test.txt', '9c-test.txt']          
# a resposta do número
resposta = [[0],[0],[0], [1],[1],[1], [2],[2],[2], [3],[3],[3], [4],[4],[4], [5],[5],[5], [6],[6],[6], [7],[7],[7], [8],[8],[8], [9], [9], [9]] 

i = 0
for arquivo in arquivos:           # para cada arquivo de treinamento
    data =  getData( arquivo )            # pegue os dados do arquivo
    dataSet.addSample( data, resposta[i] )  # add dados no dataSet
    i = i + 1


# trainer
trainer = BackpropTrainer( network, dataSet )
error = 1
iteration = 0
outputs = []
file = open("outputs.txt", "w") #arquivo para guardar os resultados

while error > 0.001: # 10 ^ -3
    error = trainer.train()
    outputs.append( error )
    iteration += 1    
    print ( iteration, error )
    file.write( str(error)+"\n" )

file.close()

# Fase de teste
arquivos = ['0.txt','1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt', '7.txt', '8.txt', '9.txt']
for arquivo in arquivos:
    data =  getData( arquivo )
    print ( network.activate( data ) )


# plot graph
plt.ioff()
plt.plot( outputs )
plt.xlabel('Iterações')
plt.ylabel('Erro Quadrático')
plt.show()

