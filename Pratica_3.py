#!/usr/bin/env python
# coding: utf-8

# # Relatório Prático Nº. 3
# # Redes Neurais Artificiais
# **Aluno: Fernando Elias de Melo Borges - 2020260050** <br />
# **Curso: Engenharia de Sistemas e Automação** <br />
# **Título: Plota dados e reta de duas dimensões** <br />
# Data: 07/12/2020

# ## Objetivo:
# Esta experiência tem por objetivo principal a implementação dos dados de cada classe e da reta de fronteira de decisão para conjuntos de dados bidimensionais, de forma a observar os dados utilizados e a tomada de decisão pelo modelo.

# ##  Script e Resultados:
# **1 - importando as bibliotecas a serem utilizadas:** <br />
# numpy para cálculos algébricos e matplotlib para geração de gráficos

# In[1]:


import numpy as np
import matplotlib.pyplot as plt 


# **2 - Definindo a função de treinamento do perceptron:** <br />
# A função de treinamento do perceptron consiste no ajuste automático dos pesos e bias do neurônio de acordo com o erro de classificação. <br />
# O treinamento usa como medida de custo a soma do erro quadrático (SEQ) e os critérios de parada são o custo ser menor que a medida de tolerância ou o número de épocas atingir o máximo especificado pelo usuário.

# In[2]:


def treina_perceptron(W, b, X, yd, alfa, max_epocas, tolerancia):
    N = X.shape[1]
    vetor_seq = []
    seq = tolerancia
    epoca = 1
    seq = 100
    while(epoca <= max_epocas and seq >= tolerancia):
        seq = 0
        for i in range(0, N):
            y,u = perceptron(X[:,i], W, b)
            erro = yd[i] - y
            W = W+(alfa*erro*X[:,i])
            b = b+alfa*erro
            seq = seq + (erro**2)  
        epoca = epoca+1
        vetor_seq.append(seq)

    vetor_seq = np.array(vetor_seq)
    return W, b, vetor_seq


# **3 - Definindo a função perceptron utilizada:**<br />
# Como se trata de um modelo de uma camada e 2 variáveis de entrada, pode-se definir as funções da rede neural como:<br />
# $u = w_1x_1 + w_2x_2 + b$ (1)  &nbsp;&nbsp; e &nbsp;&nbsp;  $ \left\{\begin{matrix} y = 0, & se & u < 0 \\  \\ y= 1, & se & u \geq 0 \end{matrix}\right. $ (2) 
# 
# Tal função pode ser escrita de forma matricial, utilizada nesta função:
# $ y = f(wX+b) $, sendo a função $f$ dada pela equação (2).

# In[3]:


def perceptron(X, w, b):
    u = np.dot(w,X)+b
    if len(u.shape) == 1:
        u = u[:, np.newaxis]
    y = -np.ones(u.shape[1])

    for i in range(u.shape[1]):
        if u[0,i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
    return y, u


# **4 - Função de geração de gráficos de evolução do erro:**<br />
# Esta função plota o gráfico do erro em relação às epocas de treinamento de maneira a ilustrar a evolução do erro e a convergência do algoritmo.

# In[4]:


def plot_seq(vetor_seq):
    hor = np.arange(1, vetor_seq.shape[0]+1)
    hor = hor.astype(int)
    plt.plot(hor,vetor_seq,'.-')
    plt.ylabel('SEQ', fontsize = 14)
    plt.xlabel('epocas', fontsize = 14)
    plt.title('Grafico de evolução do erro', fontsize = 14)


# **5 - Função de geração de gráfico do dados da classe:**<br />
# Esta função plota os dados em 2 dimensões no espaço de acordo com cada classe estipulada. Esta função permite a visualização de até 10 classes, conforme estipulado na experiência

# In[5]:


def plotadc2d(X, yd):
    if ((yd.shape[1] == 1) or (yd.shape == 2)) :
        n_classes = 2
    else:
        n_classes = yd.shape[1]
    
    if yd.shape[1] == 1:
        x_0 = np.where(yd[:,0] == 0); x_0 = x_0[0]
        x_1 = np.where(yd[:,0] == 1); x_1 = x_1[0]
        plt.plot(X[0,x_0], X[1,x_0], 'bo')
        plt.plot(X[0,x_1], X[1,x_1], 'rx')
        classes = ['classe 0', 'classe 1']
    
    else:
        labels = ['bo', 'rx', 'cv', 'kX','gD', 'm+', 'r*', 'b^', 'rs', 'k8']
        classes = []
        for i in range(n_classes):
            a = np.where(yd[:,i] == 1)
            a = a[0]
            plt.plot(X[0,a], X[1,a], labels[i])
            classes.append('classe '+str(i))
    
    plt.xlabel('x1', fontsize = 14)
    plt.ylabel('x2', fontsize = 14)
    first_legend = plt.legend(classes, loc = 'lower left')
    ax = plt.gca().add_artist(first_legend)


# **6 - Função de geração de gráfico de separação:** <br />
# Essa função serve para gerar um gráfico mostrando a fronteira de decisão (uma vez que os dados estão em 2D, há essa possiblidade. <br />
# A fronteira de decisão foi retirada da função de decisão $u = w_1x_1 + w_2x_2 + b$. <br />
# Tomando $u = 0$ e isolando $x_2$, tem-se: $x_2 = \frac{1}{w_2}(-w_1x_1 - b)$. <br />
# Assim, tomando uma determinada faixa de valores para $x_1$, pode ser gerada a fronteira de separação.

# In[6]:


def plotareta(w,b,x1):
    x2 = (1/w[0,1]) * (-w[0,0]*x1 - b)
    p1, = plt.plot(x1,x2, 'k--', label = 'reta de separação')
    plt.title('Gráfico de separação do Perceptron', fontsize = 14)
    plt.legend(handles = [p1], loc = 'lower right')
    plt.show()
    


# **7 - Teste do treinamento e Rede Neural treinada para função AND:**<br />
# Primeiramente, foi testado o algoritmo de treinamento e a rede treinada para a seguinte base de dados que modela a função lógica AND, sendo $X$ as entradas e $y_d$ a saída desejada. <br />
# $ X = \begin{bmatrix} 0 & 0 & 1 & 1 \\  0 & 1 & 0 & 1 \end{bmatrix} $. &nbsp;&nbsp; e &nbsp;&nbsp; $y_d = \begin{bmatrix} 0 & 0 & 0 & 1 \end{bmatrix}$ <br />
# Os pesos e bias foram inicializados aleatóriamente entre -1 e 1.
# 

# In[7]:


X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
yd = np.array([[0, 0, 0, 1]]).transpose()
W = np.random.uniform(-1, 1, (1, X.shape[0]))
b = np.random.uniform(-1, 1)
tolerancia = 0.001
alfa = 1.2
max_epocas = 10
W, b, vetor_seq = treina_perceptron(W, b, X, yd, alfa, max_epocas, tolerancia)

plt.figure(1)
plot_seq(vetor_seq)
y, u = perceptron(X, W, b)
print('saida antes da funcao de ativação:',u)
print('saida da rede:', y)
plt.figure(2)
plotadc2d(X,yd)
intervalo = np.linspace(-0.1,1.1, num = 13)
plotareta(W,b,intervalo)


# In[8]:


yd


# ### Comentários:
# Após a geração da função perceptron foi possível, além de visualizar como cada classe se dispõe no espaço, observa-se, também, como o perceptron classifica os dados. Desta forma, é possível ter uma clara visualização da fronteira de decisão da Rede Neural.

# **8 - Teste do treinamento e Rede Neural treinada para função OR:** <br />
# Em seguida ao teste aplicado para a função AND, foi testado o algoritmo de treinamento e a rede treinada para a seguinte base de dados que modela a função lógica OR, sendo $X$ as entradas e $y_d$ a saída desejada.<br /> 
# $ X = \begin{bmatrix} 0 & 0 & 1 & 1 \\  0 & 1 & 0 & 1 \end{bmatrix} $. &nbsp;&nbsp; e &nbsp;&nbsp; $y_d = \begin{bmatrix} 0 & 0 & 0 & 1 \end{bmatrix}$ <br />
# Os pesos e bias foram inicializados aleatóriamente entre -1 e 1.

# In[16]:


X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
yd = np.array([[0, 1, 1, 1]]).transpose()
W = np.random.uniform(-1, 1, (1, X.shape[0]))
b = np.random.uniform(-1, 1)
tolerancia = 0.001
alfa = 1.2
max_epocas = 10
W, b, vetor_seq = treina_perceptron(W, b, X, yd, alfa, max_epocas, tolerancia)

plt.figure(1)
plot_seq(vetor_seq)
y, u = perceptron(X, W, b)
print('saida antes da funcao de ativação:',u)
print('saida da rede:', y)
plt.figure(2)
plotadc2d(X,yd)
intervalo = np.linspace(-0.1,1.1, num = 13)
plotareta(W,b,intervalo)


# ### Comentários:
# Assim como na função AND, na função OR também é possível observar a tomada de decisão, mostrando a diferença na posição da reta de acordo com a disposição das classes no espaço bidimensional.

# ## Conclusões:
# Por meio desta experiência, foi posssível praticar e observar como gerar a fronteira de decisão para casos de dados dispostos em 2D. A fronteira de separação permite analisar os limites que cada classe pode atingir sem ser classificada erroneamente pelo modelo. Tal prática também é interessante para analisar a disposição das classes no espaço a fim de observar um melhor ajuste do modelo a ser utilizado.
