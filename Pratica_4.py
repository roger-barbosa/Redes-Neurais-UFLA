#!/usr/bin/env python
# coding: utf-8

# # Relatório Prático Nº. 4
# # Redes Neurais Artificiais
# **Aluno: Fernando Elias de Melo Borges - 2020260050** <br />
# **Curso: Engenharia de Sistemas e Automação** <br />
# **Título: Classificação de dados gaussianos** <br />
# Data: 19/12/2020

# ## Objetivo:
# Esta experiência tem por objetivo principal a classificação de dados não binários aleatórios que possuem distribuição de probablilidades gaussiana. Além disto, permitir, também a visualização dos dados e sua reta de separação.

# ##  Script e Resultados:
# **1 - importando as bibliotecas a serem utilizadas:** <br />
# **numpy** para cálculos algébricos e **matplotlib** para geração de gráficos

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

# In[16]:


def plotadc2d(X, yd):
    n_classes = int(np.max(yd) + 1)
    plt.figure()
    labels = ['bo', 'rx', 'cv', 'kX','gD', 'm+', 'r*', 'b^', 'rs', 'k8']
    classes = []
    for i in range(n_classes):
        a = np.where(yd == i)
        a = a[1]
        plt.plot(X[0,a], X[1,a], labels[i])
        classes.append('classe '+str(i))

    plt.xlabel('x1', fontsize = 14)
    plt.ylabel('x2', fontsize = 14)
    first_legend = plt.legend(classes, loc = 'best')
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
    


# **7 - Função de geração de dados gaussianos:** <br />
# Esta função permite a geração de dados aleatórios com distribuição de probabilidades gaussiana, cuja função PDF é dada por:<br />
# $P(X) = \frac{1}{2\pi\sigma^2}e^\frac{(x-\mu)^2}{2\sigma^2}$ <br />
# A função gera os dados baseados nesta distribuição por meio do fornecimento do número de classes $nc$, número de amostras para cada classe $npc$, das médias de cada variável de cada classe $mc$ e das variâncias de cada classe $varc$.

# In[7]:


def geragauss(nc, npc, mc, varc):

    N = int(nc*npc)
    X = np.zeros((2,N))
    inicio = 0
    fim = npc
    yd = -np.ones((1,N))

    for i in range(0, nc):
        mu_x1 = mc[0,i]
        sigma_x1 = np.sqrt(varc[0,i])
        u = np.random.randn(1,npc)
        x1 = sigma_x1 * u + mu_x1

        mu_x2 = mc[1,i]
        sigma_x2 = np.sqrt(varc[1,i])
        u = np.random.randn(1,npc)
        x2 = sigma_x1 * u + mu_x2

        x = np.concatenate((x1,x2), axis = 0)
        X[:,inicio:fim] = x
        yd[0, inicio:fim] = i

        inicio = (i+1)*npc
        fim = (i+2)*npc

    return X, yd


# **8 - Função de mistura do conjunto de dados:** <br />
# Esta função embaralha os dados sequenciados por classe do item **7**. Assim, os dados ficam melhor distribuidos, melhorando o processo de treinamento da Rede Neural.

# In[8]:


def mistura(X,yd):
    dados = np.concatenate((X,yd), axis = 0).transpose()
    np.random.shuffle(dados)
    xp = dados[:, 0:2].transpose()
    yp = dados[:,2]
    yp = yp.reshape(1,yp.shape[0])
    
    return xp, yp


# **9 - Teste das funções criadas e treinamento da Rede Neural com dados gaussianos:** <br />
# Esta parte do script chama as funções anteriormente criadas. Desta forma ocorre a execução da Rede Neural com conjuntos de dados gaussianos.<br />
# Primeiramente, são gerados os dados e feita sua mistura, com posterior plotagem da distruibuição dos mesmos.<br />
# Em seguida é treinada a rede neural e gerado o gráfico de evolução do custo (neste caso, a soma do erro quadrático (SEQ)).<br />
# Por fim, é gerada a reta de separação do perceptron juntamente com a distribuição dos dados no espaço bidimensional.
# 

# In[26]:


#gerando e embaralhando os dados:
nc = 2
npc = 50
mc = np.array([[2, 2.2], [0.1, 0.3]]).transpose()
varc = np.array([[0.3, 0.1], [0.2, 0.4]]).transpose()

X, yd = geragauss(nc,npc,mc,varc)
xp, yp = mistura(X, yd)

#plotando a distribuição dos dados:
plt.figure()
plotadc2d(xp, yp)
plt.title('Distribuição dos dados')

#treinamento da Rede Neural:
W = np.random.uniform(-1, 1, (1, X.shape[0]))
b = np.random.uniform(-1, 1)
plt.figure()
plot_seq(vetor_seq)

#gerando saída do perceptron e plot dos dados com a reta de separação:
y, u = perceptron(xp, W, b)
print('saida antes da funcao de ativação:',u)
print('saida da rede:', y)
plt.figure()
plotadc2d(X,yd)
intervalo = np.linspace(-1,3, num = 13)
plotareta(W,b,intervalo)

plt.show()


# ### Comentários:
# Foi observada a boa capacidade do perceptron de camada única de classificar dados linearmente separáveis. Por meio do teste de um conjunto de dados contendo uma determinada distribuição de probabilidades. O embaralhamento dos dados, além de melhorar a distruição do conjunto, não deixando os dados sequenciados por classe, também, via testes empíricos nesta experiência, tornou a convergência mais rápida, necessitando de menos épocas para ajuste dos pesos.

# ## Conclusões:
# Esta prática permitiu mostrar a classificação de dados contínuos e aleatórios por meio do perceptron de camada única. Mostrou também, a rápida convergência do perceptron nestes casos por meio do baixo número de épocas necessárias para a convergência do modelo.<br />
# A experiência também permitiu analisar o efeito do teorema de convergência do perceptron, sendo que o mesmo sempre convergirá quando se tem conjuntos de dados linearmente separáveis.
