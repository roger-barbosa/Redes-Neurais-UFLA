#!/usr/bin/env python
# coding: utf-8

# # Relatório Prático Nº. 5
# # Redes Neurais Artificiais
# **Aluno: Fernando Elias de Melo Borges - 2020260050** <br />
# **Curso: Engenharia de Sistemas e Automação** <br />
# **Título: Adaline** <br />
# Data: 25/01/2021

# ## Objetivo:
# Esta prática tem por objetivo o uso da predição por regressão utilizando o algoritmo adaline. O adaline consiste em um neurônio artificial como o perceptron com a diferença na função de ativação. Enquanto o perceptron de Rosenblatt utiliza um função de ativação degrau, o adaline faz uso de uma função de ativação linear na saída.

# ##  Script e Resultados:
# **1 - importando as bibliotecas a serem utilizadas:** <br />
# **numpy** para cálculos algébricos e **matplotlib** para geração de gráficos

# In[1]:


import numpy as np
import matplotlib.pyplot as plt 


# **2 - Definindo a função de treinamento do adaline:** <br />
# A função de treinamento do adaline consiste no ajuste automático dos pesos e bias do neurônio de acordo com o erro de regressão. <br />
# O treinamento usa como medida de custo a soma do erro quadrático (SEQ) e os critérios de parada são o custo ser menor que a medida de tolerância ou o número de épocas atingir o máximo especificado pelo usuário. A função de ajuste dos pesos do adaline é igual a do perceptron, exceto pela chamada da saída (no lugar da chamada do perceptron é chamado o adaline).

# In[2]:


def treina_adaline(W, b, X, yd, alfa, max_epocas, tolerancia):
    N = X.shape[1]
    vetor_seq = []
    epoca = 1
    seq = 100
    while(epoca <= max_epocas and seq >= tolerancia):
        seq = 0
        for i in range(0, N):
            y = yadaline(X[:,i], W, b)
            erro = yd[i] - y
            W = W+(alfa*erro*X[:,i])
            b = b+alfa*erro
            seq = seq + (erro**2)  
        epoca = epoca+1
        vetor_seq.append(seq[0])

    vetor_seq = np.array(vetor_seq)
    return W, b, vetor_seq


# **3 - Definindo a função do adaline:**<br />
# Similar ao perceptron com a diferença da função de ativação, foi reaproveitado o código do perceptron com os parâmetros de entrada e saída com a remoção da função degrau de saída. Logo, foi utilizada uma função de ativação linear do tipo $y = mu+c$. Sendo $u$ a saida do neurônio, na forma matricial $u  = wX + b$, $m$ o coeficiente linear e $c$ o coeficiente angular.<br />
# Considerando $m = 1$ e $c = 0$, pode-se associar $y = u$. Logo, tendo somente a saída do produto matricial.

# In[3]:


def yadaline(X, w, b):
    y = np.dot(w,X)+b
    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    return y


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


# **5 - Função de mistura do conjunto de dados:** <br />
# Esta função embaralha os dados sequenciados de maneira a tornar a distribuição dos dados aleatória, reduzindo viés no treinamento, otimizando o processo.

# In[5]:


def mistura(X,yd):
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    
    if len(yd.shape) == 1:
        yd = yd[:, np.newaxis]
    
    dados = np.concatenate((X,yd), axis = 1)
    np.random.shuffle(dados)
    xp = dados[:, 0:-1].transpose()
    yp = dados[:,-1].transpose()
    
    if len(yp.shape) == 1:
        yp = yp[:, np.newaxis]
    
    return xp, yp


# **6 - Teste das funções criadas e treinamento da Rede Neural adaline:** <br />
# Nesta seção é feito o teste inicial com um conjunto de dados que modelam a função $y = ax + b$. A partir dos dados de entrada $x$ e $y$ é treinada a rede adaline que retorna como saídas os pesos da rede e o vetor da medida de erro, no caso o somatório do erro quadrático (SEQ). Nesta seção também são apresentados a nuvem de pontos da entrada e saída e o gráfico de evolução do SEQ com o avanço das épocas de treinamento.

# In[7]:


a = 1
b = 0
x = np.linspace(0,100, num = 201)
x = x[:, np.newaxis]
y = a*x + b
xp, yp = mistura(x,y)
plt.figure()
plt.plot(x, y, 'o')
plt.xlabel('x', fontsize = 14)
plt.ylabel('y', fontsize = 14)
plt.title('Nuvem de pontos da entrada da rede adaline', fontsize = 14)

W = np.random.uniform(-1, 1, (1, xp.shape[0]))
b = np.random.uniform(-1, 1)
tolerancia = 1e-9
alfa = 1e-4
max_epocas = 10000
W, b, vetor_seq = treina_adaline(W, b, xp, yp, alfa, max_epocas, tolerancia)

plt.figure()
plot_seq(vetor_seq[1:100])


# **7 - Plotagem dos dados comparando a reta do adaline e os dados inseridos no treino:** <br />
# Nesta seção é plotada a disposição da saída do adaline em forma de reta comparando com a nuvem de pontos anteriormente gerada dos dados de treino. Nesta seção é observado como o adaline se comporta e aprende a função que descreve os dados de entrada.

# In[11]:


y_pred = yadaline(x.transpose(),W,b)
plt.figure()
plt.plot(x,y, 'ro')
plt.plot(x,y_pred.transpose(), 'b')
plt.xlabel('x', fontsize = 14)
plt.ylabel('y', fontsize = 14)
plt.legend(['Nuvem de pontos da entrada', 'Reta mostrando saída do adaline'])
plt.title('Nuvem de pontos da entrada e reta da saída rede adaline', fontsize = 14)


# **8 - Teste do adaline em entradas ruidosas:** <br />
# Nesta parte da prática, é realizada a inclusão de ruído na saída $y$ já mostrada na seção **6**. sendo a entrada $x$ mantida, foi tomada a saída e acrescido um ruído gaussiano na mesma.  <br />
# Após a inclusão do ruído na saída, foi treinada novamente a adaline com os valores de saída ruidosa e obtidos os pesos e o vetor de SEQ.<br />
# Assim como na seção **6**, nesta seção são gerados a nuvem de pontos dos dados de treino e o gráfico de evolução do SEQ em relação ao avanço das épocas de treinamento.

# In[12]:


ruido =  np.random.standard_normal(size=y.shape)
y = y + ruido

xp, yp = mistura(x,y)
plt.figure()
plt.plot(x, y, 'o')
plt.xlabel('x', fontsize = 14)
plt.ylabel('y', fontsize = 14)
plt.title('Nuvem de pontos da entrada da rede adaline', fontsize = 14)

W = np.random.uniform(-1, 1, (1, xp.shape[0]))
b = np.random.uniform(-1, 1)
tolerancia = 1e-9
alfa = 1e-4
max_epocas = 10000
W, b, vetor_seq = treina_adaline(W, b, xp, yp, alfa, max_epocas, tolerancia)

plt.figure()
plot_seq(vetor_seq)


# **9 - Plotagem da nuvem de dados com ruído e comparativo com a reta de saída do adaline:** <br />
# Nesta seção, é plotada a nova nuvem de pontos dos dados com a inclusão do ruído e novamente plotada a reta de saída do adaline.<br />
# Neste novo gráfico é observado o comportamento do adaline com relação aos ruídos presentes na entrada e como o modelo se comporta com dados ruidosos.

# In[13]:


y_pred = yadaline(x.transpose(),W,b)
plt.figure()
plt.plot(x,y, 'ro')
plt.plot(x,y_pred.transpose(), 'b')
plt.xlabel('x', fontsize = 14)
plt.ylabel('y', fontsize = 14)
plt.legend(['Nuvem de pontos da entrada', 'Reta mostrando saída do adaline'])
plt.title('Nuvem de pontos da entrada e reta da saída rede adaline', fontsize = 14)


# ### Comentários:
# Nesta experiência foi observado o uso do adaline e seu comportamento frente a dados sem ruído e dados ruidosos. Por meio de visualização gráfica, foi observado o comportamento do modelo como regressor. O interessante desta observação foi a percepção da robustez do adaline frente à dados ruidosos, onde a rede neural conseguiu capturar somente a função que descreve o modelo sem "decorar" os dados de treinamento. Tal fato se faz importante frente à regressão de sinais reais que, comumente, são aquisitados com ruído.

# ## Conclusões:
# Esta experiência permitiu a observação de outro modelo de rede neural: o adaline. Foi, inicialmente realizada uma prática com dados modelados sem ruído para entendimento do funcionamento do algoritmo e, em seguida, aplicado o modelo em dados ruidosos, simulando dados reais, estes geralmente ruidosos, como, por exemplo, ruídos provenientes de instruimentação e/ou medição. <br />
# Tal prática permitiu, além da fixação dos conceitos das aulas teóricas, a obervação de uma saída sem a presença de overfitting. Onde, caso houvesse tal efeito, a saída acompanharia totalemnte os dados de treinamento e a reta apresentaria oscilações dado o ruído presente nos dados de treino. <br />
# Por fim, foi aprendido nesta prática o uso do adaline como regressor e sua aplicação frente à dados ruidosos.
