import numpy as np
import matplotlib.pyplot as plt

"""
Função: yperceptron
Desenvolvida em: Python 3.9 em 06/09/2021
Bibliotecas Utilizadas: numpy 1.21.2  
Finalidade: Via função degrau (threshold) determina a saída do neurônio
Parametros: W: Vetor de pesos de cada entrada do neurônio
            b: Valor do bias 
            X: Matriz de amostra de dados         
Retorno: y: Saída no neurônio para cada amostra X
"""
# Definir função perceptron
def yperceptron(W, b, X):
    y = np.dot(W, X) + b
    for i in range(len(y)):
        if y[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
    return y

"""
Função: treina_perceptron
Desenvolvida em: Python 3.9 em 14/09/2021
Bibliotecas Utilizadas: numpy 1.21.2 e matplotlib 3.4.3 
Finalidade: Ajustar os pesos repetidamente do neurônio para minimizar alguma medida de erro no conjunto de treinamento
Parametros: W:          Matriz de pesos; para cada neurônio k = 1
            b:          Vetor (k x 1) com valor do bias de cada neurônio
            X:          Matriz (m x N) com as amostras dos dados em colunas
            yd:         Matriz (k x N) com as saídas desejadas para cada amostra de dados(X)
            alfa:       Taxa de correção do peso; taxa de aprendizagem
            max_epocas: Valor máximo de épocas de treinamento
            tol:        Erro máximo tolerável
Retorno: W: Matriz de pesos ajustada
         b: bias ajustado
         VetorSEQ: Somatório dos erros quadráticos por épocas     
"""
def treina_perceptron(W, b, X, yd, alfa, maxEpocas, tol):
    N = X.shape[1]  # Número de amostras de X
    SEQ = tol  # Somatório dos Erros Quadráticos
    Epoca = 1  # O treinamento do neurônio começa na época 1
    VetorSEQ = np.array([])  # Vetor para armazenar o SEQ
    while (Epoca <= maxEpocas) and (SEQ >= tol):
        # Inicia a sequencia da Epoca
        SEQ = 0
        for i in range(N):
            y = yperceptron(W, b, X[:, i])  # Determina a saída para amostra i
            erro = yd[i] - y  # Determinar o erro
            W = W + (alfa * erro * X[:, i])  # Atualiza o vetor de pesos
            b = b + alfa * erro  # Atualiza o bias
            SEQ = SEQ + (erro ** 2)
        VetorSEQ = np.append(VetorSEQ, SEQ)  # Salva o SEQ da Epoca
        Epoca = Epoca + 1
    return W, b, VetorSEQ, Epoca


def plotareta(W, b, intervalo, plot=False):
    x1 = np.arange(intervalo[0] - 0.1, intervalo[1] + 0.1, 0.1)
    #                                w1.x1 + w2.x2 + b = 0 OU
    # x2 = -(W[0,0] * x1 + b) / W[0,1]   # -(w1.x1 + b0) / w2
    x2 = (1 / W[0, 1]) * (-W[0, 0] * x1 - b)
    x2 = x2[0, :]
    plt.plot(x1, x2)
    # plt.xlim(x1.min(), x2.max())
    if plot:
        plt.show()


##########################
def plotadc2d(X, yd):
    if ((yd.shape[1] == 1) or (yd.shape == 2)):
        n_classes = 2
    else:
        n_classes = yd.shape[1]

    if yd.shape[1] == 1:
        x_0 = np.where(yd[:, 0] == 0);
        x_0 = x_0[0]
        x_1 = np.where(yd[:, 0] == 1);
        x_1 = x_1[0]
        plt.plot(X[0, x_0], X[1, x_0], 'bo')
        plt.plot(X[0, x_1], X[1, x_1], 'rx')
        classes = ['classe 0', 'classe 1']

    else:
        labels = ['bo', 'rx', 'cv', 'kX', 'gD', 'm+', 'r*', 'b^', 'rs', 'k8']
        classes = []
        for i in range(n_classes):
            a = np.where(yd[:, i] == 1)
            a = a[0]
            plt.plot(X[0, a], X[1, a], labels[i])
            classes.append('classe ' + str(i))

    plt.xlabel('x1', fontsize=14)
    plt.ylabel('x2', fontsize=14)
    first_legend = plt.legend(classes, loc='lower left')
    ax = plt.gca().add_artist(first_legend)

# Função de geração de dados gaussianos
# Esta função permite a geração de dados aleatórios com distribuição de probabilidades gaussiana,
# cuja função PDF é dada por:
# $P(X) = \frac{1}{2\pi\sigma^2}e^\frac{(x-\mu)^2}{2\sigma^2}$ <br />
# A função gera os dados baseados nesta distribuição por meio do fornecimento do número de classes $nc$, número de amostras para cada classe $npc$, das médias de cada variável de cada classe $mc$ e das variâncias de cada classe $varc$.

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

# Função de mistura do conjunto de dados:
# Esta função embaralha os dados sequenciados por classe
# Assim, os dados ficam melhor distribuidos, melhorando o processo de treinamento da Rede Neural.

def mistura(X, yd):
    dados = np.concatenate((X, yd), axis=0).transpose()
    np.random.shuffle(dados)
    xp = dados[:, 0:2].transpose()
    yp = dados[:, 2]
    yp = yp.reshape(1, yp.shape[0])

    return xp, yp

# Definindo a função do adaline
# Similar ao perceptron com a diferença da função de ativação, foi reaproveitado o código do perceptron
# com os parâmetros de entrada e saída com a remoção da função degrau de saída.
# Logo, foi utilizada uma função de ativação linear do tipo $y = mu+c$.
# Sendo $u$ a saida do neurônio, na forma matricial $u  = wX + b$, $m$ o coeficiente linear e $c$ o coeficiente angular.
# Considerando $m = 1$ e $c = 0$, pode-se associar $y = u$. Logo, tendo somente a saída do produto matricial.

def yadaline(X, w, b):
    y = np.dot(w,X)+b
    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    return y

# Definindo a função de treinamento do adaline:
# A função de treinamento do adaline consiste no ajuste automático dos pesos e bias do neurônio de acordo com o
# erro de regressão. <br />
# O treinamento usa como medida de custo a soma do erro quadrático (SEQ) e os critérios de parada são o custo ser menor
# que a medida de tolerância ou o número de épocas atingir o máximo especificado pelo usuário.
# A função de ajuste dos pesos do adaline é igual a do perceptron, exceto pela chamada da saída
# (no lugar da chamada do perceptron é chamado o adaline).

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

# Função de geração de gráficos de evolução do erro
# Esta função plota o gráfico do erro em relação às epocas de treinamento de maneira a ilustrar a evolução do erro
# e a convergência do algoritmo.

def plot_seq(vetor_seq):
    hor = np.arange(1, vetor_seq.shape[0]+1)
    hor = hor.astype(int)
    plt.plot(hor,vetor_seq,'.-')
    plt.ylabel('SEQ', fontsize = 14)
    plt.xlabel('epocas', fontsize = 14)
    plt.title('Grafico de evolução do erro', fontsize = 14)

# **5 - Função de mistura do conjunto de dados:** <br />
# Esta função embaralha os dados sequenciados de maneira a tornar a distribuição dos dados aleatória,
# reduzindo viés no treinamento, otimizando o processo.

def mistura(X, yd):
    if len(X.shape) == 1:
        X = X[:, np.newaxis]

    if len(yd.shape) == 1:
        yd = yd[:, np.newaxis]

    dados = np.concatenate((X, yd), axis=1)
    np.random.shuffle(dados)
    xp = dados[:, 0:-1].transpose()
    yp = dados[:, -1].transpose()

    if len(yp.shape) == 1:
        yp = yp[:, np.newaxis]

    return xp, yp













