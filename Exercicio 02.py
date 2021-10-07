# ALgoritmo de treinamento do perceptron

# Importar bibliotecas
import numpy as np
from matplotlib import pyplot as plt


# Definir função perceptron
def yperceptron(W, b, X):
    y = np.dot(W, X) + b
    for i in range(len(y)):
        if y[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
    return y


# Define função de treinamento
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


##########################


# Parâmetros para o treinamento do perceptron função lógica AND
W = np.random.uniform(-1, 1, (1, 2))
b = np.random.uniform(-1, 1, (1, 1))
W_and = W
W_or = W
b_and = b
b_or = b

X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])

yd = np.array([[0], [0], [0], [1]])  # Saída desejável
alfa = 1.2  # Taxa de aprendizado do neurônio
max_epocas = 10
tol = 0.001  # Erro máximo admitido
print('-----------------------------------------------------------')
print('Executando para a função lógica AND')
W, b, VetorSEQ, Epoca = treina_perceptron(W_and, b_and, X, yd, alfa, max_epocas, tol)

#y = []
#for i in range(X.shape[1]):
#    y.append(yperceptron(W, b, X[:, i]))


print("W = ", W)
print("b =", b)
print("VetorSEQ =", VetorSEQ)
print("Epoca =", Epoca)

# Gráfico treinamento da Função lógica AND
plt.plot(list(range(0, Epoca - 1)), VetorSEQ.tolist(), marker=".")
plt.title(" Evolução do SEQ no treinamento da Função lógica AND")
plt.xlabel("Época")
plt.ylabel("SEQ")
plt.show()

plotadc2d(X=X, yd=yd)
plotareta(W=W, b=b, intervalo=(0, 1), plot=True)

# Parâmetros para o treinamento do perceptron função lógica OR

yd = np.array([[0], [1], [1], [1]])  # Saída desejável
alfa = 1.2
max_epocas = 10
tol = 0.001
print('-----------------------------------------------------------')
print('Executando para a função lógica OR')
W, b, VetorSEQ, Epoca = treina_perceptron(W_or, b_or, X, yd, alfa, max_epocas, tol)

#y = []
#for i in range(X.shape[1]):
#    y.append(yperceptron(W, b, X[:, i]))
#print('y = ', y)

print("W = ", W)
print("b =", b)
print("VetorSEQ =", VetorSEQ)
print("Epoca =", Epoca)

# Gráfico treinamento da Função lógica OR
plt.plot(list(range(0, Epoca - 1)), VetorSEQ.tolist(), marker=".")
plt.title("Evolução do SEQ no treinamento da Função lógica OR")
plt.xlabel("Época")
plt.ylabel("SEQ")
plt.show()

plotadc2d(X=X, yd=yd)
plotareta(W=W, b=b, intervalo=(0, 1), plot=True)