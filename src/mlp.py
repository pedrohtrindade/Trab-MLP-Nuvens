import numpy as np
def relu(z):
  
    return np.maximum(0, z)


def softmax(logits):
    
    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("logits deve ter formato (amostras, classes)")
    # Subtrair o maior valor de cada linha evita overflow em exp().
    deslocados = logits - np.max(logits, axis=1, keepdims=True)
    exponenciais = np.exp(deslocados)
    return exponenciais / np.sum(exponenciais, axis=1, keepdims=True)


class MLP:
   

    def __init__(self, n_entradas=1024, camadas_ocultas=(128, 64),
                 n_saidas=4, semente=42):
        tamanhos = (n_entradas, *camadas_ocultas, n_saidas)
        if any(not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n <= 0
               for n in tamanhos):
            raise ValueError("Todos os tamanhos de camada devem ser inteiros positivos")

        self.tamanhos = tamanhos
        self.pesos = []
        self.bias = []
        gerador = np.random.default_rng(semente)

        for indice, (entrada, saida) in enumerate(zip(tamanhos[:-1], tamanhos[1:])):
            # He para camadas ReLU; Glorot para a camada de saída.
            escala = (np.sqrt(2.0 / entrada) if indice < len(tamanhos) - 2
                      else np.sqrt(2.0 / (entrada + saida)))
            self.pesos.append(
                (gerador.standard_normal((entrada, saida), dtype=np.float32)
                 * np.float32(escala)).astype(np.float32)
            )
            self.bias.append(np.zeros((1, saida), dtype=np.float32))

    def forward(self, X, retornar_cache=False):
      
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.tamanhos[0]:
            raise ValueError(f"X deve ter formato (amostras, {self.tamanhos[0]})")
        if X.shape[0] == 0:
            raise ValueError("X deve conter pelo menos uma amostra")
        if not np.all(np.isfinite(X)):
            raise ValueError("X deve conter apenas valores finitos")

        ativacoes = [X]
        pre_ativacoes = []
        atual = X
        for pesos, bias in zip(self.pesos[:-1], self.bias[:-1]):
            z = atual @ pesos + bias
            pre_ativacoes.append(z)
            atual = relu(z)
            ativacoes.append(atual)

        logits = atual @ self.pesos[-1] + self.bias[-1]
        pre_ativacoes.append(logits)
        probabilidades = softmax(logits)
        ativacoes.append(probabilidades)

        if retornar_cache:
            return probabilidades, {"ativacoes": ativacoes,
                                    "pre_ativacoes": pre_ativacoes}
        return probabilidades

    def _validar_rotulos(self, y, n_amostras):
        y = np.asarray(y)
        if y.shape != (n_amostras,) or not np.issubdtype(y.dtype, np.integer):
            raise ValueError("y deve ter um rótulo inteiro por amostra")
        if np.any(y < 0) or np.any(y >= self.tamanhos[-1]):
            raise ValueError("y contém rótulos fora do intervalo das classes")
        return y

    def entropia_cruzada(self, logits, y):
       
        logits = np.asarray(logits)
        if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] != self.tamanhos[-1]:
            raise ValueError("logits deve ter formato (amostras, classes)")
        y = self._validar_rotulos(y, logits.shape[0])
        maiores = np.max(logits, axis=1, keepdims=True)
        log_soma_exp = maiores[:, 0] + np.log(
            np.sum(np.exp(logits - maiores), axis=1)
        )
        return float(np.mean(log_soma_exp - logits[np.arange(len(y)), y]))

    def perda(self, X, y):
        
        _, cache = self.forward(X, retornar_cache=True)
        return self.entropia_cruzada(cache["pre_ativacoes"][-1], y)

    def backward(self, cache, y):
       
        ativacoes = cache["ativacoes"]
        pre_ativacoes = cache["pre_ativacoes"]
        n_amostras = ativacoes[0].shape[0]
        y = self._validar_rotulos(y, n_amostras)

        # Softmax + entropia cruzada: dL/dZ = (P - Y_one_hot) / N.
        delta = ativacoes[-1].copy()
        delta[np.arange(n_amostras), y] -= 1
        delta /= n_amostras
        grad_pesos = [None] * len(self.pesos)
        grad_bias = [None] * len(self.bias)

        for camada in range(len(self.pesos) - 1, -1, -1):
            grad_pesos[camada] = ativacoes[camada].T @ delta
            grad_bias[camada] = np.sum(delta, axis=0, keepdims=True)
            if camada > 0:
                # A derivada de ReLU é 1 quando Z > 0 e 0 caso contrário.
                delta = (delta @ self.pesos[camada].T) * (pre_ativacoes[camada - 1] > 0)

        return grad_pesos, grad_bias

    def atualizar(self, grad_pesos, grad_bias, taxa_aprendizado):
       
        if not np.isfinite(taxa_aprendizado) or taxa_aprendizado <= 0:
            raise ValueError("A taxa de aprendizado deve ser positiva e finita")
        if len(grad_pesos) != len(self.pesos) or len(grad_bias) != len(self.bias):
            raise ValueError("Quantidade de gradientes diferente da quantidade de camadas")
        for indice in range(len(self.pesos)):
            if grad_pesos[indice].shape != self.pesos[indice].shape or grad_bias[indice].shape != self.bias[indice].shape:
                raise ValueError("Formato de gradiente incompatível com os parâmetros")
            self.pesos[indice] -= taxa_aprendizado * grad_pesos[indice]
            self.bias[indice] -= taxa_aprendizado * grad_bias[indice]

    def salvar(self, caminho):
       
        conteudo = {"tamanhos": np.asarray(self.tamanhos, dtype=np.int64)}
        for indice, (pesos, bias) in enumerate(zip(self.pesos, self.bias)):
            conteudo[f"pesos_{indice}"] = pesos
            conteudo[f"bias_{indice}"] = bias
        np.savez_compressed(caminho, **conteudo)

    @classmethod
    def carregar(cls, caminho):
       
        with np.load(caminho, allow_pickle=False) as dados:
            tamanhos = tuple(int(n) for n in dados["tamanhos"])
            if len(tamanhos) < 2:
                raise ValueError("Arquitetura inválida no arquivo de modelo")
            rede = cls(tamanhos[0], tamanhos[1:-1], tamanhos[-1])
            for indice in range(len(rede.pesos)):
                pesos = dados[f"pesos_{indice}"]
                bias = dados[f"bias_{indice}"]
                if pesos.shape != rede.pesos[indice].shape or bias.shape != rede.bias[indice].shape:
                    raise ValueError("Formato dos parâmetros inválido no arquivo de modelo")
                rede.pesos[indice] = pesos.copy()
                rede.bias[indice] = bias.copy()
        return rede
