"""Interface gráfica para classificar uma foto de nuvem com a MLP treinada.

Execute: python interface_grafica.py
"""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from classificar_imagem import preparar_entrada
from src.config import CLASSES
from src.mlp import MLP


RAIZ = Path(__file__).resolve().parent
MODELO = RAIZ / "modelos" / "mlp_etapa4.npz"
ROTULOS = {
    "stratus": "Stratus",
    "cumulus": "Cumulus",
    "cirrus": "Cirrus",
    "cumulonimbus": "Cumulonimbus",
}


def classificar(caminho, rede):
    """Devolve as probabilidades usando o mesmo preparo do programa de terminal."""
    vetor = preparar_entrada(caminho, rede.tamanhos[0])
    return rede.forward(vetor[None, :])[0]


class InterfaceNuvens:
    def __init__(self, janela, rede):
        self.janela = janela
        self.rede = rede
        self.foto = None  # Mantém a imagem Tk viva enquanto ela estiver na tela.
        janela.title("Classificador de Nuvens")
        janela.geometry("680x660")
        janela.minsize(580, 580)

        quadro = ttk.Frame(janela, padding=22)
        quadro.pack(fill="both", expand=True)
        quadro.columnconfigure(0, weight=1)

        ttk.Label(quadro, text="Classificador de Nuvens",
                  font=("Segoe UI", 21, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(quadro, text="Escolha uma foto para ver a previsão da MLP.",
                  font=("Segoe UI", 11)).grid(row=1, column=0, sticky="w", pady=(4, 14))

        self.botao = ttk.Button(quadro, text="Escolher imagem...", command=self.escolher_imagem)
        self.botao.grid(row=2, column=0, sticky="w")
        self.nome_arquivo = tk.StringVar(value="Nenhuma imagem selecionada")
        ttk.Label(quadro, textvariable=self.nome_arquivo, wraplength=610).grid(
            row=3, column=0, sticky="w", pady=(10, 8))

        self.previa = ttk.Label(quadro, text="A imagem aparecerá aqui", anchor="center",
                                relief="solid")
        self.previa.grid(row=4, column=0, sticky="nsew", pady=(0, 16))
        quadro.rowconfigure(4, weight=1)

        self.previsao = tk.StringVar(value="Classe prevista: —")
        ttk.Label(quadro, textvariable=self.previsao,
                  font=("Segoe UI", 14, "bold")).grid(row=5, column=0, sticky="w", pady=(0, 10))

        self.barras = []
        self.valores = []
        for indice, classe in enumerate(CLASSES):
            linha = ttk.Frame(quadro)
            linha.grid(row=6 + indice, column=0, sticky="ew", pady=4)
            linha.columnconfigure(1, weight=1)
            ttk.Label(linha, text=ROTULOS[classe], width=16).grid(row=0, column=0, sticky="w")
            barra = ttk.Progressbar(linha, maximum=100, value=0)
            barra.grid(row=0, column=1, sticky="ew", padx=(8, 12))
            valor = tk.StringVar(value="—")
            ttk.Label(linha, textvariable=valor, width=8, anchor="e").grid(row=0, column=2)
            self.barras.append(barra)
            self.valores.append(valor)

        ttk.Label(quadro, text="Porcentagens estimadas pelo modelo; não garantem certeza.",
                  wraplength=610).grid(row=10, column=0, sticky="w", pady=(14, 0))

    def escolher_imagem(self):
        caminho = filedialog.askopenfilename(
            title="Escolha uma imagem de nuvem",
            initialdir=str(RAIZ / "imagens_teste"),
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("Todos os arquivos", "*.*")],
        )
        if caminho:
            self.mostrar_resultado(Path(caminho))

    def mostrar_resultado(self, caminho):
        try:
            # Calcula e abre a prévia antes de trocar o resultado exibido.
            probabilidades = classificar(caminho, self.rede)
            with Image.open(caminho) as imagem:
                previa = ImageOps.exif_transpose(imagem).convert("RGB")
                previa.thumbnail((570, 290), Image.Resampling.LANCZOS)
                foto = ImageTk.PhotoImage(previa)
        except (OSError, ValueError, KeyError) as erro:
            messagebox.showerror("Não foi possível classificar", str(erro), parent=self.janela)
            return

        self.foto = foto
        self.previa.configure(image=foto, text="")
        self.nome_arquivo.set(f"Imagem: {caminho.name}")
        indice = int(probabilidades.argmax())
        self.previsao.set(f"Classe prevista: {ROTULOS[CLASSES[indice]]}")
        for barra, valor, probabilidade in zip(self.barras, self.valores, probabilidades):
            barra.configure(value=float(probabilidade) * 100)
            valor.set(f"{probabilidade:.1%}")


def main():
    try:
        rede = MLP.carregar(MODELO)
        if rede.tamanhos[-1] != len(CLASSES):
            raise ValueError("O modelo não tem as quatro classes esperadas")
    except (OSError, ValueError, KeyError) as erro:
        messagebox.showerror("Modelo indisponível", f"Não foi possível carregar {MODELO}:\n{erro}")
        return
    janela = tk.Tk()
    InterfaceNuvens(janela, rede)
    janela.mainloop()


if __name__ == "__main__":
    main()
