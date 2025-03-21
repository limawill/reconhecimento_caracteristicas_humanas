<p align="center">
    <img src="imagens/Rostos_Detectado.png" align="center" width="80%">
</p>
<p align="center"><h1 align="center">Reconhecimento Facial de Sexo, Emoção e Idade com CNNs</h1></p>
<br>

## Motivação

Bem-vindo ao repositório do meu Trabalho de Conclusão de Curso (TCC)! Este projeto desenvolve algoritmos em Python para reconhecimento facial, focando na estimativa de idade, classificação de sexo e identificação de emoções por meio de Redes Neurais Convolucionais (CNNs). A abordagem utiliza três modelos distintos, cada um otimizado para uma tarefa específica, visando alta precisão e eficiência em aplicações como sistemas de segurança e marketing.

## Descrição

Desenvolvido como parte do meu TCC em Data Science e Analytics no MBA USP Esalq, este projeto explora o uso de CNNs especializadas para análise facial. Os modelos foram treinados em datasets específicos:

- **UTKFace**: Estimativa de idade e classificação de sexo.
- **CK+ (Cohn-Kanade Plus)**: Reconhecimento de emoções.
- **CelebA**: Validação visual.

## Resultados Detalhados

- Alta precisão na classificação de sexo e emoções, com _InceptionV3_ como destaque.
- Baixo Erro Médio Absoluto (MAE) na estimativa de idade.
- Viabilidade comprovada para aplicações práticas.

| Tarefa                    | Métrica             | Valor    |
| ------------------------- | ------------------- | -------- |
| Classificação de Sexo     | Acurácia            | 95%      |
| Estimativa de Idade       | Erro Médio Absoluto | 4.2 anos |
| Reconhecimento de Emoções | F1-Score            | 0.89     |

<p align="center">
    <img src="curva_aprendizado.png" alt="Curva de Aprendizado" width="60%">
</p>

## Tecnologias Utilizadas

- Python 3.11
- TensorFlow e Keras (para CNNs)
- Jupyter Notebook (análise e visualização)
- OpenCV (processamento de imagens)
- Matplotlib e Seaborn (gráficos)

# Pré-requisitos

- Python 3.11 ou superior
- Jupyter Notebook
- Bibliotecas:
  - `tensorflow`
  - `keras`
  - `pandas`
  - `numpy`
  - `opencv-python` (para manipulação de imagens)
  - `matplotlib` (para visualizações)
  - `seaborn` (para visualizações)

## Instalação

### 1. Clone o repositorio:

```sh
❯ git clone https://github.com/limawill/reconhecimento_caracteristicas_humanas.git
```

### 2. Acesse a pasta:

```sh
❯ cd reconhecimento_caracteristicas_humanas
```

### 3. Crie um ambiente virtual:

```sh
❯ python -m venv .venv
❯ source .venv/bin/activate  # Linux/Mac
❯ venv\Scripts\activate     # Windows
```

### 4. Instale as dependências:

```sh
❯ pip install -r requirements.txt
```

### 5. Baixe os arquivos .h5:

Acesse: [Handlebars templates](http://handlebarsjs.com/) e faça download do arquivo zip. Descompacte e copie as pastas `humor`, `idade` e`sexo` para pasta `modelos`.

## Uso

- Caso queria ver executar os scripts de treinamento:

```sh
❯ jupyter notebook CNNs/tcc-idade.ipynb  # Estimativa de idade
❯ jupyter notebook CNNs/tcc-sexo.ipynb   # Classificação de sexo
❯ jupyter notebook CNNs/tcc-humor.ipynb  # Reconhecimento de emoções
```

- Detecção do rosto e estimativas de idade da lib DeepFace

```sh
❯ python deepface_rostos.py
```

- Teste de captura do rosto:

```sh
❯ python detector_rostos.py
```

- Detecção do rosto e estimativas de idade, humor e sexo:

```sh
❯ python face_detection_final.py
```

- Selecione qual CNN deseja executar:

<p align="center">
<img src="imagens/menu1.png" align="center" width="80%">
</p>

- Selecione qual tipo de entrada imagens ou video:

<p align="center">
<img src="imagens/menu2.png" align="center" width="80%">
</p>

## Exemplos de Saída

- Captura de rosto em imagens:
<p align="center">
<img src="imagens/resultado_foto.png" align="center" width="60%">
</p>

- Captura de rosto em video (webcam):
<p align="center">
<img src="imagens/resultado_video.png" align="center" width="60%">
</p>

## Visão Geral da Estrutura

```sh
└── reconhecimento_caracteristicas_humanas.git/
    ├── CNNs
    │   ├── tcc-humor.ipynb
    │   ├── tcc-idade.ipynb
    │   └── tcc-sexo.ipynb
    ├── LICENSE
    ├── README.md
    ├── deepface_rostos.py
    ├── detector_rostos.py
    ├── face_detection_final.py
    ├── face_detection_with_h5_idade.py
    ├── modelos
    │   ├── humor
    │   ├── idade
    │   └── sexo
    └── requirements.txt
```

## Estrutura do Projeto

Aqui descrição do repositório e o propósito de cada arquivo/pasta:

- **`CNNs/`**: Contém os notebooks Jupyter usados no desenvolvimento e análise.
  - `tcc-humor.ipynb`: Notebook usado para treinamento e avaliação do modelo de reconhecimento de emoções.
  - `tcc-idade.ipynb`: Notebook usado para estimativa de idade com CNNs.
  - `tcc-sexo.ipynb`: Notebook usado para classificação de sexo.
- **`modelos/`**: Armazena os arquivos `.h5` dos modelos treinados.
  - `humor/`: Modelos para reconhecimento de emoções.
  - `idade/`: Modelos para estimativa de idade.
  - `sexo/`: Modelos para classificação de sexo.
- **`deepface_rostos.py`**: Script para análise facial usando a biblioteca DeepFace (primeiro estudo).
- **`detector_rostos.py`**: Script de estudo do openCV para capturar e a detecção de rostos em imagens ou vídeos.
- **`face_detection_final.py`**: Script principal para executar a detecção e classificação facial.
- **`face_detection_with_h5_idade.py`**: Script inicial para estimativa de idade usando um modelo `.h5`.
- **`requirements.txt`**: Lista de dependências do projeto.
- **`LICENSE`**: Arquivo com a licença do projeto.
- **`README.md`**: Este arquivo.

_Nota_: Os datasets (UTKFace, CK+, CelebA) não estão incluídos devido ao tamanho, mas os links para download estão no passo 5 do item Instalação.

## Limitações

- Os modelos podem ter desempenho reduzido em imagens de baixa resolução.
- O reconhecimento de emoções é limitado às categorias do CK+.

## Próximos Passos

- Otimizar os modelos pra execução em tempo real.
- Adicionar suporte a mais emoções e faixas etárias.

## Contribuições

- **💬 [Participe das Discussões](https://github.com/limawill/reconhecimento_caracteristicas_humanas.git/discussions)**: Compartilhe suas ideias, dê feedback ou faça perguntas.
- **🐛 [Reporte Problemas](https://github.com/limawill/reconhecimento_caracteristicas_humanas.git/issues)**: Envie bugs encontrados ou registre pedidos de novas funcionalidades para o projeto `reconhecimento_caracteristicas_humanas.git`.
- **💡 [Envie Pull Requests](https://github.com/limawill/reconhecimento_caracteristicas_humanas.git/blob/main/CONTRIBUTING.md)**: Revise PRs abertos e envie seus próprios PRs.

<details closed>
<summary>Diretrizes de Contribuição</summary>

1. **Faça um Fork do Repositório**: Comece fazendo um fork do repositório do projeto para sua conta no GitHub.
2. **Clone Localmente**: Clone o repositório forkado para sua máquina local usando um cliente Git.
   ```sh
   ❯ git clone https://github.com/limawill/reconhecimento_caracteristicas_humanas.git
   ```
3. **Crie uma Nova Branch**: Sempre trabalhe em uma nova branch, dando a ela um nome descritivo.

```sh
❯ git checkout -b nova-funcionalidade-x
```

4. **Faça Suas Alterações**: Desenvolva e teste suas mudanças localmente.
5. **Commit Suas Alterações**: Faça o commit com uma mensagem clara descrevendo suas atualizações.

```sh
❯ git commit -m 'Implementada nova funcionalidade x.'
```

6. **Envie para o GitHub**: Envie as alterações para seu repositório forkado.

```sh
❯ git push origin nova-funcionalidade-x
```

7. **Envie um Pull Request**: Crie um PR para o repositório original do projeto. Descreva claramente as mudanças e suas motivações.
8. **Revisão**: Após seu PR ser revisado e aprovado, ele será mesclado ao branch principal. Parabéns pela sua contribuição!
</details>

## Datasets

- [UTKFace](https://susanqq.github.io/UTKFace/)
- [CK+](http://www.jeffcohn.net/Resources/)
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## Agradecimentos

Agradeço ao meu orientador Ricardo James, à USP Esalq e ao Grok da xAI por me ajudar com este README!

## Licença

Este projeto está sob a [MIT License](https://github.com/limawill/reconhecimento_caracteristicas_humanas/blob/master/LICENSE)
