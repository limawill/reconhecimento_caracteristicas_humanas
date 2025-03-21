<p align="center">
    <img src="Rostos_Detectado.png" align="center" width="80%">
</p>
<p align="center"><h1 align="center">Reconhecimento Facial de Sexo, Emoção e Idade com CNNs</h1></p>
<br>

Bem-vindo ao repositório do meu Trabalho de Conclusão de Curso (TCC)! Este projeto desenvolve algoritmos em Python para reconhecimento facial, focando na estimativa de idade, classificação de sexo e identificação de emoções por meio de Redes Neurais Convolucionais (CNNs). A abordagem utiliza três modelos distintos, cada um otimizado para uma tarefa específica, visando alta precisão e eficiência em aplicações como sistemas de segurança e marketing.

## Descrição

Desenvolvido como parte do meu TCC em Data Science e Analytics no MBA USP ESSALQ, este projeto explora o uso de CNNs especializadas para análise facial. Os modelos foram treinados em datasets específicos:

- **UTKFace**: Estimativa de idade e classificação de sexo.
- **CK+ (Cohn-Kanade Plus)**: Reconhecimento de emoções.
- **CelebA**: Validação visual.

### Resultados

- Alta precisão na classificação de sexo e emoções, com _InceptionV3_ como destaque.
- Baixo Erro Médio Absoluto (MAE) na estimativa de idade.
- Viabilidade comprovada para aplicações práticas e construção de modelos personalizados.

Este repositório contém o código-fonte, o notebook Jupyter com a análise e os modelos, além de instruções para reproduzir os experimentos.

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
  - `seabor` (para visualizações)

## Project Structure

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

## Instalação

### 1. Clone o repositorio:

```sh
git clone https://github.com/limawill/reconhecimento_caracteristicas_humanas.git
```

### 2. Acesse a pasta:

```sh
cd reconhecimento-facial-tcc
```

### 3. Crie um ambiente virtual:

```sh
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 4. Instale as dependências:

```sh
pip install -r requirements.txt
```

### 5. Baixe os arquivos .h5:

Acesse:
Faça download do arquivo zip, descompacte e copia as 3 pastas:

- `humor`
- `idade`
- `sexo`

Volte a pasta

```sh
cd reconhecimento-facial-tcc
```

e cole o conteudo dentro da pasta modelos

## Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
  - [ Project Index](#-project-index)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Testing](#-testing)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---

## Overview

<code>❯ REPLACE-ME</code>

---

## Features

<code>❯ REPLACE-ME</code>

---

### Project Index

<details open>
	<summary><b><code>RECONHECIMENTO_CARACTERISTICAS_HUMANAS.GIT/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/limawill/reconhecimento_caracteristicas_humanas.git/blob/master/deepface_rostos.py'>deepface_rostos.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/limawill/reconhecimento_caracteristicas_humanas.git/blob/master/detector_rostos.py'>detector_rostos.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/limawill/reconhecimento_caracteristicas_humanas.git/blob/master/face_detection_final.py'>face_detection_final.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/limawill/reconhecimento_caracteristicas_humanas.git/blob/master/face_detection_with_h5_idade.py'>face_detection_with_h5_idade.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/limawill/reconhecimento_caracteristicas_humanas.git/blob/master/requirements.txt'>requirements.txt</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- CNNs Submodule -->
		<summary><b>CNNs</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/limawill/reconhecimento_caracteristicas_humanas.git/blob/master/CNNs/tcc-humor.ipynb'>tcc-humor.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/limawill/reconhecimento_caracteristicas_humanas.git/blob/master/CNNs/tcc-idade.ipynb'>tcc-idade.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/limawill/reconhecimento_caracteristicas_humanas.git/blob/master/CNNs/tcc-sexo.ipynb'>tcc-sexo.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
</details>

---

## Getting Started

### Prerequisites

Before getting started with reconhecimento_caracteristicas_humanas.git, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python
- **Package Manager:** Pip

### Installation

Install reconhecimento_caracteristicas_humanas.git using one of the following methods:

**Build from source:**

1. Clone the reconhecimento_caracteristicas_humanas.git repository:

```sh
❯ git clone https://github.com/limawill/reconhecimento_caracteristicas_humanas.git
```

2. Navigate to the project directory:

```sh
❯ cd reconhecimento_caracteristicas_humanas.git
```

3. Install the project dependencies:

**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pip install -r requirements.txt
```

### Usage

Run reconhecimento_caracteristicas_humanas.git using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ python {entrypoint}
```

### Testing

Run the test suite using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pytest
```

---

## Project Roadmap

- [x] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

## Contributing

- **💬 [Join the Discussions](https://github.com/limawill/reconhecimento_caracteristicas_humanas.git/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/limawill/reconhecimento_caracteristicas_humanas.git/issues)**: Submit bugs found or log feature requests for the `reconhecimento_caracteristicas_humanas.git` project.
- **💡 [Submit Pull Requests](https://github.com/limawill/reconhecimento_caracteristicas_humanas.git/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/limawill/reconhecimento_caracteristicas_humanas.git
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/limawill/reconhecimento_caracteristicas_humanas.git/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=limawill/reconhecimento_caracteristicas_humanas.git">
   </a>
</p>
</details>

---

## License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

## Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---
