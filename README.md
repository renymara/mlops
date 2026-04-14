
# Projeto MLOps com MLFlow

> **Atenção usuários Windows:**
>
> Antes de seguir as instruções abaixo, recomenda-se configurar o WSL (Windows Subsystem for Linux) para garantir compatibilidade total com Docker, Conda e comandos Linux. Siga o guia:
>
> [Guia de instalação e uso do WSL](docs/WSL.md)

## Pré-requisitos

1. Conta no GitHub (gratuita)
2. VS Code instalado ([baixar](https://code.visualstudio.com/))
3. Anaconda ou Miniconda instalado ([baixar](https://docs.conda.io/en/latest/miniconda.html))
4. Git instalado
5. Docker Desktop ([baixar](https://www.docker.com/products/docker-desktop/))

## Fazer Fork e Clonar o Repositório

1. No GitHub, acesse: https://github.com/weslleymoura/mlops
2. Clique em **Fork**
3. Depois, clone seu fork:
   ```bash
   git clone https://github.com/SEU-USUARIO/mlops.git
   cd mlops
   ```

## Criar e Ativar o Ambiente Conda

1. Crie o ambiente:
   ```bash
   conda create -n mlops-util-env python=3.11
   ```
2. Ative o ambiente:
   ```bash
   conda activate mlops-util-env
   ```
3. Instale as dependências:
   ```bash
   conda install -c conda-forge --file requirements/requirements_conda.txt
   ```

### Abrir o Jupyter Lab

1. Ative o seu ambiente conda
   ```bash
   conda activate mlops-util-env
   ```
2. Abra o Jupyter Lab:
   ```bash
   jupyter lab
   ```
3. Abra os notebooks na pasta `notebooks/`.

## Subir os Serviços Docker

1. No diretório do projeto, execute:
   ```bash
   docker compose up
   ```
2. Os serviços MLflow, MinIO e Postgres serão iniciados.

### Acessar os Serviços

- MLflow UI: http://localhost:5010
- MinIO Console: http://localhost:9001 (user: `user`, senha: `password`)
- Postgres: localhost:5433 (user: `user`, senha: `password`)

## Dicas

- Use VS Code para editar os scripts do projeto.
- Use o ambiente Conda para rodar notebooks e scripts Python.
- Pare os serviços Docker com:
  ```bash
  docker compose down
  ```
- [Arquivo de referência (comandos úteis)](docs/REFERENCE.md)

Pronto! Agora você pode trabalhar localmente com MLflow, MinIO, Postgres e Jupyter Lab.

---
