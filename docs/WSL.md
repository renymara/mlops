# Guia Rápido: Instalando e Usando o WSL no Windows

O WSL (Windows Subsystem for Linux) permite rodar um ambiente Linux diretamente no Windows, facilitando o uso de ferramentas e comandos comuns em projetos de ciência de dados e MLOps.

## Passo 1: Habilitar o WSL

1. Abra o PowerShell como Administrador e execute:
   ```powershell
   wsl --install
   ```
   Isso instalará o WSL e a distribuição Ubuntu padrão.

2. Reinicie o computador se solicitado.

## Passo 2: Configurar o Ubuntu

1. Após reiniciar, o Ubuntu será iniciado automaticamente para configuração inicial.
2. Crie um nome de usuário e senha Linux quando solicitado.

## Passo 3: Atualizar o Ubuntu

No terminal Ubuntu (WSL), execute:
```bash
sudo apt update && sudo apt upgrade -y
```

## Passo 4: Instalar utilitários úteis (opcional)

No Ubuntu (WSL):
```bash
sudo apt install git wget curl build-essential -y
```

## Passo 5: Instalar o Anaconda/Miniconda (opcional, recomendado)

1. Baixe o instalador do Miniconda:
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   ```
2. Instale:
   ```bash
   bash Miniconda3-latest-Linux-x86_64.sh
   ```
3. Siga as instruções na tela e feche/reabra o terminal após a instalação.

## Passo 6: Acesse sua pasta do projeto

No Ubuntu (WSL), acesse seus arquivos do Windows em `/mnt/c/Users/SEU_USUARIO/`.
Exemplo:
```bash
cd /mnt/c/Users/SEU_USUARIO/mlops
```

## Passo 7: Siga o README do projeto

Agora, dentro do terminal WSL, siga normalmente as instruções do arquivo `README.md` do projeto para instalar dependências, rodar Docker, Jupyter Lab, etc.

---

**Dica:**
- Você pode abrir o terminal WSL pelo menu iniciar digitando "Ubuntu" ou "WSL".
- O VS Code pode ser aberto dentro do WSL usando o comando `code .` (requer extensão Remote - WSL instalada no VS Code).
