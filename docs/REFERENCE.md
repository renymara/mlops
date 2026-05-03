# Referência Rápida — Comandos Essenciais

## Status e Logs

```bash
# Ver status de todos os containers
docker compose ps

# Ver logs de todos os serviços
docker compose logs

# Ver logs do MLflow (em tempo real)
docker compose logs -f mlflow-server

# Ver logs do CloudServer (object storage)
docker compose logs -f cloudserver

# Ver logs de um serviço específico
docker logs NOME_DO_CONTAINER
```

## Gerenciar Serviços

```bash
# Iniciar todos os serviços
docker compose up -d

# Parar todos os serviços
docker compose down

# Reiniciar um serviço específico
docker compose restart mlflow-server
docker compose restart cloudserver
docker compose restart postgres

# Reconstruir e iniciar
docker compose up -d --build

# Parar e remover volumes (limpa dados)
docker compose down -v
```

## Diagnosticar Problemas

```bash
# Entrar em um container
docker exec -it mlops-mlflow-server-1 bash
```

## Gerenciar Ambientes

```bash
# Criar ambiente
conda create -n mlops-util-env python=3.11

# Ativar ambiente
conda activate mlops-util-env

# Desativar ambiente
conda deactivate

# Listar ambientes
conda env list

# Remover ambiente
conda env remove -n mlops-util-env
```

## Gerenciar Pacotes

```bash
# Instalar pacotes do arquivo
conda install -c conda-forge --file requirements_conda.txt

# Instalar pacote específico
conda install -c conda-forge NOME_DO_PACOTE
```

## MLFlow

### Configuração Básica do MLFlow

```python
import mlflow

# Configurar URI do servidor
mlflow.set_tracking_uri('http://localhost:5010')

# Verificar URI configurada
print(mlflow.get_tracking_uri())

# Criar/selecionar experimento
mlflow.set_experiment("meu-experimento")

# Listar experimentos
client = mlflow.MlflowClient()
experiments = client.list_experiments()
for exp in experiments:
    print(f"{exp.name} (ID: {exp.experiment_id})")
```

### Logging

```python
import mlflow

# Iniciar run
with mlflow.start_run():
    # Log de parâmetros
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 100)
    
    # Log de métricas
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.05)
    
    # Log de modelo
    mlflow.sklearn.log_model(model, "model")
    
    # Log de artefato
    mlflow.log_artifact("output.txt")
```

### Consultar Runs

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Buscar runs de um experimento
experiment_id = "0"
runs = client.search_runs(experiment_id)

for run in runs:
    print(f"Run ID: {run.info.run_id}")
    print(f"Metrics: {run.data.metrics}")
    print(f"Params: {run.data.params}")
```

## Git (Terminal Local)

### Operações Básicas

```bash
# Ver status
git status

# Adicionar arquivos
git add .
git add ARQUIVO_ESPECIFICO

# Commit
git commit -m "Mensagem descritiva"

# Push para o fork
git push origin main

# Pull (atualizar local)
git pull origin main

# Ver histórico
git log --oneline

# Ver diferenças
git diff
```

### Branches

```bash
# Criar nova branch
git checkout -b feature/nova-funcionalidade

# Listar branches
git branch

# Mudar de branch
git checkout main

# Merge de branch
git checkout main
git merge feature/nova-funcionalidade

# Deletar branch
git branch -d feature/nova-funcionalidade
```

## Acessar Serviços

### URLs

| Serviço | URL | Credenciais |
|---------|-----|-------------|
| MLflow UI | http://localhost:5010 | - |
| CloudServer (S3 API) | http://localhost:9000 | user / password |
| Postgres (MLflow) | localhost:5433 | user / password |
| Postgres (MLOps) | localhost:5434 | mlops_user / admin |

### Testar Conexões

```bash
# Testar MLflow
curl http://localhost:5010/health

# Testar CloudServer (retorna 200 se saudável)
curl -s -o /dev/null -w "%{http_code}" http://localhost:9000/_/healthcheck

# Testar Postgres (MLflow)
psql -h localhost -p 5433 -U user -d db

# Testar Postgres (MLOps)
psql -h localhost -p 5434 -U mlops_user -d mlops_db
```

### MLflow não abre

```bash
docker compose restart mlflow-server
docker compose logs mlflow-server
```

### Listar arquivos no CloudServer

O CloudServer não tem interface web. Para navegar nos buckets, entre no container em modo interativo:

```bash
# Entrar no container com AWS CLI
docker compose run --rm \
  -e AWS_ACCESS_KEY_ID=user \
  -e AWS_SECRET_ACCESS_KEY=password \
  -e AWS_DEFAULT_REGION=us-east-1 \
  --entrypoint sh \
  cloudserver-create-bucket
```

Dentro do container, use os comandos abaixo:

```bash
# Listar buckets
aws --endpoint-url http://cloudserver:8000 s3 ls

# Listar conteúdo do bucket de artefatos
aws --endpoint-url http://cloudserver:8000 s3 ls s3://bucket

# Listar recursivamente (todos os artefatos)
aws --endpoint-url http://cloudserver:8000 s3 ls s3://bucket --recursive

# Sair do container
exit
```

