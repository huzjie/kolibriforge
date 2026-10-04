# 部署

## Docker

```bash
docker build -f docker/Dockerfile -t kolibriforge .
docker run -p 8000:8000 kolibriforge
```

## Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
```

## Helm

```bash
helm install kolibriforge helm/kolibriforge
```

## OpenAI 兼容 API

服务暴露 `/health`、`/v1/models`、`/v1/chat/completions`、`/v1/completions`。
