Despliegue de un Modelo de Deep Learning en AKS
Clasificación de Imágenes (CIFAR-10) con Flask, Docker y Kubernetes
Autor: Stevens Bohorquez Ruiz

Descripción
Guía paso a paso para implementar, escalar y administrar un modelo de red neuronal convolucional (CNN) en Azure Kubernetes Service (AKS), incluyendo:

Entrenamiento del modelo con el dataset CIFAR-10.

Despliegue de una API de predicción con Flask.

Contenerización con Docker y publicación en Docker Hub.

Configuración de un clúster Kubernetes en Azure.

Exposición del servicio mediante un LoadBalancer.

Requisitos Previos
✅ Cuenta en Azure y Docker Hub.
✅ Azure CLI y kubectl instalados localmente.
✅ Conocimientos básicos de Kubernetes, Docker y Python.

Pasos de Implementación
1. Configuración del Clúster AKS en Azure
Crear un grupo de recursos y un clúster AKS con 2 nodos desde el portal de Azure.

Verificar el clúster:

bash
az aks get-credentials --resource-group gr-kubernetes-proyecto2 --name aks-proyecto2
kubectl get nodes
2. Entrenamiento del Modelo (CIFAR-10)
Descargar el dataset desde CIFAR-10.

Entrenar el modelo CNN con train_model.py (guardado como cifar10_model.params en /model).

Probar las predicciones localmente con Flask (app.py) y Postman.

3. Contenerización con Docker
Crear Dockerfile y requirements.txt.

Construir y ejecutar la imagen localmente:

bash
docker build -t cifar10-api:latest .
docker run -p 5000:5000 cifar10-api:latest
4. Publicación en Docker Hub
Subir la imagen a Docker Hub:

bash
docker tag cifar10-api <tu-usuario>/cifar10-api
docker push <tu-usuario>/cifar10-api
5. Despliegue en Kubernetes
Aplicar el manifiesto deployment.yaml:

bash
kubectl apply -f deployment.yaml
Exponer el servicio con un LoadBalancer:

bash
kubectl expose deployment kubermatic-dl-deployment --type=LoadBalancer --port 80 --target-port 5000
kubectl get service  # Obtener la IP pública
6. Pruebas y Monitoreo
Acceder a la API desde: http://<IP-PÚBLICA>/predict (ejemplo en vídeo incluido).

Monitorear métricas en el dashboard de AKS.

Estructura del Proyecto
.
├── model/                  # Modelo entrenado (cifar10_model.params)
├── app.py                  # API Flask
├── train_model.py          # Script de entrenamiento
├── requirements.txt        # Dependencias
├── Dockerfile              # Configuración de Docker
└── deployment.yaml         # Configuración de Kubernetes
Notas Importantes
📊 Se incluyen capturas de las métricas de AKS en el repositorio y un video ilustrativo. 