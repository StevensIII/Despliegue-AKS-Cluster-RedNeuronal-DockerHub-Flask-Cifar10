import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn
import os

def train_cifar10_model():
    # Definir el contexto (GPU si está disponible, si no CPU)
    ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()

    # Descargar el conjunto de datos CIFAR-10
    transform = gluon.data.vision.transforms.ToTensor()
    train_dataset = gluon.data.vision.CIFAR10(train=True).transform_first(transform)
    test_dataset = gluon.data.vision.CIFAR10(train=False).transform_first(transform)

    train_data = gluon.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_data = gluon.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Definir una red neuronal simple
    net = nn.Sequential()
    net.add(
        nn.Dense(256, activation="relu"),
        nn.Dense(128, activation="relu"),
        nn.Dense(10)
    )
    net.initialize(ctx=ctx)

    # Definir la función de pérdida y el optimizador
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam')

    # Entrenar el modelo
    epochs = 10
    for epoch in range(epochs):
        cumulative_loss = 0
        for data, label in train_data:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = loss_fn(output, label)
            loss.backward()
            trainer.step(batch_size=data.shape[0])
            cumulative_loss += loss.sum().asscalar()
        print(f"Epoch {epoch+1}, loss: {cumulative_loss/len(train_dataset)}")

    # Crear carpeta 'model' si no existe
    if not os.path.exists('model'):
        os.makedirs('model')

    # Guardar los parámetros entrenados
    net.save_parameters('model/cifar10_model.params')
    # net.export("cifar10_model")  

if __name__ == "__main__":
    train_cifar10_model()