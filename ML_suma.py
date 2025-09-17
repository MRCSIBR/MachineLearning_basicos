# Aprendizaje Automático para Sumar Dos Números
# Este script demuestra cómo usar una red neuronal para aprender la suma de dos números.
# Usaremos PyTorch para construir y entrenar el modelo, generar datos sintéticos y visualizar los resultados.

# Importar Bibliotecas
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Establecer una semilla aleatoria para la reproducibilidad
torch.manual_seed(42)
np.random.seed(42)

# Paso 1: Generar Datos Sintéticos
# Crear pares de números (a, b) y sus sumas (a + b) con un ligero ruido.
# Generar 1000 pares de números entre -10 y 10.
n_samples = 1000
X = np.random.uniform(-10, 10, (n_samples, 2))  # Dos números por muestra
y = X[:, 0] + X[:, 1] + np.random.normal(0, 0.1, n_samples)  # Suma con ruido

# Convertir a tensores de PyTorch
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).reshape(-1, 1)

# Visualizar un subconjunto de los datos
plt.figure(figsize=(6, 4))
plt.scatter(X[:100, 0], X[:100, 1], c=y[:100], cmap='viridis')
plt.colorbar(label='Suma (a + b)')
plt.xlabel('Número a')
plt.ylabel('Número b')
plt.title('Datos de Muestra: Dos Números y Su Suma')
plt.savefig('data_visualization.png')
plt.close()

# Paso 2: Definir la Red Neuronal
# Una red neuronal simple de alimentación hacia adelante (feedforward) con una capa oculta:
# - Capa de entrada: 2 neuronas (para los dos números)
# - Capa oculta: 10 neuronas con activación ReLU
# - Capa de salida: 1 neurona (para la suma)
class SumNet(nn.Module):
    def __init__(self):
        super(SumNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.layers(x)

model = SumNet()

# Paso 3: Configurar la Función de Pérdida y el Optimizador
# - Pérdida: Error Cuadrático Medio (MSE) para regresión
# - Optimizador: Descenso de Gradiente Estocástico (SGD)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Paso 4: Entrenar el Modelo
# Entrenar durante 1000 épocas, imprimiendo la pérdida cada 100 épocas
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()  # Limpiar los gradientes anteriores
    outputs = model(X_tensor)  # Pase hacia adelante (forward pass)
    loss = criterion(outputs, y_tensor)  # Calcular la pérdida
    loss.backward()  # Pase hacia atrás (backward pass)
    optimizer.step()  # Actualizar los pesos

    if (epoch + 1) % 100 == 0:
        print(f'Época [{epoch+1}/{epochs}], Pérdida: {loss.item():.4f}')

# Paso 5: Evaluar y Visualizar los Resultados
# Calcular predicciones y graficar las sumas reales vs. las predichas
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor).numpy()
y_true = y_tensor.numpy()

# Graficar sumas reales vs. predichas
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, alpha=0.5, color='blue')
plt.plot([-20, 20], [-20, 20], 'r--', label='Predicción perfecta (y=x)')
plt.xlabel('Suma Real')
plt.ylabel('Suma Predicha')
plt.title('Sumas Reales vs. Sumas Predichas')
plt.legend()
plt.grid(True)
plt.savefig('sum_predictions.png')
plt.close()

# Probar algunos ejemplos
test_inputs = torch.FloatTensor([[1, 2], [5, -3], [-4, 7]])
with torch.no_grad():
    test_preds = model(test_inputs).numpy()
for i, (a, b) in enumerate(test_inputs):
    print(f'Entrada: {a:.1f} + {b:.1f}, Suma Predicha: {test_preds[i][0]:.2f}, Suma Real: {a+b:.2f}')

# Puntos Clave:
# - Las redes neuronales pueden aprender funciones simples como la suma.
# - Preparación de datos: Convertir entradas/salidas a tensores.
# - Modelo: Una red pequeña con una capa oculta es suficiente.
# - Pérdida/Optimización: MSE y SGD son estándar para la regresión.
# - Visualización: Graficar los valores reales vs. los predichos evalúa el rendimiento.
