# **Big Data, Grafos y Machine Learning**  

Este repositorio está dedicado a explorar la **Teoría de Grafos y sus aplicaciones en Big Data e Inteligencia Artificial**. Contiene recursos teóricos, implementaciones prácticas y proyectos que demuestran cómo los **grafos** pueden utilizarse para resolver problemas complejos en **análisis de redes, optimización de algoritmos y modelado de estructuras de datos**.  

Los **grafos** son fundamentales en diversos ámbitos como:  

- **Redes sociales:** detección de comunidades, análisis de influencia (ejemplo: TikTok, Twitter).  
- **Recomendaciones de productos:** sistemas de recomendación (Netflix, Amazon).  
- **Ciberseguridad y detección de fraudes:** análisis de transacciones bancarias.  
- **Optimización de rutas:** transporte y logística.  
- **Machine Learning con Grafos:** Graph Neural Networks (GNNs) aplicadas a predicción de enlaces, clasificación de nodos y detección de anomalías.  

---

## **Contenido**  

- [Clases de Big Data](#clases-de-big-data)  
- [Teoría de Grafos](#teoría-de-grafos)  
- [Machine Learning con Grafos](#machine-learning-con-grafos)  
- [Aplicaciones](#aplicaciones)  
- [Ejemplos Visuales](#ejemplos-visuales)  
- [Recursos Adicionales](#recursos-adicionales)  
- [Uso del Repositorio](#uso-del-repositorio)  
- [Colaboración](#colaboración)  
- [Licencia](#licencia)  

---

## **Clases de Big Data**  

- `BIG_DATA_Clase_Cuarta.ipynb`: Introducción al uso de la teoría de grafos en redes sociales, con ejemplos prácticos sobre **detección de comunidades**.  
- `BIG_DATA_Digrafos_y_Redes_Clase.ipynb`: **Dígrafos** y análisis de redes orientadas en el contexto de Big Data.  

Ejemplo: **Detección de comunidades en redes sociales con NetworkX**  

```python
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community

G = nx.karate_club_graph()
communities = community.greedy_modularity_communities(G)

colors = []
for node in G.nodes():
    for i, comm in enumerate(communities):
        if node in comm:
            colors.append(i)

nx.draw(G, with_labels=True, node_color=colors, cmap=plt.cm.rainbow)
plt.show()
```

---

## **Teoría de Grafos**  

- `Graph_Convolution_Network.ipynb`: Implementación y estudio de **Graph Convolutional Networks (GCNs)**.  
- `Hipergrafo_y_ciclo_hamiltoniano.ipynb`: Implementación de **ciclo hamiltoniano en un hipergrafo**.  

Ejemplo: **Ciclo Hamiltoniano en un grafo completo**  

```python
import networkx as nx

G = nx.complete_graph(5)
hamiltonian_cycle = list(nx.find_cycle(G, orientation="ignore"))

print("Ciclo Hamiltoniano:", hamiltonian_cycle)
```

---

## **Machine Learning con Grafos**  

- `Prediccion_de_Enlaces.ipynb`: **Predicción de enlaces** en redes sociales usando aprendizaje automático.  
- `Graph_Convolution_Network.ipynb`: Implementación de **Graph Neural Networks (GNNs) con PyTorch Geometric**.  

Ejemplo: **Graph Convolutional Network (GCN) con PyTorch Geometric**  

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[1], [2], [3]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCN()
output = model(data)
print(output)
```

---

## **Aplicaciones**  

- `Revolucion_de_la_Credibilidad.ipynb`: Análisis basado en grafos aplicado a **credibilidad en redes sociales**.  
- `Wheel_graphs.ipynb`: Estudio sobre los **grafos tipo rueda** y sus propiedades.  
- `Proyecto_TikTok_GNN.ipynb`: **Modelo de recomendación de contenido en TikTok** con GNNs.  
- `Fraude_Financiero_Grafos.ipynb`: **Detección de fraude financiero** usando teoría de grafos.  
- `Optimización_Rutas.ipynb`: Algoritmos como **Dijkstra y A*** aplicados a logística y transporte.  

Ejemplo: **Modelo de recomendación en TikTok con GNNs**  

```python
import torch
from torch_geometric.nn import GraphConv
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)  
x = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float)  

data = Data(x=x, edge_index=edge_index)

class TikTokGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(2, 16)
        self.conv2 = GraphConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = TikTokGNN()
output = model(data)
print(output)
```

---

## **Ejemplos Visuales**  

### **Animación de Dodecaedro y BFS**  

Esta animación muestra el recorrido en un **dodecaedro** usando **BFS (Búsqueda en Anchura)**.  

![Animación de Dodecaedro y BFS](dfs_dodecaedro_rotacion_lenta.gif)  

---

## **Recursos Adicionales**  

- [Introduction to Graph Theory - Robin J. Wilson](https://link)  
- [Graph Neural Networks: A Review](https://arxiv.org/abs/1901.00596)  
- [NetworkX Documentation](https://networkx.org/)  

---

## **Uso del Repositorio**  

Para clonar el repositorio y ejecutar los notebooks:  

```bash
git clone https://github.com/sgevatschnaider/Grafos.git
cd Grafos
jupyter notebook
```

Abre el notebook de tu interés y ejecuta los ejemplos interactivos.  

---

## **Colaboración**  

¡Las contribuciones son bienvenidas! Sigue estos pasos:  

1. Haz un **fork** del repositorio.  
2. Crea una nueva rama:  
   ```bash
   git checkout -b feature/nueva-caracteristica
   ```  
3. Realiza tus cambios y súbelos:  
   ```bash
   git commit -m 'Añadir nueva característica'
   git push origin feature/nueva-caracteristica
   ```  
4. Abre un **Pull Request** para revisión.  

---

## **Licencia**  

Este proyecto está bajo la licencia **MIT**. Consulta el archivo LICENSE para más detalles.  

