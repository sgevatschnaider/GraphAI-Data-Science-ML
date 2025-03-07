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

- [Introduction to Graph Theory - Robin J. Wilson](https://www.maths.ed.ac.uk/~v1ranick/papers/wilsongraph.pdf)
- [Introductory Graph Theory  por Jacques Verstraete](https://cseweb.ucsd.edu/~dakane/Math154/154-textbook.pdf)  
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

# Notebooks Interactivos  
Puedes abrir y ejecutar los notebooks en Google Colab directamente desde aquí:  

---

## 📊 **Tabla de Contenidos**  


| 📄 **Recurso** | 📥 **Acceso** |
|--------------|------------|
| **Introducción al BIG DATA** | [![📄 Descargar PDF](https://img.shields.io/badge/📄%20Descargar-Introducción%20al%20BIG%20DATA-red?style=for-the-badge)](https://github.com/sgevatschnaider/GraphAI-Data-Science-ML/blob/ba97ebadab45f05b0b5b3b4b5ca2fc7156a24691/BIG%20DATA%20INTRODUCCIÓN.pdf) |
| **Cómo TikTok Sabe lo que Quieres Ver** | [![📖 Leer en GitHub Pages](https://img.shields.io/badge/📖%20Leer%20en-GitHub%20Pages-blue?style=for-the-badge&logo=github)](https://sgevatschnaider.github.io/GraphAI-Data-Science-ML/blog/tiktok-algoritmo.html) |
| **Introducción a Big Data en Google Colab** | [![🚀 Abrir en Colab](https://img.shields.io/badge/🚀%20Abrir%20en-Google%20Colab-orange?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/github/sgevatschnaider/GraphAI-Data-Science-ML/blob/main/notebooks/Clase_Introduccion_BigData_2025.ipynb) |
| **Sistemas de Recomendación y TikTok** | [![🔍 Abrir en Colab](https://img.shields.io/badge/🔍%20Abrir%20en-Google%20Colab-orange?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/drive/1eqcIUhjwrKRj4_4rFv_tg7vRYkxkjuUE) |
| **Evolución y Funcionamiento de los Sistemas de Recomendación** | [![📄 Descargar PDF](https://img.shields.io/badge/📄%20Descargar-Evolución%20y%20Funcionamiento%20de%20los%20Sistemas%20de%20Recomendación-red?style=for-the-badge)](https://github.com/sgevatschnaider/GraphAI-Data-Science-ML/blob/35919d0c04d0f2e447590877c33420003bfcfcc6/Evolución%20y%20Funcionamiento%20de%20los%20Sistemas%20de%20Recomendación.pdf) |
|| **Preguntas Big Data** | [![📄 Descargar PDF](https://img.shields.io/badge/📄%20Descargar-Preguntas%20Big%20Data-red?style=for-the-badge)](https://github.com/sgevatschnaider/GraphAI-Data-Science-ML/blob/9b93a3f622c3d4a544fe593d8ede12f4f1de2f14/Preguntas_Big_Data.pdf) |
| **Sistemas de Recomendación (Nuevo)** | [![📄 Descargar PDF](https://img.shields.io/badge/📄%20Descargar-Sistemas%20de%20Recomendación-red?style=for-the-badge)](https://github.com/sgevatschnaider/GraphAI-Data-Science-ML/blob/faec436535519a2b9d44cc4482021e107f59191e/Sistemas_de_Recomendacion.pdf) |

---

---
## Acceso al Material Educativo

**Este repositorio es solo de lectura para los estudiantes.**  
Los alumnos pueden **descargar y utilizar** el material, pero **no pueden editarlo**.  
Si tienes dudas o sugerencias, abre un **Issue** en GitHub. 

**Para descargar los archivos, usa los enlaces disponibles en la sección de contenidos.**



## **Licencia**  

Este proyecto está bajo la licencia **MIT**. Consulta el archivo LICENSE para más detalles.  

