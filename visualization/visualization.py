import sqlite3
import pandas as pd
from pyvis.network import Network
import numpy as np


def get_graph(df):
    gr = np.zeros((best_n_clusters, best_n_clusters), dtype=int)

    last_person_id: int = None
    prev_cluster: int = None

    for row in df.itertuples():
        person_id = row.person_id
        cluster = row.cluster

        if person_id == last_person_id:
            gr[prev_cluster][cluster] += 1

        last_person_id = person_id
        prev_cluster = cluster

    gr_max = np.max(gr)
    for i in range(-3, best_n_clusters):
        for j in range(-3, best_n_clusters):
            gr[i][j] = gr_max - gr[i][j] + 1

    np.max(gr)

    return gr


conn = sqlite3.connect('../db/normalized_data.db')
cursor = conn.cursor()
df = pd.read_sql("SELECT * FROM clusters", conn)
conn.close()

df2 = df[['person_id', 'cluster']]
best_n_clusters = df2['cluster'].nunique()

# Предположим, это ваша матрица смежности
size = best_n_clusters  # или количество узлов в вашем графе
matrix = get_graph(df2)

# Инициализация сети
net = Network(height="100%", width="100%", bgcolor="#222222", font_color="white")

# Добавление узлов и ребер
for i, row in enumerate(matrix):
    for j, val in enumerate(row):
        if i != j and val != 0:  # Исключаем петли и ненужные ребра
            net.add_node(int(i), label=str(i), title=str(i))
            net.add_node(int(j), label=str(j), title=str(j))
            net.add_edge(int(i), int(j), value=int(val))
            print(i, j)
# Устанавливаем физику для лучшего размещения
net.set_options("""
var options = {
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -100,
      "centralGravity": 0.005,
      "springLength": 100,
      "springConstant": 0.18
    },
    "maxVelocity": 146,
    "solver": "forceAtlas2Based",
    "timestep": 0.35,
    "stabilization": {"iterations": 150}
  }
}
""")

# Сохраняем и отображаем граф
net.show("graph.html")
