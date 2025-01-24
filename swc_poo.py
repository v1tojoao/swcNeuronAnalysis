import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  # Para colormap
from matplotlib.colors import Normalize  # Para normalização de valores

class Node:
    def __init__(self,index,x,y,z,r,parent=None,type=None):#son=None,
        #informacoes do nó do grafo swc padrão
        self.id = index
        self.xyz = np.array([x,y,z])
        self.r = r
        self.parent = parent
        self.type = type

        #informacoes adicionais
        self.distancia=0 #distancia até o nó "parent"
        self.angulo=0    #angulo com a reta formada pelos 2 ultimos nós "parent"
        #self.son = son

    def __repr__(self):
        return f"No(index='{self.id}'\t xyz='{self.xyz}'\t r={self.r}\t parent='{self.parent}'\t type={self.type}\t distancia={self.distancia}\t angulo={self.angulo})\n"#son={self.son}\t 

    def calc_distancia(self,parent):
        self.distancia = np.linalg.norm(self.xyz - parent.xyz)
        
        
    def calc_angulo(self,parent,grand):
        v_reta = (parent.xyz - grand.xyz)*10000
        v_self = (self.xyz - parent.xyz)*10000
        
        norm_reta = np.linalg.norm(v_reta)#+0.00001
        norm_self = np.linalg.norm(v_self)#+0.00001
        if norm_reta > 0 and norm_self > 0:
            cos_theta = np.dot(v_reta,v_self)/(norm_reta*norm_self)
            theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Usar np.clip para evitar erro devido a imprecisão numérica
            self.angulo = np.degrees(theta_rad)
        else:
            print(v_reta,v_self,parent.xyz,grand.xyz,self.xyz)
            self.angulo = 0
        

class Metadata:
    def __init__(self,swcName,imgName,classeOrg,respTracado,bioInfo=None,imgConfig="lente=63x; dias=90; marcadores=DAPI,MAP2",pxSize_xy=None,pxSize_z=None):
        self.imgName = imgName          # Nome da imagem
        self.swcName = swcName          # Nome do arquivo SWC
        self.classeOrg = classeOrg      # Classe do organoide (Fd,Fv,Td,Tv,AF,AT,A4).
        self.respTracado = respTracado  # Responsável pelo traçado
        self.bioInfo = bioInfo          # Informações biológicas
        self.imgConfig = imgConfig      # Configurações da imagem
        self.pxSize_xy = pxSize_xy      # Tamanho do pixel no plano XY
        self.pxSize_z = pxSize_z        # Tamanho do pixel no plano Z

    def __repr__(self):
        return (
            f"Metadata(imgName='{self.imgName}'\n classeOrg='{self.classeOrg}'\n bioInfo={self.bioInfo}\n respTracado='{self.respTracado}'\n imgConfig={self.imgConfig}\n pxSize_xy={self.pxSize_xy}\n pxSize_z={self.pxSize_z})"
        )
    

class Neuron:
    def __init__(self, metadata:Metadata=None):
        self.metadata=metadata
        self.G = None
        if metadata!= None:
            self.read_swc(metadata.swcName)
        pass

    def searchNodes(self):
        list = [self.rootNode]

    def getParent(self, no:Node):
        i = no.parent
        if i != -1:
            return self.G.nodes[i]['dados']
        else:
            print("Nó raiz nao tem parent")
            return None
    
    def read_swc(self, swcpath,classe =None):
        self.G = nx.Graph()
        df = pd_swc(swcpath,classe)
        for index, row in df.iterrows():
            aux_no = Node(index=index,x=row['x'],y=row['y'],z=row['z'],r=row['radius'],parent=row['parent'],type=row['type'])
            self.G.add_node(aux_no.id,dados=aux_no)
            if aux_no.parent != -1:
                self.G.add_edge(aux_no.parent,aux_no.id)
                aux_parent = self.getParent(aux_no)
                self.G.nodes[aux_no.id]["dados"].calc_distancia(aux_parent)
                if aux_parent.parent != -1:
                    aux_grand = self.getParent(aux_parent)
                    self.G.nodes[aux_no.id]["dados"].calc_angulo(aux_parent,aux_grand)

    def neuron_view(self):
        pos = {self.G.nodes[node]["dados"].id: self.G.nodes[node]["dados"] for node in self.G.nodes()}

        # Criar o gráfico 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plotando as arestas
        for (u, v) in self.G.edges():
            x_vals = [pos[u].xyz[0], pos[v].xyz[0]]
            y_vals = [pos[u].xyz[1], pos[v].xyz[1]]
            z_vals = [pos[u].xyz[2], pos[v].xyz[2]]
            ax.plot(x_vals, y_vals, z_vals, color='b', alpha=0.6)  # Ajuste de cor e transparência
        
        # Plotando os nós como bolas
        norm = Normalize(vmin=0, vmax=180)  # Normalizar entre o menor e o maior raio
        cmap = cm.Reds 
        for n, node in pos.items():
            x,y,z = node.xyz
            raio_normalizado = norm(node.angulo)  # Normalizar o raio para o intervalo [0, 1]
            cor = cmap(raio_normalizado)
            ax.scatter(x, y, z, s=node.r*3, color=cor)#, label=f'Node {n}')  # Aumente o valor de `s` para mudar o tamanho da bola
            #ax.text(x, y, z, f'{n}', color='red', fontsize=12)  # Adicionando rótulos aos nós
        
        # Ajustes finais de visualização
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Visualização 3D do Grafo")
        
        # Mostrar a visualização
        plt.show()
    
def pd_swc(swcpath,classe=None):
    df = pd.read_csv(swcpath,index_col='id',sep='\s+', comment='#', header=None,
                 names=["id", "type", "x", "y", "z", "radius", "parent"])
    return df
    






























#==============================================================================================================
'''def neuron_view(self):
        # Criar o gráfico 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plotando as arestas
        for (u, v) in G.edges():
            x_vals = [pos[u][0], pos[v][0]]
            y_vals = [pos[u][1], pos[v][1]]
            z_vals = [pos[u][2], pos[v][2]]
            ax.plot(x_vals, y_vals, z_vals, color='b', alpha=0.6)  # Ajuste de cor e transparência
        
        # Plotando os nós como bolas
        for node, (x, y, z) in pos.items():
            ax.scatter(x, y, z, s=100, label=f'Node {node}')  # Aumente o valor de `s` para mudar o tamanho da bola
            ax.text(x, y, z, f'{node}', color='red', fontsize=12)  # Adicionando rótulos aos nós
        
        # Ajustes finais de visualização
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Visualização 3D do Grafo")
        
        # Mostrar a visualização
        plt.show()'''