import numpy as np
from pandas import DataFrame
import networkx as nx
import matplotlib.pyplot as plt
import pylab


class Map:
    def __init__(self, full_connections, colour_map,pos=None, state_labels =None):
        self.full_connections = full_connections
        split = [[list(l) for l in zip(*temp)] for temp in full_connections]
        connections= []
        directions= []
        for temp in split:
            if temp == []:
                connections.append([])
                directions.append([])
                continue
            connections.append(temp[1])
            directions.append(temp[0])
        self.connections = np.array(connections)
        self.direction_map = np.array(directions)
        self.colour_map = np.array(colour_map)
        self.num_states = len(colour_map)
        self.pos = pos
        self.G = nx.Graph()
        if state_labels == None:
            self.state_labels = range(0, len(colour_map))
        else:
            self.state_labels = state_labels

    def find_connections(self, node):
        return self.connections[node]

    def is_connected(self, n1, n2, bilateral=False):
        b = n2 in self.connections[n1]
        if bilateral:
            b = n2 in self.connections[n1] or n1 in self.connections[n2]
        return b

    def has_path(self, n1, n2):
        try:
            b= nx.has_path(self.G, n1, n2)
        except:
            b = False
        return b

    def colour_of(self, node):
        return self.colour_map[node]

    def init_G(self):
        edges = []
        for i in range(0, np.size(self.connections, 0)):
            for j in range(0, np.size(self.connections[i], 0)):
                edges.append((i, self.connections[i][j]))
        self.G.add_edges_from(edges)


    def show(self,node_weights=None, actual_state=None, orientation=None, delay=0, title ='Map', show=1, save=0, save_title='Default', fig = None, figsize=(7.5,6), ax=None):
        plt.ion()
        if ax is not None:
            ax.cla()
        if node_weights is None:
            node_weights = np.ones(self.num_states)
        edges = []
        for i in range(0, np.size(self.connections, 0)):
            for j in range(0, np.size(self.connections[i], 0)):
                edges.append((i, self.connections[i][j]))
        G = self.G
        G.add_edges_from(edges)
        if self.pos ==None:
            pos = self.get_pos()
            self.pos = pos
        else:
            pos = self.pos

        for i in range(0, len(self.colour_map)):
            if i != actual_state or actual_state is None:
                nx.draw_networkx_nodes(G, pos, nodelist=[i], node_color=[self.colour_map[i]],node_size= node_weights[i]*1000, ax=ax)
                nx.draw_networkx_labels(G, pos, labels={i : self.state_labels[i]}, ax=ax)
            else:
                if orientation is None:
                    nx.draw_networkx_nodes(G, pos, nodelist=[i], node_color=[self.colour_map[i]],
                                           node_size=node_weights[i] * 1000, node_shape='x', ax=ax)
                    nx.draw_networkx_labels(G, pos, labels={i: 'Current state'}, ax=ax)
                else:
                    markers = np.array(['<','^','>','v'])
                    nx.draw_networkx_nodes(G, pos, nodelist=[i], node_color=[self.colour_map[i]],
                                           node_size=node_weights[i] * 1000, node_shape=markers[orientation], ax=ax)
                    nx.draw_networkx_labels(G, pos, labels={i: 'Current state'}, ax=ax)

        nx.draw_networkx_edges(G, pos, ax=ax)
        # print edges
        # nx.draw(G)
        # nx.draw_networkx_labels(G,labels=range(1, len(self.colour_map)))
        if ax is not None:
            ax.set_title(title)
        if save:
            plt.savefig('Media/'+save_title)
            if fig is not None:
                fig.set_size_inches(figsize[0], figsize[1])
        if show:
            pylab.draw()
            if fig is not None:
                fig.set_size_inches(figsize[0], figsize[1])
            plt.pause(delay)



    def get_transition_model(self, noise =0.2, noise_type=0):
        num_states = len(self.colour_map)
        model = {
            'N': np.zeros([num_states, num_states]),
            'S': np.zeros([num_states, num_states]),
            'W': np.zeros([num_states, num_states]),
            'E': np.zeros([num_states, num_states]),
        }
        for i, connections in enumerate(self.connections):
            for j, connection in enumerate(connections):
                    card = self.direction_map[i][j]
                    model[card][i, connection] = 1

                    flipped_card = self.flip_cardinal(card)
                    model[flipped_card][connection, i] = 1

        for k, df in model.items():
            mat = model[k]

            for i in range(num_states):
                if np.all(mat[i,:] == 0):
                    mat[i, i] = 1
            tmp2 =[]
            for row in mat.tolist():
                tmp = []
                for i, col in enumerate(row):
                    if noise_type==0:
                        if col==1:
                            tmp.append(col-noise)
                        elif self.is_connected( np.argmax(row), i, True):
                            tmp.append(noise / 3)
                        else:
                            tmp.append(0)
                    elif noise_type==1:
                        if col == 1:
                            tmp.append(col - noise)
                        else:
                            tmp.append(noise/(num_states-1))
                tmp2.append(tmp)
            mat = tmp2
            #mat = [[col-noise if col == 1 else  if ) else 0) for col in row] for row in mat.tolist()]

            for i in range(num_states):
                mat[i] = mat[i]/np.sum(mat[i])

            model[k] = mat
        f = lambda x: model[x]
        #print model
        return f

    def get_sensor_readings(self, position, orientation):
        readings = np.zeros(4)
        for connection in self.full_connections[position]:
            if connection[0] == 'N':
                readings[1] = 1
            if connection[0] == 'E':
                readings[2] = 1
        try:
            for connection in self.full_connections[position-1]:
                if connection[1] == position and connection[0] == 'E':
                    readings[0] = 1
        except:
            print(position-1)
        try:
            for i in range(position):
                for connection in self.full_connections[i]:
                    if connection[1] == position and connection[0] == 'N':
                        readings[3] = 1
                        break
        except:
            print(self.num_states)

        return np.array(readings[[orientation-1, orientation, (orientation+1) % 4]])

    def get_sensor_model(self, noise=0.2):
        num_states = len(self.colour_map)
        f = lambda x : [val-noise if val == 1 else noise/num_states for val in np.array(self.colour_map) == x]
        return f


    @staticmethod
    def flip_cardinal(card):
        d = dict([('N', 'S'), ('S', 'N'), ('W', 'E'), ('E', 'W')])
        return d[card]

    def get_pos(self):
        pos = {1: np.array([5, 5])}
        for i in range(0, np.size(self.connections, 0)):
            for j in range(0, len(self.connections[i])):
                x = 0
                y = 0
                ind = self.connections[i][j]
                card = self.direction_map[i][j]
                if card == 'N':
                    x, y = 0, 1
                elif card == 'S':
                    x, y = 0, -1
                elif card == 'E':
                    x, y = 1, 0
                else:
                    x, y = -1, 0
                pos[ind] = np.array([pos[i + 1][0] + x, pos[i + 1][1] + y])
        return pos

    @staticmethod
    def random_grid_map(num_colours, length, random_state=None):
        full_connections, colour_map, pos = Map.random_grid_connections(num_colours, length, random_state=random_state)
        return Map(full_connections, colour_map, pos)

    @staticmethod
    def random_grid_connections(num_colours, length=10, random_state=None):
        #length = max(length, 2*np.floor(np.sqrt(num_nodes)))
        full_connections = []
        pos = []
        for j in range(0, length):
            for i in range(0, length):
                st = set()
                if random_state is None:
                    cardinals = np.random.choice(['N', 'E', 'D'], 2, replace=False)
                else:
                    cardinals = random_state.choice(['N', 'E', 'D'], 2, replace=False)
                for card in cardinals:
                    if card == 'N' and j < length-1:
                        st.add((card, (j+1)*length+i))
                    elif card == 'E' and i < length-1:
                        st.add((card, j*length+i+1))
                full_connections.append(list(st))
                pos.append((i, j))
        if random_state is None:
            colour_map = np.random.choice(['Y', 'B', 'R', 'G', 'O'][0:num_colours], length**2,replace=True)
        else:
            colour_map = random_state.choice(['Y', 'B', 'R', 'G', 'O'][0:num_colours], length**2,replace=True)


        return full_connections, colour_map, pos

    @staticmethod
    def random_connections(num_nodes, num_colours):
        full_connections = []
        for i in range(1,num_nodes ):
            i_connections = []
            for j in range(i, num_nodes):
                cardinal = np.random.choice(['N', 'S', 'E', 'W', 'D'])
                if cardinal == 'D':
                    continue
                i_connections.append((j+1,cardinal))
            full_connections.append(i_connections)
        colour_map = np.random.choice(['Y', 'B', 'R', 'G', 'O'][0:num_colours],num_nodes,replace=True)
        return full_connections, colour_map



def test():
    # full_connections = [[(2, 'N'), (3, 'S')], [(1, 'S')]]
    # colour_map = ['B', 'R', 'R']
    full_connections, colour_map, pos = Map.random_grid_connections(3,6)

    #print full_connections
    #print colour_map
    #print pos
    map = Map(full_connections, colour_map, pos = pos)
    #print map.connections
    #print map.direction_map
    f = map.get_transition_model()
    g = map.get_sensor_model()
    print(f('S'))
    print(g('R'))
    #map.show()
    #print nx.has_path(map.G, 0, 35)


if __name__ == '__main__':
    test()
