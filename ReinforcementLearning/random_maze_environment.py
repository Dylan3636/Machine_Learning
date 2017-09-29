from tools.Map import Map
import matplotlib.pyplot as plt
import numpy as np

MOVEMENTS = ['LEFT_TURN', 'FORWARD', 'RIGHT_TURN']
CARDINALS = ['N', 'E', 'W', 'S']

class random_maze:
    def __init__(self, length, num_colours, action_type=0, ax=None, noise=0, noise_type=0, init_position=None,init_orientation=None, end_position=None, auto_randomize=0.2, debug=0, save_data=0):
        self.length = length
        self.num_colours = num_colours
        self.action_type = action_type
        self.actions = [MOVEMENTS, CARDINALS][action_type]
        self.num_actions = len(self.actions)
        self.randomize = auto_randomize
        self.ax = plt.subplot(111) if ax is None else ax
        self.maze = Map.random_grid_map(length=length, num_colours=num_colours)
        self.transition_model = self.maze.get_transition_model(noise=noise, noise_type=noise_type)
        self.start_position = 0
        self.end_position = self.maze.num_states-1 if end_position is None else end_position
        self.prev_position = init_position
        self.prev_orientation = init_orientation
        self.prev_action = 'Start'
        self.graph_data = [[],[]]
        self.counter = 0
        self.current_return = 0
        self.save_data = save_data
        self.debug = debug

    def step(self, action):
        if self.action_type == 0:
            self.counter += 1
            if hasattr(action, '__int__'):
                action = self.actions[int(action)]

            prev_position = self.prev_position.copy()
            prev_orientation = self.prev_orientation.copy()

            if action == 'FORWARD':
                card_action = get_cardinal(prev_orientation)
                position = int(np.random.choice(self.maze.num_states, p=self.transition_model(card_action)[np.argmax(prev_position)]))
                current_position = self.position_encoder(position)
                current_orientation = prev_orientation.copy()
            else:
                current_orientation = self.get_next_orientation(action, prev_orientation)
                current_position = prev_position.copy()

            current_readings = self.maze.get_sensor_readings(np.argmax(current_position), orientation=np.argmax(current_orientation))

            if np.argmax(current_position) == self.end_position:
                reward = 1
                done = 1
                success = 1
            elif self.counter == 3*self.maze.num_states:
                reward = -1
                done = 1
                success = 0
            elif np.argmax(prev_position) == np.argmax(current_position) and (action == 'FORWARD'):
                reward = -0.2
                done = 0
                success=2

            elif action == 'FORWARD':
                pos_ind = np.argmax(current_position)
                length = self.length
                reward = -float(length-(pos_ind % length) + length-(pos_ind/length))/(20*length)
                done = 0
                success=2
            else:
                reward = -0.1
                done = 0
                success=2
            count = self.counter
            if done:
                self.current_return += reward
                self.prev_action = action
                if np.random.rand()<self.randomize:
                    position, _ = self.randomize_state()
                    self.randomize_maze(start=np.argmax(position),end= self.end_position )
                self.graph_data[0].append(self.current_return)
                self.current_return = 0
                self.graph_data[1].append(success)
                self.counter = 0
            else:
                self.prev_action=action
                self.prev_position = current_position.copy()
                self.prev_orientation = current_orientation.copy()
                self.current_return += reward

            return [current_position, current_orientation, current_readings], reward, done, ['Not Successful', 'Succesful', count][success]

    def reset_state(self):
        self.prev_position = np.zeros(self.maze.num_states)
        self.prev_position[0] = 1
        self.prev_orientation = np.zeros(4)
        self.prev_orientation[1] = 1

    def reset(self, randomize=None):
        randomize = self.randomize if randomize is None else randomize
        if np.random.rand() < randomize:
            position, _ = self.randomize_state()
            self.randomize_maze(start=np.argmax(position), end=self.end_position)
        else:
            self.reset_state()
        return [self.prev_position, self.prev_orientation, self.maze.get_sensor_readings(np.argmax(self.prev_position), np.argmax(self.prev_orientation))]

    def render(self, debug_info =''):
        save_title = 'move_{}'.format(self.counter)
        self.maze.show(actual_state=np.argmax(self.prev_position), orientation=np.argmax(self.prev_orientation), delay=0.05,
                 title='Current Score: {}\n Last Action Taken: {}\n Move: {}'.format(
                     self.current_return, self.prev_action,self.counter + '\n {}'.format(debug_info) if self.debug else ''
                 ), show=1, save=self.save_data,
                 save_title=save_title, ax=self.ax)

    def randomize_state(self):
        connected = False
        while not connected:
            position = np.zeros(self.maze.num_states)
            position[np.random.choice(self.maze.num_states)] = 1
            orientation = np.zeros(4)
            orientation[np.random.choice(range(4))] = 1
            connected = self.maze.has_path(np.argmax(position), self.end_position)
        self.prev_position= position
        self.prev_orientation = orientation
        return position, orientation

    def randomize_maze(self, start=0, end=None, ax=None, delay=1.0, display=0):
        end = self.end_position if end is None else end
        connected = False
        length = self.length
        num_colours = self.num_colours
        ax = self.ax if ax is None else ax
        while not connected:
            maze = Map.random_grid_map(num_colours, length)
            maze.show(delay=delay,ax=ax, show=display)
            ax.cla()
            connected = maze.has_path(start, end)
        self.maze = maze
        return maze

    def position_encoder(self, position):
        encoded_position = np.zeros(self.maze.num_states)
        encoded_position[position] = 1
        return encoded_position

    def get_next_orientation(self, action, prev_orientation):
        new_orientation = np.zeros(4)
        i = np.argmax(prev_orientation)
        if action == 'LEFT_TURN':
            new_orientation[(i - 1) % 4] = 1
        elif action == 'RIGHT_TURN':
            new_orientation[(i + 1) % 4] = 1
        return new_orientation


def get_cardinal(orientation):
    if orientation[0]:
        return 'W'
    if orientation[1]:
        return 'N'
    if orientation[2]:
        return 'E'
    if orientation[3]:
        return 'S'
