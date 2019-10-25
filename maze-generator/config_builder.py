from animalai.envs.arena_config import ArenaConfig, RGB, Item, Vector3, Arena
import random
import numpy as np
from maze import Maze, DIRECTIONS, N,S,W,E
import os
import argparse
import yaml

GOAL_NAMES = {'GoodGoal', 'GoodGoalBounce',
              'BadGoal', 'BadGoalBounce',
              'GoodGoalMulti', 'GoodGoalMultiBounce'}

STATIC_GOAL_NAMES = {'GoodGoal', 'BadGoal', 'GoodGoalMulti'}

GOOD_GOAL_NAMES = {'GoodGoal', 'GoodGoalBounce',
                   'GoodGoalMulti', 'GoodGoalMultiBounce'}

ZONE_NAMES = {'DeathZone', 'HotZone'}

TUNNEL_NAMES = {'CylinderTunnel', 'CylinderTunnelTransparent'}

IMMOVABLE_NAMES = {'Wall', 'WallTransparent', 'Ramp', 'CylinderTunnel', 'CylinderTunnelTransparent'}

COLORED_NAMES = {'Wall', 'Ramp', 'CylinderTunnel'}

MOVABLE_NAMES = {'Cardbox1', 'Cardbox2', 'UObject', 'LObject', 'LObject2'}

ALL_NAMES = GOAL_NAMES | ZONE_NAMES | IMMOVABLE_NAMES | MOVABLE_NAMES

SWAP_CLASSES = [GOAL_NAMES, ZONE_NAMES, {'Ramp'},
                {'Cardbox1', 'Cardbox2'},
                {'UObject', 'LObject', 'LObject2'},
                {'Wall', 'WallTransparent'},
                {'CylinderTunnel', 'CylinderTunnelTransparent'}]

FIXED_COLORS = [RGB(153, 153, 153), RGB(0, 0, 255)]

POS_MAX = np.array([39.9, 10, 39.9])
POS_MIN = np.array([0.1,  0,  0.1])
POS_MASK = np.array([1.0, 0.0, 1.0])

FIXED_COLORS = [RGB(153, 153, 153), RGB(0, 0, 255)]

MIN_NUM_ITEMS = 3
MAX_NUM_ITEMS = 25

ARENA_X = 40.
ARENA_Z = 40.

def fix_size(size, size_error=0.01):
    "To avoid collisions due to float rounding errors"
    size = [s - size_error for s in size]
    return size


def make_wall(pos, size, rotation, transparent_prob=0., color_prob=.0, size_error=0.01):
    pos = [randomize(p) for p in pos]
    size = [randomize(s) for s in size]
    size = fix_size(size, size_error)
    rotation = randomize(rotation)

    if not check_position(pos): return None


    is_transparent = random.random() < transparent_prob
    random_color = random.random() < color_prob
    name = "WallTransparent" if is_transparent else "Wall"
    color = RGB(-1,-1,-1) if random_color else RGB(153,153,153)

    wall = Item(
        name,
        positions=[Vector3(*pos)],
        rotations=[rotation],
        sizes=[Vector3(*size)],
        colors=[color]
    )

    return wall


def make_goal(reward, pos=(-1,-1,-1), is_multi=False, bounce_prob=0.0):

    reward = randomize(reward)
    pos = [randomize(p) for p in pos]

    name = "GoodGoal"

    is_bounce = random.random() < bounce_prob
    if is_multi: name += "Multi"
    if is_bounce: name += "Bounce"

    goal = Item(
        name,
        positions=[Vector3(*pos)],
        sizes=[Vector3(reward, reward, reward)]
    )
    return goal


def make_bad_goal(penalty, pos=(-1,-1,-1), bounce_prob=0.0):
    penalty = randomize(penalty)
    pos = [randomize(p) for p in pos]

    name = "BadGoal"

    is_bounce = random.random() < bounce_prob
    if is_bounce: name += "Bounce"

    bad_goal = Item(
        name,
        positions=[Vector3(*pos)],
        sizes=[Vector3(penalty, penalty, penalty)]
    )
    return bad_goal


def make_ramp(pos, size, rotation, color_prob=.0, size_error=0.01):
    pos = [randomize(p) for p in pos]
    size = [randomize(s) for s in size]
    size = fix_size(size, size_error)

    rotation = randomize(rotation)

    if not check_position(pos): return None

    random_color = random.random() < color_prob
    name = "Ramp"
    color = RGB(-1,-1,-1) if random_color else RGB(255,0,255)

    ramp = Item(
        name,
        positions=[Vector3(*pos)],
        rotations=[rotation],
        sizes=[Vector3(*size)],
        colors=[color]
    )
    return ramp

def fix_agent(pos, rotation=-1):
    rotation = randomize(rotation)
    pos = [randomize(p) for p in pos]
    if not check_position(pos): return None

    agent = Item(
        name="Agent",
        positions=[Vector3(*pos)],
        rotations=[rotation],
    )
    return agent

N, S, W, E = ('n', 's', 'w', 'e')


def randomize(float_or_pair):
    if np.isscalar(float_or_pair):
        return float_or_pair
    else:
        low = float_or_pair[0]
        high = float_or_pair[1]
        return random.uniform(low, high)


class ArenaMaze(object):
    X_MIN = 0.
    X_MAX = 40.
    Z_MIN = 0.
    Z_MAX = 40.

    def __init__(
            self, X,Z,
            wall_width=1.,
            wall_height=4.,#could be a pair of values
            color_prob=0.,
            transparent_prob=0.,
            size_error=0.01,
            offset=0.,
            verbose=False,
    ):
        super(ArenaMaze, self).__init__()
        off_x, off_z = (offset, offset) if np.isscalar(offset) else offset
        self.X_MIN = self.X_MIN + off_x
        self.X_MAX = self.X_MAX - off_x

        self.Z_MIN = self.Z_MIN + off_z
        self.Z_MAX = self.Z_MAX - off_z
        self.arena_len_x = self.X_MAX - self.X_MIN
        self.arena_len_z = self.Z_MAX - self.Z_MIN

        self.x_cells = X
        self.z_cells = Z
        self.wall_width=wall_width
        self.wall_height=wall_height
        self.color_prob=color_prob
        self.transparent_prob=transparent_prob
        self.size_error = size_error

        self.maze = Maze.generate(X, Z)

        self.cell_len_x = (self.arena_len_x - (self.x_cells - 1) * wall_width) / self.x_cells
        self.cell_len_z = (self.arena_len_z - (self.z_cells - 1) * wall_width) / self.z_cells

        self.h_wall_size = (self.cell_len_x, wall_height, wall_width)
        self.v_wall_size = (wall_width, wall_height, self.cell_len_z)
        self.pillar_size = (wall_width, wall_height, wall_width)

        self.sizes = dict(
            cell=(self.cell_len_x, self.cell_len_z),
            h_wall=self.h_wall_size,
            v_wall=self.v_wall_size,
            pillar=self.pillar_size
        )

        self._walls = self._create_maze_walls()
        self._goals = []
        self._movables = []
        self._obstacles = []
        self.agent = None

        if verbose:
            print('MAZE:')
            print(self.maze)
            print('one_side cells:',  len(self.maze.one_wall_cells()))
            print("corridor cells:",  len(self.maze.corridor_cells()))
            print('no_corner cells:', len(self.maze.no_corner_cells()))
            print('num_walls:', len(self._walls))

    def _make_grid_wall(self, cell, direction):
        assert direction in DIRECTIONS, \
            "A direction must be one of those: {}".format(DIRECTIONS)
        x,z = cell
        pos_x, pos_z = self.top_left_point(x,z)
        #pos_x = self.cell_len_x * x + self.wall_width * (x - 1) + self.X_MIN
        #pos_z = self.cell_len_z * z + self.wall_width * (z - 1) + self.Z_MIN

        if direction in [N,S]:
            size = self.h_wall_size

            pos_x += self.wall_width + self.cell_len_x / 2
            pos_z += self.wall_width / 2.
            if direction == S:
                pos_z += self.cell_len_z + self.wall_width
        else:
            size = self.v_wall_size

            pos_x += self.wall_width / 2.
            pos_z += self.wall_width + self.cell_len_z / 2
            if direction == 'e':
                pos_x += self.cell_len_x + self.wall_width

        pos = (pos_x, 0., pos_z)

        wall = make_wall(pos, size, 0, self.transparent_prob, self.color_prob, self.size_error)
        return wall

    def _make_grid_pillar(self, south_east_cell):
        x, z = south_east_cell
        print("pillar to the top left of {}".format((x,z)))
        pos_x, pos_z = self.top_left_point(x,z) #without walls
        pos_x += self.wall_width * 0.5 #move to the center of wall
        pos_z += self.wall_width * 0.5
        print('pillar coords:', (pos_x, pos_z))
        #pos_x = self.cell_len_x * x + self.wall_width * (x - 0.5) + self.X_MIN
        #pos_z = self.cell_len_z * z + self.wall_width * (z - 0.5) + self.Z_MIN

        pillar = make_wall(
            (pos_x, 0., pos_z),
            self.pillar_size,
            0, self.transparent_prob, self.color_prob,
            self.size_error
        )
        print("pillar is not None" if pillar else "None", "\n")
        return pillar

    def top_left_point(self, x,z, inside_walls=False):
        """
        Returns coordinates of top_left(up->down, left->right) corner of a cell.
        If inside_walls is True then return point inside the surrounding walls,
        Otherwise surrounding walls are considered to be a part of the cell
        (Adjacent cells share walls between them)
        """
        num_x_walls = x if inside_walls else x-1
        num_z_walls = z if inside_walls else z-1
        pos_x = self.cell_len_x * x + self.wall_width * num_x_walls + self.X_MIN
        pos_z = self.cell_len_z * z + self.wall_width * num_z_walls + self.Z_MIN
        return (pos_x, pos_z)

    def _create_maze_walls(self):
        walls = []
        # cycle to create walls:
        for z in range(self.z_cells):
            for x in range(self.x_cells):
                directions = [E,S]
                if z == 0: directions.append(N)
                if x == 0: directions.append(W)

                for d in directions:  # wall directions
                    if d in self.maze[x, z]:  # is wall exists?
                        wall = self._make_grid_wall((x,z), d)
                        if wall: walls.append(wall)

        for z in range(0, self.z_cells+1):
            for x in range(0, self.x_cells+1):
                prev_cell = self.maze[x - 1, z - 1] #up and left from the current
                prev_walls = prev_cell.walls if prev_cell else set()
                se_walls = set('se') & prev_walls

                curr_cell = self.maze[x,z]
                curr_walls = curr_cell.walls if curr_cell else set()
                nw_walls = set('nw') & curr_walls

                if se_walls or nw_walls:
                    pillar = self._make_grid_pillar((x,z))
                    if pillar:
                        walls.append(pillar)

        #cycles above ommit two corner pillars, and i'm to lazy to rewrite entire algorithm:
        if self.maze[0, self.z_cells-1].walls & set('ws'):
            pillar = self._make_grid_pillar((0, self.z_cells))
            if pillar: walls.append(pillar)

        if self.maze[self.x_cells-1, 0].walls & set('ne'):
            pillar = self._make_grid_pillar((self.x_cells, 0))
            if pillar: walls.append(pillar)


        return walls

    def add_goals(self, reward, num_goals=1, bounce_prob=0.2, area_delim=0.01):
        goals = []
        if num_goals == 1:
            goal = make_goal(reward, bounce_prob=bounce_prob)
            goals.append(goal)

        else:
            z_area = self.arena_len_z /num_goals
            for i in range(num_goals):
                low_z = z_area*i+area_delim + self.Z_MIN
                high_z = z_area*(i+1)-area_delim + self.Z_MIN
                pos_i = (-1,-1, (low_z, high_z))
                goal_i = make_goal(reward, pos_i, True, bounce_prob)
                goals.append(goal_i)

        self._goals.extend(goals)

    def add_bad_goals(self, reward, num_goals=1, bounce_prob=0.2, area_delim=-0.01):
        goals = []
        if num_goals == 1:
            goal = make_goal(reward, bounce_prob=bounce_prob)
            goals.append(goal)

        else:
            x_area = self.arena_len_x / num_goals
            for i in range(num_goals):
                low_x = x_area*i + area_delim + self.X_MIN
                high_x = x_area*(i + 1) - area_delim + self.X_MIN
                pos_i = ((low_x, high_x), -1, -1)
                goal_i = make_bad_goal(reward, pos_i, bounce_prob)
                goals.append(goal_i)

        self._goals.extend(goals)

    def add_ramps(self, n, percent=None, between_cells=False):
        cells = self.maze.corridor_cells()
        n = min(len(cells), n)
        if n == 0: return
        ids = set(np.random.choice(len(cells), size=n, replace=False))
        for id in ids:
            x,z = cells[id]
            is_horizontal = 'n' in self.maze[x,z].walls #horisontal corridor
            print('add horizontal ramp' if is_horizontal else 'add vertical ramp')
            obstacle_objects = self._make_ramp((x,z), is_horizontal)
            self._obstacles.extend(obstacle_objects)

    def _make_ramp(self, cell, is_horizontal):
        min_ramp_length = 2.5
        max_ramp_width = 5
        max_ramp_height = 3
        #check if there is enough space for a ramp obstacle:
        min_cell_side = min(self.cell_len_x, self.cell_len_z)
        assert max_ramp_width < min_cell_side
        assert min_ramp_length < (min_cell_side-self.wall_width)/2

        x,z = cell
        left_x, top_z = self.top_left_point(x, z, inside_walls=True)
        center_x = left_x + self.cell_len_x / 2
        center_z = top_z + self.cell_len_z / 2

        if is_horizontal:
            cell_len, cell_width = self.cell_len_x, self.cell_len_z
        else:
            cell_len, cell_width = self.cell_len_z, self.cell_len_x

        length = randomize( (min_ramp_length, (cell_len-self.wall_width)/2) )
        width =  randomize((1.8, max_ramp_width))
        height = randomize((1., min(self.wall_height-1, max_ramp_height, length)))

        # x - is a width of a ramp, z is it's length ¯\_(ツ)_/¯
        ramp_size = (width, height, length)

        offset = length / 2 + self.wall_width / 2

        if is_horizontal:
            # angle=0 means upward direction is from north->south,  angle=270: west-> east ¯\_(ツ)_/¯
            l_angle = 270
            wall_angle=0
            l_ramp_pos = (center_x-offset, 0, center_z)
            r_ramp_pos = (center_x+offset, 0, center_z)
        else:
            l_angle = 180
            wall_angle=90
            l_ramp_pos = (center_x, 0, center_z-offset)
            r_ramp_pos = (center_x, 0, center_z+offset)

        right_angle = (180 + l_angle)%360
        l_ramp = make_ramp(l_ramp_pos, ramp_size, l_angle, size_error=self.size_error)
        r_ramp = make_ramp(r_ramp_pos, ramp_size, right_angle, size_error=self.size_error)

        wall_pos = (center_x, 0., center_z)
        wall_size = (self.wall_width, height, cell_width)
        obstacle_wall = make_wall(
            wall_pos, wall_size,
            wall_angle,
            self.transparent_prob,
            self.color_prob,
            self.size_error
        )

        return l_ramp, obstacle_wall, r_ramp


    def fix_agent_inside(self, cell=None):
        if cell is None:
            x_min, x_max = self.X_MIN, self.X_MAX
            z_min, z_max = self.Z_MIN, self.Z_MAX
        else:
            x_min, z_min = self.top_left_point(*cell, inside_walls=True)
            x_max, z_max = x_min + self.cell_len_x, z_min + self.cell_len_z

        pos = [(x_min, x_max), 3.5, (z_min, z_max)]
        self.agent = fix_agent(pos)


    def add_tunnels(self, cell, wall_width):
        pass

    def build_config(self,T):
        items = []
        items.extend(self._walls)
        items.extend(self._obstacles)
        items.extend(self._goals)
        if self.agent:
            items.append(self.agent)

        arena = Arena(T, items)
        config = ArenaConfig()
        config.arenas[0] = arena
        return config


def check_position(pos, margin_error=0.01):
    x, y, z = pos
    correct_x = ((0+margin_error) < x < (ARENA_X-margin_error))
    correct_z = ((0+margin_error) < z < (ARENA_Z-margin_error))
    correct_y = y >= 0.
    return correct_x and correct_y and correct_z


def ensure_dir(file_path):
    """
    Checks if the containing directories exist,
    and if not, creates them.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_config(config, name):
    ensure_dir(name)
    with open(filename,'w') as f:
        yaml.dump(config, f)


def handle_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "maze_size", nargs=2, type=int, default=(3, 3),
        help='Split arena on (x, z) cells and generates maze based on these cells. Bigger x and z values will lead to a harder maze with more narrow corridors.'
    )
    parser.add_argument(
        "-sd", '--save-dir', required=True,
        help='where to save newly generated mazes'
    )
    parser.add_argument(
        '-m', '--mazes', type=int, default=5,
        help='Number of mazes to create [default: 5]')
    parser.add_argument(
        '-t', '--time', type=int,
        default=500,
        help='Episode duration [default: 500]'
    )
    parser.add_argument(
        '-nr', '--num-ramps', type=int, default=2,
        help="Desired number of ramps in a maze [default: 2]"
    )
    parser.add_argument(
        '-ng', '--num-goals', type=int, default=2,
        help="Nuber of goals in maze, if n > 1 then we use gold goals, [default: 2]"
    )
    parser.add_argument(
        '-nt', '--num-tunnels', type=int, default=0,
        help='Desired number of tunnels in a maze [default: ignored]'
    )
    parser.add_argument(
        '-nd', '--num-death-zones', type=int, default=0,
        help="Desired number of death zones instead of walls [default: ignored]"
    )
    parser.add_argument(
        '--color-prob', type=float, default=0.15,
        help="Probability of random color in  for walls and tunnels [default: 0.15]"
    )
    parser.add_argument(
        '--transparent-prob', type=float, default=0.15,
        help="Chance to generate a transparent object(walls and tunnels) [default: 0.15]"
    )
    parser.add_argument(
        '--offset', nargs=2, default=(0.,0.), type=float,
        help="(offset_x, offset_z) moves outer walls of the maze from arena borders.For example:\n"
             "offset=(0., 0.) then maze covers entire arena;\n"
             "offset=(10.,10.) them maze has size (20x20), covers 1/4 of the arena and is placed in the center."

    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = handle_commandline()

    X,Y = args.maze_size

    num_bad_goals = 2

    name_template = os.path.join(
        args.save_dir, 'maze-{}x{}-{}.yaml'
    )

    for i in range(args.mazes):
        print('=== MAZE#{} ==='.format(i + 1))
        maze = ArenaMaze(
            X,Y,
            wall_height=4.,
            color_prob=args.color_prob,
            transparent_prob=args.transparent_prob,
            offset=args.offset,
            verbose=True,
        )

        maze.add_goals((1.,2.), args.num_goals, bounce_prob=0.2)
        maze.add_bad_goals(1.5, num_bad_goals, 0.2)
        maze.add_ramps(args.num_ramps)
        maze.fix_agent_inside()

        config = maze.build_config(args.time)
        filename = name_template.format(X, Y, i + 1)
        save_config(config, filename)
