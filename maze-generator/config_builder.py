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


def make_item(name, fix_color=True):
    """
    Make randomized item by name
    fix_color: Use standard colors (True) or randomize colors (False)
    """
    if name not in ALL_NAMES:
        raise ValueError('Unknown name')
    if not fix_color:
        color = RGB(*np.random.randint(255, size=3))
    elif name == 'Wall':
        color = FIXED_COLORS[random.randint(0, 1)]
    elif name == 'CylinderTunnel':
        color = FIXED_COLORS[0]
    elif name == 'Ramp':
        color = RGB(255, 0, 255)
    else:
        color = FIXED_COLORS[0]

    size_min, size_max = size_constraints_from_name(name)
    size = np.random.rand(3) * (size_max - size_min) + size_min
    size = Vector3(*[float(e) for e in size])

    radius = 0.5 * np.sqrt(size.x ** 2 + size.z ** 2)
    radius = min(radius, 19)

    position = np.random.rand(3) * (POS_MAX - POS_MIN - POS_MASK * 2 * radius) + POS_MIN + POS_MASK * radius
    position = Vector3(*[float(e) for e in position])
    if ('Zone' in name) or ('Tunnel' in name):
        position.y = 0

    rotation = float(np.random.rand()) * 360

    item = Item(name, [position], [rotation], [size], [color])
    return item


def make_wall(pos, size, rotation, transparent_prob=0., color_prob=.0):
    pos = [randomize(p) for p in pos]
    size = [randomize(s) for s in size]
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


def make_ramp(pos, size, rotation, color_prob=.0):
    pos = [randomize(p) for p in pos]
    size = [randomize(s) for s in size]
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


N, S, W, E = ('n', 's', 'w', 'e')


def randomize(float_or_pair):
    if np.isscalar(float_or_pair):
        return float_or_pair
    else:
        low = float_or_pair[0]
        high = float_or_pair[1]
        return random.uniform(low, high)


class ArenaMaze(object):
    ARENA_X = 40.
    ARENA_Z = 40.

    def __init__(
            self, X,Z,
            wall_width=1.,
            wall_height=4.,#could be a pair of values
            color_prob=0.,
            transparent_prob=0.,
            width_error=0.01,
            verbose=False,
    ):
        super(ArenaMaze, self).__init__()
        self.x_cells = X
        self.z_cells = Z
        self.wall_width=wall_width - width_error
        self.wall_height=wall_height
        self.color_prob=color_prob
        self.transparent_prob=transparent_prob

        self.maze = Maze.generate(X, Z)
        self.cell_len_x = (ARENA_X - (self.x_cells - 1) * wall_width) / self.x_cells
        self.cell_len_z = (ARENA_Z - (self.z_cells - 1) * wall_width) / self.z_cells

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

        if verbose:
            print('MAZE:')
            print(self.maze)
            print('one_side cells:',  len(self.maze.one_wall_cells()))
            print("corridor cells:",  len(self.maze.corridor_cells()))
            print('no_corner cells:', len(self.maze.no_corner_cells()))

    def _make_grid_wall(self, cell, direction):
        assert direction in DIRECTIONS, \
            "A direction must be one of those: {}".format(DIRECTIONS)
        x,z = cell
        pos_x = self.cell_len_x * x + self.wall_width * (x - 1)
        pos_z = self.cell_len_z * z + self.wall_width * (z - 1)

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

        wall = make_wall(pos, size, 0, self.transparent_prob, self.color_prob)
        return wall

    def _make_grid_pillar(self, south_east_cell):
        x, z = south_east_cell

        pos_x = self.cell_len_x * x + self.wall_width * (x - 0.5)
        pos_z = self.cell_len_z * z + self.wall_width * (z - 0.5)

        pillar = make_wall(
            (pos_x, 0., pos_z),
            self.pillar_size,
            0, self.transparent_prob, self.color_prob
        )
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
        pos_x = self.cell_len_x * x + self.wall_width * num_x_walls
        pos_z = self.cell_len_z * z + self.wall_width * num_z_walls
        return (pos_x, pos_z)

    def _create_maze_walls(self):
        walls = []
        # cycle to create walls:
        for z in range(self.z_cells):
            for x in range(self.x_cells):
                for d in [E, S]:  # wall directions
                    if d in self.maze[x, z]:  # is wall exists?
                        wall = self._make_grid_wall((x,z), d)
                        if wall: walls.append(wall)

        for z in range(1, self.z_cells):
            for x in range(1, self.x_cells):
                se_walls = set('se') & self.maze[x - 1, z - 1].walls
                nw_walls = set('nw') & self.maze[x, z].walls
                if se_walls or nw_walls:
                    pillar = self._make_grid_pillar((x,z))
                    if pillar:
                        walls.append(pillar)

        return walls

    def add_goals(self, reward, num_goals=1, bounce_prob=0.2, area_delim=0.01):
        goals = []
        if num_goals == 1:
            goal = make_goal(reward, bounce_prob=bounce_prob)
            goals.append(goal)

        else:
            z_area = self.ARENA_Z/num_goals
            for i in range(num_goals):
                low_z = z_area*i+area_delim
                high_z = z_area*(i+1)-area_delim
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
            x_area = self.ARENA_X / num_goals
            for i in range(num_goals):
                low_x = x_area*i + area_delim
                high_x = x_area*(i + 1) - area_delim
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
        min_ramp_length = 3
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
        l_ramp = make_ramp(l_ramp_pos, ramp_size, l_angle)
        r_ramp = make_ramp(r_ramp_pos, ramp_size, right_angle)

        wall_pos = (center_x, 0., center_z)
        wall_size = (self.wall_width, height, cell_width)
        obstacle_wall = make_wall(
            wall_pos, wall_size,
            wall_angle,
            self.transparent_prob,
            self.color_prob
        )

        return l_ramp, obstacle_wall, r_ramp

    def add_tunnels(self, cell, wall_width):
        pass

    def build_config(self,T):
        items = []
        items.extend(self._walls)
        items.extend(self._obstacles)
        items.extend(self._goals)

        arena = Arena(T, items)
        config = ArenaConfig()
        config.arenas[0] = arena
        return config


def save_config(config, name):
    ensure_dir(name)
    with open(filename,'w') as f:
        yaml.dump(config, f)


def create_maze_walls(
    x_cells,
    z_cells,
    wall_width=1.,
    wall_height=2.,
    **wall_kwargs,
):
    wall_width -= 0.01
    maze = Maze.generate(x_cells, z_cells)
    print('MAZE:')
    print(maze)
    print('one_side cells:', len(maze.one_wall_cells()))
    print("corridor cells:", len(maze.corridor_cells()))
    print('no_corner cells:', len(maze.no_corner_cells()))

    walls = []
    grid_size = (x_cells, z_cells)
    #cycle to create walls:
    for z in range(z_cells):
        for x in range(x_cells):
            for d in [E, S]: #wall directions
                if d in maze[x,z]: # is wall exists?
                    wall = make_grid_wall(
                        (x,z), grid_size, d,
                        wall_width, wall_height,
                        **wall_kwargs
                    )

                    if wall: walls.append(wall)

    pillar_size = (wall_width, wall_height, wall_width)
    for z in range(1,z_cells):
        for x in range(1, x_cells):
            se_walls =  set('se') & maze[x-1, z-1].walls
            nw_walls = set('nw') & maze[x,z].walls
            if se_walls or nw_walls:
                pillar = make_grid_pillar(
                    (x,z), grid_size,
                    wall_width, wall_height,
                    **wall_kwargs
                )
                if pillar:
                    walls.append(pillar)

    return walls


def make_grid_wall(
        cell, grid_size,
        direction,
        wall_width=1.,
        wall_height=2.,
        transparent_prob=0.,
        color_prob=0.
):
    x_cells, z_cells = grid_size
    x, z = cell
    x_wall_len = (ARENA_X - (x_cells - 1) * wall_width) / x_cells
    z_wall_len = (ARENA_Z - (z_cells - 1) * wall_width) / z_cells
    h_wall_size = (x_wall_len, wall_height, wall_width)
    v_wall_size = (wall_width, wall_height, z_wall_len)

    pos_x = x_wall_len*x + wall_width*(x-1)
    pos_z = z_wall_len*z + wall_width*(z-1)

    if direction in 'ns':
        size = h_wall_size

        pos_x += wall_width + x_wall_len/2
        pos_z += wall_width/2.
        if direction == 's':
            pos_z += z_wall_len + wall_width
    else:
        size = v_wall_size

        pos_x += wall_width/2.
        pos_z += wall_width + z_wall_len/2
        if direction == 'e':
            pos_x += x_wall_len + wall_width

    pos = (pos_x, 0., pos_z)

    if check_position(pos):
        return make_wall(pos, size, 0, transparent_prob, color_prob)
    else:
        return None


def make_grid_pillar(
        cell,
        grid_size,
        wall_width=1.,
        wall_height=2.,
        transparent_prob=0.,
        color_prob=0.,
):
    x_cells, z_cells = grid_size
    x, z = cell
    x_wall_len = (ARENA_X - (x_cells - 1) * wall_width) / x_cells
    z_wall_len = (ARENA_Z - (z_cells - 1) * wall_width) / z_cells

    pos_x = x_wall_len * x + wall_width * (x - 0.5)
    pos_z = z_wall_len * z + wall_width * (z - 0.5)

    pillar = make_wall(
        (pos_x, 0., pos_z),
        (wall_width, wall_height, wall_width),
        0, transparent_prob, color_prob
    )
    return pillar


def check_position(pos, margin_error=0.01):
    x, y, z = pos
    correct_x = ((0+margin_error) < x < (ARENA_X-margin_error))
    correct_z = ((0+margin_error) < z < (ARENA_Z-margin_error))
    correct_y = y >= 0.
    return correct_x and correct_y and correct_z


def create_maze(
        X, Z, time=500,
        n_goals = 1, #if n_goals == 1 -> green target, otherwise -> gold_target
        wall_height=2.,
        wall_width=1.,
        transparent_prob=0.,
        color_prob=0.
):
    assert n_goals >= 0, "Negative number of goals?"
    items = []

    walls = create_maze_walls(
        X, Z, wall_width, wall_height,
        transparent_prob=transparent_prob,
        color_prob=color_prob,
    )
    print('generated  {} walls!'.format(len(walls)))
    items.extend(walls)
    items.append(make_goal(2.5))
    items.append(make_bad_goal(1.))

    arena = Arena(time, items)
    config = ArenaConfig()
    config.arenas[0] = arena
    return config


def ensure_dir(file_path):
    """
    Checks if the containing directories exist,
    and if not, creates them.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = handle_commandline()

    X,Y = args.maze_size
    num_goals = 3 if X * Y >= 10 else 2
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
            verbose=True
        )

        maze.add_goals((1.,2.), num_goals, bounce_prob=0.2)
        maze.add_bad_goals(1.5, 2, 0.2)
        maze.add_ramps(args.num_ramps)

        config = maze.build_config(args.time)
        filename = name_template.format(X, Y, i + 1)
        save_config(config, filename)
