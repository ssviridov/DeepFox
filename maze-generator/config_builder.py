from animalai.envs.arena_config import ArenaConfig, RGB, Item, Vector3, Arena
import random
import numpy as np
from maze import Maze
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

def size_constraints_from_name(name):
    """Returns size constraints for specific object name"""
    if 'Goal' in name:
        size_min = np.ones(3)*0.5
        size_max = np.ones(3)*5
    elif 'Zone' in name:
        size_min = np.array([1.0,  0.0, 1.0])
        #size_max = np.array([40.0, 0.0, 40.0])
        size_max = np.array([10.0, 0.0, 10.0])
    elif 'Object' in name:
        size_min = np.array([1.0, 0.3, 3.0])
        #size_max = np.array([5.0, 2.0, 20.0])
        size_max = np.array([5.0, 2.0, 10.0])
    elif 'Cardbox' in name:
        size_min = np.ones(3)*0.5
        size_max = np.ones(3)*10
    elif 'Tunnel' in name:
        size_min = np.ones(3)*2.5
        size_max = np.ones(3)*10
    elif 'Ramp' in name:
        size_min = np.array([0.5, 0.1, 0.5])
        #size_max = np.array([40.0, 10.0, 40.0])
        size_max = np.array([10.0, 10.0, 10.0])
    elif 'Wall' in name:
        size_min = np.array([0.1, 0.1, 0.1])
        size_max = np.array([10.0, 10.0, 10.0])
        #size_max = np.array([40.0, 10.0, 40.0])
    else:
        raise ValueError('Unknown name')
    return size_min, size_max


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
    name = "BadGoal"

    is_bounce = random.random() < bounce_prob
    if is_bounce: name += "Bounce"

    bad_goal = Item(
        name,
        positions=[Vector3(*pos)],
        sizes=[Vector3(penalty, penalty, penalty)]
    )
    return bad_goal

N, S, W, E = ('n', 's', 'w', 'e')

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
    correct_x = (0+margin_error < x < ARENA_X-margin_error)
    correct_z = (0+margin_error < z < ARENA_Z-margin_error)
    correct_y = y >= 0.
    return correct_x and correct_y and correct_z


def create_maze(
        X, Z, time=500,
        wall_height=2.,
        wall_width=1.,
        transparent_prob=0.,
        color_prob=0.
):
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
        "-sd", '--save-dir',
        help='where to save newly generated mazes'
    )
    parser.add_argument(
        '-n', type=int, default=5,
        help='Number of mazes to create')
    parser.add_argument(
        '-t', '--time', type=int,
        default=500,
        help='Episode duration'
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = handle_commandline()
    X,Y = args.maze_size
    name_template = os.path.join(
        args.save_dir, 'maze-{}x{}-{}.yaml'
    )
    ensure_dir(name_template)

    for i in range(args.n):
        print('=== MAZE#{} ==='.format(i+1))
        config = create_maze(
            X,Y, args.time, 4., color_prob=0.15,
            transparent_prob=0.15
        )
        filename = name_template.format(X,Y,i+1)
        with open(filename,'w') as f:
            yaml.dump(config, f)
