import time
import argparse
from PIL import Image
from gym.envs.classic_control.rendering import SimpleImageViewer
import numpy as np
import pyglet.window
from a2c_ppo_acktr.aai_wrapper import make_env_aai
from a2c_ppo_acktr.aai_config_generator import SingleConfigGenerator, ListSampler, \
    FixedTimeGenerator, RandomizedGenerator
import cv2
from a2c_ppo_acktr.preprocessors import ObjectClassifier

aai_path = "aai_resources/env/AnimalAI"
config_path = "aai_resources/asorokin/" #"aai_resources/test_configs/"
config_path = config_path + "grid_mazes/5x5" #/5_walls/5_walls_gold.yaml"


def get_config_name(env_aai):
    name = env_aai.unwrapped.config_name[:-5]
    return name


class EnvInteractor(SimpleImageViewer):
    """
    This class manages the user interface for playing
    obstacle tower, pausing the game, etc.
    """
    def __init__(self):
        super().__init__(maxwidth=800)
        self.keys = pyglet.window.key.KeyStateHandler()
        self._paused = False
        self._jump = False
        self._finish_early = False
        self._last_image = None
        self.imshow(np.zeros([168, 168, 3], dtype=np.uint8))

    def imshow(self, image):
        self._last_image = image
        was_none = self.window is None
        image = Image.fromarray(image.astype(np.uint8))
        image = image.resize((800, 400))
        image = np.array(image)
        super().imshow(image)
        if was_none:
            self.window.event(self.on_key_press)
            self.window.push_handlers(self.keys)

    def get_action(self):
        move = 0 #[None, FORWARD, BACKWARD]
        turn = 0 #[None, RIGHT, LEFT]
        NUM_TURNS = 3

        if self.keys[pyglet.window.key.UP]:
            move = 1
        elif self.keys[pyglet.window.key.DOWN]:
            move = 2
        if self.keys[pyglet.window.key.LEFT]:
            turn += 2
        elif self.keys[pyglet.window.key.RIGHT]:
            turn += 1

        action_id = move*NUM_TURNS + turn

        if self.keys[pyglet.window.key.ESCAPE]:
            self._finish_early = True

        return action_id

    def pause(self):
        self._paused = True

    def paused(self):
        if self.keys[pyglet.window.key.P]:
            self._paused = True
        elif self.keys[pyglet.window.key.R]:
            self._paused = False
        return self._paused

    def finish_early(self):
        return self._finish_early

    def on_key_press(self, x, y):
        if x == pyglet.window.key.SPACE:
            self._jump = True
        return True

    def reset(self):
        self._jump = False
        self._paused = False
        self._finish_early = False

    def run_loop(self, step_fn):
        """
        Run an environment interaction loop.

        The step_fn will be continually called with
        actions, and it should return observations.

        When step_fn returns None, the loop is done.
        """
        last_time = time.time()
        print("start_episode!")
        while not self.finish_early():
            if not self.paused():
                action = self.get_action()
                obs = step_fn(action)

                if obs is None:
                    return
                self.imshow(obs)
            else:
                # Needed to run the event loop
                self.imshow(self._masked_img)
            pyglet.clock.tick()
            delta = time.time() - last_time
            time.sleep(max(0, 1 / 10 - delta))
            last_time = time.time()


def main(num_repeat, info_mode):
    if config_path.endswith("yaml"):
        gen_config = SingleConfigGenerator.from_file(config_path)
        num_configs = 1
    else:
        gen_config = ListSampler.create_from_dir(config_path)
        num_configs = len(gen_config.configs)

    num_episodes = num_repeat*num_configs

    gen_config = FixedTimeGenerator(gen_config, 1500)
    #gen_config = RandomizedGenerator(gen_config, 0.9, 0.3)

    rank = np.random.randint(1200, 2000)
    viewer = EnvInteractor()


    make = make_env_aai(
        aai_path, gen_config, rank, False,
        grid_oracle_kwargs=dict(
            oracle_type="angles",
            trace_decay=0.999,# 0.9999,
            num_angles=3,
            cell_side=2,
        ),
        channel_first=True,
        image_only=False
    )
    env = make()
    if info_mode == 'classifier':
        env = ObjectClassifier().wrap_env(env)

    for ep in range(num_episodes):
        run_episode(ep, env, viewer)
        # episodes are reset automatically


def run_episode(ep, env, viewer):
    #seed = select_seed()
    #env.seed(seed)
    seed = 17
    obs = env.reset()
    config_name = get_config_name(env)
    print("#### Episode #{}: {} ##### ".format(ep, config_name))
    record_episode(seed, env, viewer)


def rotate(vec, angle):
    betta = np.radians(angle)
    x, y, z = vec
    x1 = np.cos(betta) * x - np.sin(betta) * z
    z1 = np.sin(betta) * x + np.cos(betta) * z
    return np.array([x1, y, z1])


def concat_images(image, map_view):
    map_view = map_view[:3].transpose(1,2,0)*255
    map_view = cv2.resize(map_view, (84, 84))
    return np.concatenate((image, map_view), axis=1)


def angle360(obs):
    return (obs['angle'][0]+1.)*180.


def color_filter(image, color):
    has_color = np.logical_and.reduce(image==np.asarray(color), 2)*255
    has_color = np.stack([has_color]*3, 2)
    return np.concatenate([image, has_color], axis=1)


def record_episode(seed, env, viewer):

    action_log = []
    reward_log = []
    avg_speed = np.zeros(3)
    coef = 0.92
    total_steps = [0]
    actions =  [1]*800 #[0]*100 +  [1]*15 + [0]*10 + [1]*15 + [0]*10 + [1]*15 + [0]*10 + [1]*15 + [0]*5000
    #map_window = SimpleImageViewer(400)
    #print("Step #0 angle: {:.1f}".format(angle360(obs)))
    max_blackout = [0]
    curr_blackout = [0]

    def step(action):
        #action = actions[total_steps[0]]
        action_log.append(action)

        obs, rew, done, info = env.step(action) #actions[total_steps[0]])

        reward_log.append(rew)
        total_steps[0] += 1

        time.sleep(0.025)

        curr_blackout[0] = (curr_blackout[0]+1)*(obs['image'].max() <= 0.)
        max_blackout[0] = max(curr_blackout[0], max_blackout[0])

        if 'objects' in obs:
            print_classifier_info(total_steps[0], obs, info)
        else:
            print_default_info(total_steps[0], obs, info)

        img = obs['image'].transpose(1, 2, 0)
        img = concat_images(img, obs['visited'])
        #print("VISITED:\n", obs['visited'][0])
        #map_window.imshow(map2img(obs['visited']))
        if done:
            print('\nFinal Reward: {}, max_blackout: {}\nINFO: {}\n'.format(rew, max_blackout[0], info))
            return None

        return img


    viewer.run_loop(step)

    #viewer.run_loop(step)

def print_default_info(step, obs, info, ):
    x, y, z = obs['pos']
    dx, dy, dz = obs['speed']
    angle = obs['angle'][0]
    color = obs['image'][:,42, 42]

    print("\rstep#{} angle={:.1f}, pos=({:.2f}, {:.2f}, {:.2f}),"
          " speed=({:.2f}, {:.2f}, {:.2f}), n_visited={}, expl_r={}, pixel[42,42]: {}".format(
        step, (angle + 1.) * 180, x * 70, z * 70, y * 70,
        dx * 10, dz * 10, dy * 10,
        info['grid_oracle']['n_visited'],
        info['grid_oracle']['r'], color.astype(int)  # rew
    ), end="")


LABELS = (
        'Nothing', 'Wall', 'WallTransparent', 'Ramp',
        'CylinderTunnel', 'CylinderTunnelTransparent',
        'Cardbox', 'UObject/LObject', 'GoodGoal', 'BadGoal',
        'GoodGoadMulti', 'DeathZone', 'HotZone'
    )

def print_classifier_info(step, obs, info):
    probs = obs['objects']
    print(
        "\rSTEP#{} N={:.2f}, W={:.2f}, WT={:.2f}, R={:.2f}, " 
        "CT={:.2f}, CTT={:.2f}, Cb={:.2f}, U/L={:.2f}, GG={:.2f} "
        "BG={:.2f}, GGM={:.2f}, DZ={:.2f}, HZ={:.2f}".format(
            step, *probs,
    ), end="")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--num-repeat',
        type=int, default=1,
        help='Number of times we play each config in the env-path folder'
    )

    parser.add_argument(
        '-i','--info', type=str,
        default='default', choices=['default', 'classifier'],
        help='outputs default info at each step or info from object classifier'
    )
    args = parser.parse_args()

    main(args.num_repeat, args.info)

