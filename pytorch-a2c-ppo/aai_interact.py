import time

from PIL import Image
from gym.envs.classic_control.rendering import SimpleImageViewer
import numpy as np
import pyglet.window
from a2c_ppo_acktr.aai_wrapper import AnimalAIWrapper
from a2c_ppo_acktr.aai_config_generator import SingleConfigGenerator

aai_path = "aai_resources/env/AnimalAI"
aai_config_dir = "aai_resources/test_configs/"
curr_config = aai_config_dir + "MySample.yaml"

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
        image = image.resize((400, 400))
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



def main():
    rank = np.random.randint(0, 1000)
    viewer = EnvInteractor()
    gen_config = SingleConfigGenerator.from_file(curr_config)
    env = AnimalAIWrapper(
        aai_path,
        rank,
        gen_config,
        channel_first=False,
        image_only=False,
    )

    run_episode(env, viewer)


def run_episode(env, viewer):
    #seed = select_seed()
    #env.seed(seed)
    seed = 17
    obs = env.reset()
    record_episode(seed, env, viewer, obs)


def rotate(vec, angle):
    betta = np.radians(angle)
    x, y, z = vec
    x1 = np.cos(betta) * x - np.sin(betta) * z
    z1 = np.sin(betta) * x + np.cos(betta) * z
    return np.array([x1, y, z1])

def record_episode(seed, env, viewer, obs):

    action_log = []
    reward_log = []
    avg_speed = np.zeros(3)
    coef = 0.92
    total_steps = [0]
    actions = [0]*100 +  [1]*15 + [0]*10 + [1]*15 + [0]*10 + [1]*15 + [0]*10 + [1]*15 + [0]*5000

    def step(action):
        action_log.append(action)

        obs, rew, done, info = env.step(action) #actions[total_steps[0]])

        x,y,z = obs['pos']
        dx,dy,dz = obs['speed']
        angle = obs['angle'][0]
        
        reward_log.append(rew)
        avg_speed[:] = coef * avg_speed + 0.001*obs['speed']
        speed_reward = np.linalg.norm(avg_speed)
        total_steps[0] += 1

        time.sleep(0.025)
        print("\rstep#{} speed_r={:0.4f} angle={}, pos=({:.2f}, {:.2f}, {:.2f}), speed=({:.2f}, {:.2f}, {:.2f})".format(
            total_steps[0], speed_reward,
            angle, x,z,y, dx,dz,dy# rew
        ), end="")

        if done:
            print('\nFinal Reward: {}, INFO: {}'.format(rew, info))
            return None

        return obs["image"]
   # #config = remove_cardbox2(env.config_generator.next_config())
    #ob0 = env.reset(config).astype(np.uint8)
    #config = env.config_generator.next_config()
    #ob1 = env.reset(config).astype(np.uint8)
    #masked = (ob1 != ob0).astype(np.uint8)*ob1
    #viewer._masked_img = masked
    viewer.run_loop(step)

    #viewer.run_loop(step)

import copy
def remove_cardbox2(config_dict):
    config = copy.deepcopy(config_dict['config'])
    cardboxes = config.arenas[0].items[2]
    cardboxes.positions.pop()
    cardboxes.rotations.pop()
    cardboxes.sizes.pop()
    return {"config":config, "config_name":"without_cardbox"}

if __name__ == "__main__":
    main()

