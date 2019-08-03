from animalai.envs import UnityEnvironment
import os
import logging
logger = logging.getLogger("mlagents.envs")
import platform
import glob
import subprocess
from sys import platform

from animalai.envs import UnityEnvironmentException

class UnityEnvHeadless(UnityEnvironment):
    def __init__(
            self, file_name=None,
            worker_id=0,
            base_port=5005,
            seed=0,
            docker_training=False,
            n_arenas=1,
            play=False,
            arenas_configurations=None,
            inference=False,
            resolution=None,
            headless=False,
    ):
        self.headless=headless
        super(UnityEnvHeadless, self).__init__(
            file_name=file_name,
            worker_id=worker_id,
            base_port=base_port,
            seed=seed,
            docker_training=docker_training,
            n_arenas=n_arenas,
            play=play,
            arenas_configurations=arenas_configurations,
            inference=inference,
            resolution=resolution,
        )

    def executable_launcher(self, file_name, docker_training):
        cwd = os.getcwd()
        file_name = (file_name.strip()
                     .replace('.app', '').replace('.exe', '').replace('.x86_64', '').replace('.x86',
                                                                                             ''))
        true_filename = os.path.basename(os.path.normpath(file_name))
        logger.debug('The true file name is {}'.format(true_filename))
        launch_string = None
        if platform == "linux" or platform == "linux2":
            candidates = glob.glob(os.path.join(cwd, file_name) + '.x86_64')
            if len(candidates) == 0:
                candidates = glob.glob(os.path.join(cwd, file_name) + '.x86')
            if len(candidates) == 0:
                candidates = glob.glob(file_name + '.x86_64')
            if len(candidates) == 0:
                candidates = glob.glob(file_name + '.x86')
            if len(candidates) > 0:
                launch_string = candidates[0]

        elif platform == 'darwin':
            candidates = glob.glob(
                os.path.join(cwd, file_name + '.app', 'Contents', 'MacOS', true_filename))
            if len(candidates) == 0:
                candidates = glob.glob(
                    os.path.join(file_name + '.app', 'Contents', 'MacOS', true_filename))
            if len(candidates) == 0:
                candidates = glob.glob(
                    os.path.join(cwd, file_name + '.app', 'Contents', 'MacOS', '*'))
            if len(candidates) == 0:
                candidates = glob.glob(os.path.join(file_name + '.app', 'Contents', 'MacOS', '*'))
            if len(candidates) > 0:
                launch_string = candidates[0]
        elif platform == 'win32':
            candidates = glob.glob(os.path.join(cwd, file_name + '.exe'))
            if len(candidates) == 0:
                candidates = glob.glob(file_name + '.exe')
            if len(candidates) > 0:
                launch_string = candidates[0]
        if launch_string is None:
            self._close()
            raise UnityEnvironmentException("Couldn't launch the {0} environment. "
                                            "Provided filename does not match any environments.\n"
                                            "If you haven't done so already, follow the instructions at: "
                                            "https://github.com/beyretb/AnimalAI-Olympics "
                                            .format(true_filename))
        else:
            logger.debug("This is the launch string {}".format(launch_string))
            # Launch Unity environment
            if not docker_training:
                if self.headless:
                    cmd_prefix = ["xvfb-run", "--auto-servernum", "--server-args", "-screen 0 640x480x24",]
                else:
                    cmd_prefix = []

                if self.play:
                    self.proc1 = subprocess.Popen(
                        cmd_prefix + [launch_string, '--port', str(self.port)])
                elif self.inference:
                    self.proc1 = subprocess.Popen(
                        cmd_prefix + [launch_string, '--port', str(self.port), '--inference'])
                else:
                    if self.resolution:
                        self.proc1 = subprocess.Popen(
                            cmd_prefix + [launch_string, '--port', str(self.port), '--resolution', str(self.resolution), '--nArenas',
                             str(self.n_arenas)])
                    else:
                        self.proc1 = subprocess.Popen(
                            cmd_prefix + [launch_string, '--port', str(self.port), '--nArenas', str(self.n_arenas)])

            else:
                """
                Comments for future maintenance:
                    xvfb-run is a wrapper around Xvfb, a virtual xserver where all
                    rendering is done to virtual memory. It automatically creates a
                    new virtual server automatically picking a server number `auto-servernum`.
                    The server is passed the arguments using `server-args`, we are telling
                    Xvfb to create Screen number 0 with width 640, height 480 and depth 24 bits.
                    Note that 640 X 480 are the default width and height. The main reason for
                    us to add this is because we'd like to change the depth from the default
                    of 8 bits to 24.
                    Unfortunately, this means that we will need to pass the arguments through
                    a shell which is why we set `shell=True`. Now, this adds its own
                    complications. E.g SIGINT can bounce off the shell and not get propagated
                    to the child processes. This is why we add `exec`, so that the shell gets
                    launched, the arguments are passed to `xvfb-run`. `exec` replaces the shell
                    we created with `xvfb`.
                """
                if self.resolution:
                    docker_ls = ("exec xvfb-run --auto-servernum"
                                 " --server-args='-screen 0 640x480x24'"
                                 " {0} --port {1} --nArenas {2} --resolution {3}").format(launch_string, str(self.port),
                                                                                          str(self.n_arenas),
                                                                                          str(self.resolution))
                else:
                    docker_ls = ("exec xvfb-run --auto-servernum"
                                 " --server-args='-screen 0 640x480x24'"
                                 " {0} --port {1} --nArenas {2}").format(launch_string, str(self.port),
                                                                         str(self.n_arenas))
                self.proc1 = subprocess.Popen(docker_ls,
                                              stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE,
                                              shell=True)
