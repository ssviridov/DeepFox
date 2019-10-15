from catalyst.rl import registry

from src.env import AnimalEnvWrapper
from src.critic import AnimalActionCritic

registry.Environment(AnimalEnvWrapper)
registry.Agent(AnimalActionCritic)
