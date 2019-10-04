# pytorch-a2c-ppo-acktr

## Run a2c, ppo, acktr on the Animal AI Olympics environment
Algorithm expects to find environment in `aai_resources/env` directory and task configs in `aai_resources/default_configs`.
But you can specify different paths using `--env-path` and `--config-dir`. 

Train recurrent PPO for 100kk interaction steps (use `--headless` for headless mode):
```bash
python3 train_aai.py --save-dir pretrained/basic_rnn --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 16 --num-steps 128 --num-mini-batch 4 --log-interval 10 --num-env-steps 100000000 --use-linear-lr-decay --entropy-coef 0.01 -rnn
```

[Kostrikov's README](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)