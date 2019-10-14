# pytorch-a2c-ppo-acktr

### Обучение

Чтобы запустить обучение используйте train_aai.py:

Пример аргументов с которыми мы обычно запускаем наши последние модели:
```
python3 train_aai.py -sd pretrained/meta-arch -et ivm3-mixed-configs --config-dir aai_resources/mixed_configs/ --extra-obs angle pos speed visited r_prev a_prev time -fs 6 -or -0.01 --algo ppo --use-gae --lr 3e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 16 --num-steps 128 --num-mini-batch 4 --log-interval 10 --num-env-steps 50000000 --use-linear-lr-decay --entropy-coef 0.01
```
Многие параметры такие же как и у Костирикова. Но советую все-таки посмотреть в файлик `a2c_ppo_ackrt/aai_arguments.py`

Советую обратить внимание на следующие аргументы:
* `-sd, --save-dir` общая папка для экспериментов которые вы хотитите сравнивать
* `-et, --experiment-tag` имя конкретного эксперимента
* `--config-dir` папка с конфигами на которых будет обучаться модель(обращайте внимание на структуру подпапок в ней)

Соответственно в папку _sd/et/_ будут сохранятся чекпоинты модели(нейросетка, оптимайзер, число апдейтов).
В папку _sd/summaries/et_ будет сохранятся статистика эксперимента для tensorboard, чтобы можно было запустить
```
tensorboard --logdir=/sd/summaries
```
и сравнить относительно друг-друга разные эксперименты.

### Перезапуск
Если что-то свалилось умерло, продолжить обучение можно так:

```python3 train_aai.py --restart sd/et/checkpoint-you-want.pt```

Все остальные аргументы скрипт возьмет из файла _sd/et/train_args.json_
Соответственно если вдруг есть желание продолжить обучение с какими-то другими аргументами придется менять их прямо в _train_args.json_

### Обучение в docker
Чтобы запустить обучение в докере. Нужно будет собрать контейнер из папки `train_docker/`:
```docker build -f train_docker/Dockerfile --tag=train:latest .```
А потом запустить скрипт `run.sh` с теме же аргументами что и `train_aai.py`:
```
./train_docker/run.sh --all --the --same -arguments
```
Правда придется не забыть указать `--docker-training`

### Протестировать обученную модель
Используем `enjoy_aai.py`:
```
python3 enjoy_aai.py path-to-dir/checkpoint-name.pt --config-dir configs/to/test-on -n 160 --cuda -d 0.00
```
Аргументы можно посмотреть прямо в файле, там просто.

### Отладка конфигов
Скрипт `aai_interact.py` позволяет играть в конфиги самому.
Никаких аргументов не принимает. Нужно поменять пути в начале файла(строки 11,12). А почему? А потому что всем лень:) Все равно из pycharm'а трудозатраты одинаковые.



[Kostrikov's README](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)