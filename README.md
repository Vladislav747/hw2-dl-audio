# ASR

- Архитектура DeepSpeech2 
- ASR вариант CTC 



#### Установка пакетов

```
source .venv/bin/activate && pip install -r requirements.txt 2>&1 | tail
```

#### Запуск скрипта onebatch test

```
python3 train.py -cn=baseline datasets=onebatchtest trainer.n_epochs=10 trainer.override=True writer=cometml 2>&1 | tail -50
```

#### Запуск скрипта example


```
python3 train.py -cn=baseline datasets=example trainer.n_epochs=10 trainer.override=True writer=cometml 2>&1 | tail -50
```

```
python3 train.py -cn=baseline datasets=example_onebatchtest trainer.n_epochs=10 trainer.override=True writer=cometml 2>&1 | tail -50
```

#### Запуск deepspeech2

Простой запуск
```bash
python3 train.py -cn=deepspeech2 \
  datasets=onebatchtest \
  trainer.n_epochs=1 \
  trainer.override=True \
  writer=cometml \
  2>&1 | head -50000000
```

Запуск с параметрами
```bash
python3 train.py -cn=deepspeech2 \
  datasets=onebatchtest \
  model.n_feats=128 \
  model.n_tokens=28 \
  model.num_rnn_layers=3 \
  model.hidden_size=512 \
  model.rnn_dropout=0.1 \
  model.input_dim=32 \
  trainer.n_epochs=50 \
  trainer.override=True \
  writer=cometml
```

#### Как запустить calc_metrics - пример вызова

Скрипт работает только с локальными папками(он не скачивает папки откуда то)

Скрипту нужны либы поэтому используем с виртуальным окружением(см установка пакетов)

```
source .venv/bin/activate && python3 calc_metrics.py \
--transcriptions_dir "calc_metrics_examples/transcriptions_dir_example" \
--predictions_dir "calc_metrics_examples/predictions_dir_example"
```


### Отчет по работе

Решил выбрать архитекуру DeepSpeechV2 - она мне показалось простой и понятной

Какие сложности возникали?
Нельзя было просто так выкачать датасет train-100 или тем более train-360 через реализованные механизмы - поэтому я выкачивал их локально и переделывал запуск под себя



#### Использованные источники

https://github.com/sooftware/deepspeech2