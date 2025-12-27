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


#### Как запустить calc_metrics - пример вызова

Скрипт работает только с локальными папками(он не скачивает папки откуда то)

Скрипту нужны либы поэтому используем с виртуальным окружением(см установка пакетов)

```
source .venv/bin/activate && python3 calc_metrics.py \
--transcriptions_dir "calc_metrics_examples/transcriptions_dir_example" \
--predictions_dir "calc_metrics_examples/predictions_dir_example"
```
