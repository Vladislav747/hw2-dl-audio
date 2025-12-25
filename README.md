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
