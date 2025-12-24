# ASR

- Архитектура DeepSpeech2 
- ASR вариант CTC 



#### Установка пакетов

```
source .venv/bin/activate && pip install -r requirements.txt 2>&1 | tail
```

#### Запуск скрипта

```
python3 train.py -cn=baseline datasets=onebatchtest trainer.n_epochs=1 trainer.override=True writer=cometml 2>&1 | head -250
```

Складываем результаты экспериментов в comet_logs
