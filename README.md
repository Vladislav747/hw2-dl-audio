# ASR

- Архитектура DeepSpeech2 
- ASR вариант CTC 



#### Установка пакетов и настройка виртуального окружения

```
source .venv/bin/activate && pip install -r requirements.txt 2>&1 | tail
```


#### Запуск baseline model(для сравнения с deepspeech2)

Запуск dataset onebatch test

```bash
python3 train.py -cn=baseline \
  datasets=onebatchtest \
  trainer.n_epochs=10 \
  trainer.override=True \
  writer=cometml \
  writer.run_name="baseline_onebatchtest" \
  2>&1
```

Запуск dataset example

```bash
python3 train.py -cn=baseline \
  datasets=example \
  trainer.n_epochs=10 \
  trainer.override=True \
  writer=cometml \
  writer.run_name="baseline_example" \
  2>&1
```

Запуск dataset example_onebatchtest

```bash
python3 train.py -cn=baseline \
  datasets=example_onebatchtest \
  trainer.n_epochs=10 \
  trainer.override=True \
  writer=cometml \
  writer.run_name="baseline_example_onebatchtest" \
  2>&1
```

#### Запуск deepspeech2 на dataset onebatchtest

Простой запуск
```bash
python3 train.py -cn=deepspeech2 \
  datasets=onebatchtest \
  trainer.n_epochs=1 \
  trainer.override=True \
  writer=cometml \
  writer.run_name="deepspeech2_example_onebatchtest" \
  2>&1
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
  writer=cometml \
  writer.run_name="deepspeech2_onebatchtest_params" 
```

Запуск с кастомным именем эксперимента в CometML
```bash
python3 train.py -cn=deepspeech2 \
  datasets=onebatchtest \
  trainer.n_epochs=1 \
  trainer.override=True \
  writer=cometml \
  writer.run_name="deepspeech2_onebatchtest" \
  2>&1
```
#### Запуск deepspeech2 на dataset example

```bash
python3 train.py -cn=deepspeech2 \
  datasets=example \
  trainer.n_epochs=1 \
  trainer.override=True \
  writer=cometml \
  writer.run_name="deepspeech2_example" \
  2>&1
```

#### Запуск deepspeech2 на dataset train_clean_360

```bash
python3 train.py -cn=deepspeech2 \
  datasets=train_clean_360 \
  trainer.n_epochs=50 \
  trainer.override=True \
  writer=cometml \
  writer.run_name="deepspeech2_train_clean_360" \
  2>&1
```

#### Как запустить inference.py

Скрипт для инференса модели на датасете и сохранения предсказаний.

```bash
source .venv/bin/activate && python3 inference.py \
  -cn=inference \
  datasets=onebatchtest \
  inferencer.from_pretrained="saved/testing/model_best.pth" \
  inferencer.save_path="predictions_onebatchtest"
```

Параметры:
- `model` - конфигурация модели (deepspeech2, baseline и т.д.)
- `datasets` - конфигурация датасета (custom_dir, example и т.д.)
- `inferencer.checkpoint_path` - путь к чекпоинту модели
- `inferencer.save_path` - путь для сохранения предсказаний (по умолчанию: `data/saved/predictions`)

Предсказания сохраняются в формате: `{UtteranceID}.txt` в папке `inferencer.save_path`.

#### Как запустить calc_metrics - пример вызова

Скрипт для подсчета метрик WER/CER на основе предсказаний и транскрипций.

Скрипт работает только с локальными папками(он не скачивает папки откуда то)

Скрипту нужны либы поэтому используем с виртуальным окружением(см установка пакетов)

```bash
source .venv/bin/activate && python3 calc_metrics.py \
--transcriptions_dir "calc_metrics_examples/transcriptions_dir_example" \
--predictions_dir "calc_metrics_examples/predictions_dir_example"
```

```
Параметры:
- `--transcriptions_dir` - путь к папке с транскрипциями
- `--predictions_dir` - путь к папке с предсказаниями модели
```


#### Как посмотреть отчет

- Все экперименты сохраняются в .cometml-runs и в конце эксперименты мы берем ссылку и выполняем команду ниже(нужно чтобы было)


пример 
```
comet upload /Users/vlad/Documents/Web/hse-dl-audio/.cometml-runs/e9yetmli2svog1gk44lqvdu79y1legx1.zip
```


#### Демо ноутбук для Google Colab

Для запуска проекта в Google Colab используйте `demo_notebook.ipynb`:

который содержится в корневой директории проекта

**Что делает notebook:**
- Клонирует репозиторий
- Устанавливает все зависимости
- Запускает быстрый тест на onebatchtest датасете
- Показывает примеры запуска полного обучения и inference
- Демонстрирует просмотр результатов

Репозиторий - https://github.com/Vladislav747/hw2-dl-audio



#### Отчет по работе

Решил выбрать архитекуру DeepSpeechV2 - она мне показалось простой и понятной

Какие сложности возникали?
   - Нельзя было просто так выкачать датасет train-100 или тем более train-360 через реализованные механизмы - поэтому я выкачивал их локально и переделывал запуск librispeech_dataset чтобы если уже есть файл он его не скачивал снова а только разаархивировал(если не хочется скачивать файл) - чтобы работало локальное скачивание надо положить файл архива в datasets/librispeech - сама разархивация не долгая и не сильно напряжная по ресурсам)
   - При обучении train-100 возникали проблемы с nan на опредленном шаге 100 ввел 



#### Использованные источники

https://github.com/sooftware/deepspeech2