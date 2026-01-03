# ASR

- Архитектура DeepSpeech2 
- ASR вариант CTC 



#### Установка пакетов и настройка виртуального окружения

```
source .venv/bin/activate && pip install -r requirements.txt 2>&1 | tail
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


#### Оценка модели на test-clean
```bash
python3 inference.py \
  -cn=inference \
  datasets=test_clean \
  inferencer.from_pretrained="saved/deepspeech2_360/model_best.pth" \
  inferencer.save_path="predictions_test_clean"
```

#### Оценка модели на test-other

```bash
python3 inference.py \
  -cn=inference \
  datasets=test_other \
  inferencer.from_pretrained="saved/deepspeech2_augs_strong_360/model_best.pth" \
  inferencer.save_path="predictions_test_other"
```

#### Как запустить inference.py

Скрипт для инференса модели на датасете и сохранения предсказаний.

**Примеры путей к сохраненным моделям (`inferencer.from_pretrained`):**

Модели содержать сохранненые веса моделей

DeepSpeech2 модели:
- `saved/deepspeech2_onebatchtest/model_best.pth`
- `saved/deepspeech2_onebatchtest_params/model_best.pth`
- `saved/deepspeech2_example/model_best.pth`
- `saved/deepspeech2_example_onebatchtest/model_best.pth`
- `saved/deepspeech2_train_clean_360/model_best.pth`

**Пример запуска:**

```bash
source .venv/bin/activate && python3 inference.py \
  -cn=inference \
  datasets=test_other \
  inferencer.from_pretrained="saved/deepspeech2_360/model_best.pth" \
  inferencer.save_path="predictions_deepspeech2_360"
```

```
Параметры:
- `model` - конфигурация модели (deepspeech2, baseline и т.д.)
- `datasets` - конфигурация датасета (custom_dir, example и т.д.)
- `inferencer.from_pretrained` - путь к чекпоинту модели (см. примеры выше)
- `inferencer.save_path` - путь для сохранения предсказаний (по умолчанию: `predictions`)
```

#### Как запустить calc_metrics - пример вызова

Скрипт для подсчета метрик WER/CER на основе предсказаний и транскрипций.

Скрипт работает только с локальными папками(он не скачивает папки откуда то - вам нужно загрузить репозиторий и положить папки в корень и указать до них путь)

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
- Показывает примеры запуска обучения и inference
- Демонстрирует просмотр результатов

Репозиторий - https://github.com/Vladislav747/hw2-dl-audio


#### Отчет по работе


Решил выбрать архитекуру DeepSpeechV2 - она мне показалось простой и понятной

Все метрики и эксперименты тут(публичный доступ)
https://www.comet.com/melo-malo/pytorch-template-asr-example?shareable=wVd1kUkDsbAUMhY6Q7sFS9LUN

##### Логи обучения: как быстро обучалась сеть

Финально я обучал две модели с augs_strong и без на train_clean_360

конфигурации  deepspeech2_360 и deepspeech2_augs_strong_360 - их веса я прикладываю в ссылке 


##### Как вы обучали финальную модель

deepspeech2_360 - на train_clean_360 не все эпохи которые хотел(10 эпох)
deepspeech2_augs_strong_360 - с аугментациями на train_clean_360 не все эпохи которые хотел(5 эпох)
Пробовал другие конфигурации но метрики были хуже


##### Что пробовали, какие эксперименты были сделаны, их анализ

Пробовал различные датасеты(train-100-clean, onebatchtest, train-360-clean, разные параметры, более количество эпох

Результаты прогона

```python
python3 inference.py \
  -cn=inference \
  datasets=test_clean \
  inferencer.from_pretrained="saved/deepspeech2_360/model_best.pth" \
  inferencer.save_path="predictions_test_clean"
```

```
 test_CER_(Argmax): 0.7635895211785624
    test_WER_(Argmax): 1.0135311097073487
    test_CER_(BeamSearch): 0.7680725542526388
    test_WER_(BeamSearch): 1.0219167681526289
    test_CER_(LM_BeamSearch): 0.7635895211785624
    test_WER_(LM_BeamSearch): 1.0135311097073487
```


```python
python3 inference.py \
  -cn=inference \
  datasets=test_other \
  inferencer.from_pretrained="saved/deepspeech2_augs_strong_360/model_best.pth" \
  inferencer.save_path="predictions_test_other"
```


```
    test_CER_(Argmax): 0.7662772396176233
    test_WER_(Argmax): 1.0302392879307147
    test_CER_(BeamSearch): 0.7695601733454066
    test_WER_(BeamSearch): 1.0431835665173377
    test_CER_(LM_BeamSearch): 0.7662772396176233
    test_WER_(LM_BeamSearch): 1.0302392879307147
```

##### Основные сложности, с которыми вы столкнулись
   - Нельзя было просто так выкачать датасет train-100 или тем более train-360 через реализованные механизмы - поэтому я выкачивал их локально и переделывал запуск librispeech_dataset чтобы если уже есть файл он его не скачивал снова а только разаархивировал(если не хочется скачивать файл) - чтобы работало локальное скачивание надо положить файл архива в datasets/librispeech - сама разархивация не долгая и не сильно напряжная по ресурсам)
   - При обучении train-100 возникали проблемы с nan на опредленном шаге 100 ввел 
   - Краткие сроки задания - не удалось достичь хороших метрик

##### Что сработало / не сработало
- Рад что заработала архитектура DeepSpeech
- Рад что пайплайны проходят
- Получил опыт - прошу по возможности пересмотреть строгость оценки(потрачено немеренно сил и времени)
- Не получилось достичь нормальных метрик CER, WER возможно не все понял


#### Использованные источники

https://github.com/sooftware/deepspeech2
Семинары DL Audio