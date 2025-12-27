#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import sys

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder import CTCTextEncoder


def load_transcriptions(transcriptions_dir: Path):
    transcriptions = {}
    for transcription_file in transcriptions_dir.glob("*.txt"):
        #.stem возвращает имя файла без расширения
        file_id = transcription_file.stem
        with open(transcription_file, "r", encoding="utf-8") as file_opened:
            transcriptions[file_id] = file_opened.read().strip()
    return transcriptions


def load_predictions(predictions_dir: Path):
    predictions = {}
    for prediction_file in predictions_dir.glob("*.txt"):
        file_id = prediction_file.stem
        with open(prediction_file, "r") as file_opened:
            predictions[file_id] = file_opened.read().strip()
    return predictions



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--transcriptions_dir", type=str, required=True)
    output_path = "calc_metrics_output"
    args = parser.parse_args()


    transcriptions_dir = Path(args.transcriptions_dir)
    predictions_dir = Path(args.predictions_dir)
    output_path = Path(output_path)

    if not transcriptions_dir.exists():
        print(f"Папка с транскрипциями не найдена: {transcriptions_dir}")
        return
    if not predictions_dir.exists():
        print(f"Папка с предсказаниями не найдена: {predictions_dir}")
        return


    text_encoder = CTCTextEncoder()

    transcriptions = load_transcriptions(transcriptions_dir)
    predictions = load_predictions(predictions_dir)

    wers = []
    cers = []

    common_ids = set(transcriptions.keys()) & set(predictions.keys())

    if len(common_ids) == 0:
        print("Ошибка: нет общих файлов")
        return

    for file_id in common_ids:
        target_text = text_encoder.normalize_text(transcriptions[file_id])
        pred_text = text_encoder.normalize_text(predictions[file_id])

        wer = calc_wer(target_text, pred_text)
        cer = calc_cer(target_text, pred_text)

        wers.append(wer)
        cers.append(cer)

    avg_wer = sum(wers) / len(wers)
    avg_cer = sum(cers) / len(cers)

    metrics = {
        "wer": avg_wer,
        "cer": avg_cer,
        "num_files": len(common_ids)
    }

    output_file = output_path / "results.json"
    with open(output_file, "w", encoding="utf-8") as file_opened:
        json.dump(metrics, file_opened)
    print(f"Результаты сохранены в {output_path}")
    
    


if __name__ == "__main__":
    main()