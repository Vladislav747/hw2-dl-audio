import logging

import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    logger = logging.getLogger('collate_fn')
    logger.info(f"[COLLATE] len dataset_items {len(dataset_items)} items")
    logger.info(f"[COLLATE] structure dataset_items { (dataset_items[0].keys())}")


    spectrograms = [item["spectrogram"] for item in dataset_items]
    text_encodeds = [item["text_encoded"] for item in dataset_items]
    texts = [item["text"] for item in dataset_items]
    audio_paths = [item["audio_path"] for item in dataset_items]
    audios = [item["audio"] for item in dataset_items]


    return {
        "spectrogram": spectrograms,
        "spectrogram_length": len(spectrograms),
        "text_encoded": text_encodeds,
        "text_encoded_length": len(text_encodeds),
        "text": texts,
        "audio_path": audio_paths,
        "audio": audios,
    }
