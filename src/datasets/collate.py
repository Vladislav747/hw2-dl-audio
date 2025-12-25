import logging

import torch
import torch.nn.functional as functional


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
    logger.info(f"[COLLATE] structure dataset_items {dataset_items[0].keys()}")

    spectrograms = [item["spectrogram"] for item in dataset_items]
    text_encodeds = [item["text_encoded"] for item in dataset_items]
    texts = [item["text"] for item in dataset_items]
    audio_paths = [item["audio_path"] for item in dataset_items]
    audios = [item["audio"] for item in dataset_items]
    
    spectrogram_lengths = torch.tensor([spec.shape[-1] for spec in spectrograms], dtype=torch.long)
    text_encoded_lengths = torch.tensor([text_enc.shape[-1] for text_enc in text_encodeds], dtype=torch.long)

    max_time = max(spec.shape[-1] for spec in spectrograms)
    batch_spectrograms = []
    for spec in spectrograms:
        spec = spec.squeeze(0)
        pad_size = max_time - spec.shape[-1]
        padded = functional.pad(spec, (0, pad_size), mode='constant', value=0)
        batch_spectrograms.append(padded)
    batch_spectrograms = torch.stack(batch_spectrograms)
    
    text_encoded_flat = [text_enc.squeeze(0) for text_enc in text_encodeds]
    batch_text_encoded = torch.cat(text_encoded_flat, dim=0)
    
    max_audio_len = max(audio.shape[-1] for audio in audios)
    batch_audios = []
    for audio in audios:
        pad_size = max_audio_len - audio.shape[-1]
        padded_audio = functional.pad(audio, (0, pad_size), mode='constant', value=0)
        batch_audios.append(padded_audio)
    batch_audios = torch.stack(batch_audios)

    return {
        "spectrogram": batch_spectrograms,
        "spectrogram_length": spectrogram_lengths,
        "text_encoded": batch_text_encoded,
        "text_encoded_length": text_encoded_lengths,
        "text": texts,
        "audio_path": audio_paths,
        "audio": batch_audios,
    }
