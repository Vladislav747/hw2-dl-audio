from pathlib import Path
import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        data = []
        audio_dir = Path(audio_dir)
        transcription_dir = Path(transcription_dir) if transcription_dir else None
        
        if not audio_dir.exists():
            raise ValueError(f"Audio directory does not exist: {audio_dir}")
        
        for path in audio_dir.iterdir():
            if not path.is_file():
                continue
                
            if path.suffix.lower() not in [".mp3", ".wav", ".flac", ".m4a"]:
                continue
            
            entry = {}
            try:
                audio_info = torchaudio.info(str(path))
                entry["audio_len"] = audio_info.num_frames / audio_info.sample_rate
                entry["path"] = str(path.absolute())
                
                if transcription_dir and transcription_dir.exists():
                    transc_path = transcription_dir / (path.stem + ".txt")
                    if transc_path.exists():
                        with transc_path.open(encoding='utf-8') as f:
                            entry["text"] = f.read().strip()
                    else:
                        continue
                else:
                    entry["text"] = ""
                
                if entry.get("text") or transcription_dir is None:
                    data.append(entry)
            except Exception as e:
                print(f"Warning: Could not process {path}: {e}")
                continue
                
        if len(data) == 0:
            raise ValueError(f"No valid audio files found in {audio_dir}")
            
        super().__init__(data, *args, **kwargs)
