"""MultiWOZ dataset loading and preprocessing."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset


@dataclass
class MultiWOZDialogue:
    """Container for MultiWOZ dialogue data."""

    dialogue_id: str
    turns: List[Dict]
    goal: Dict

    @staticmethod
    def _extract_column(value, idx):
        if isinstance(value, list):
            if idx < len(value):
                return value[idx]
            return None
        if isinstance(value, dict):
            return {
                key: MultiWOZDialogue._extract_column(sub_value, idx)
                for key, sub_value in value.items()
            }
        return value

    @classmethod
    def _normalize_turns(cls, turns_obj) -> List[Dict]:
        if isinstance(turns_obj, list):
            normalized = []
            for turn in turns_obj:
                normalized.append(cls._normalize_single_turn(turn))
            return normalized
        if isinstance(turns_obj, dict):
            lengths = [len(v) for v in turns_obj.values() if isinstance(v, list)]
            if not lengths:
                raise ValueError("Turns dict has no list columns")
            size = lengths[0]
            if any(length != size for length in lengths):
                raise ValueError("Inconsistent column lengths in turns dict")
            normalized = []
            for idx in range(size):
                turn = {key: cls._extract_column(value, idx) for key, value in turns_obj.items()}
                normalized.append(cls._normalize_single_turn(turn))
            return normalized
        raise ValueError(f"Unsupported turns data type: {type(turns_obj)}")

    @staticmethod
    def _normalize_single_turn(turn: Dict) -> Dict:
        if not isinstance(turn, dict):
            return {
                "speaker": None,
                "text": None,
                "frames": [],
                "raw_turn": turn,
            }

        speaker = turn.get("speaker")
        if isinstance(speaker, int):
            speaker = "SYSTEM" if speaker == 1 else "USER"
        elif isinstance(speaker, str) and speaker.isdigit():
            speaker = "SYSTEM" if int(speaker) == 1 else "USER"
        turn["speaker"] = speaker

        utterance = turn.get("utterance")
        if utterance is not None:
            turn.setdefault("text", utterance)
        turn.setdefault("text", turn.get("text"))

        frames = turn.get("frames")
        if isinstance(frames, dict):
            lengths = [len(v) for v in frames.values() if isinstance(v, list)]
            size = lengths[0] if lengths else 0
            normalized_frames = []
            for idx in range(size):
                frame = {
                    key: MultiWOZDialogue._extract_column(value, idx)
                    for key, value in frames.items()
                }
                if isinstance(frame.get("actions"), type(None)):
                    frame["actions"] = []
                elif isinstance(frame.get("actions"), dict):
                    frame["actions"] = [frame["actions"]]
                normalized_frames.append(frame)
            turn["frames"] = normalized_frames
        elif isinstance(frames, list):
            normalized_frames = []
            for frame in frames:
                if isinstance(frame, dict):
                    if frame.get("actions") is None:
                        frame["actions"] = []
                    elif isinstance(frame.get("actions"), dict):
                        frame["actions"] = [frame["actions"]]
                normalized_frames.append(frame)
            turn["frames"] = normalized_frames
        else:
            turn["frames"] = []

        if "dialogue_acts" not in turn and "dialogue_act" in turn:
            turn["dialogue_acts"] = turn.get("dialogue_act")

        return turn

    @classmethod
    def from_dict(cls, data: Dict) -> "MultiWOZDialogue":
        turns_normalized = cls._normalize_turns(data.get("turns", []))

        return cls(
            dialogue_id=data.get("dialogue_id", ""),
            turns=turns_normalized,
            goal=data.get("goal", {})
        )


def load_multiwoz_dataset(
    data_path: Optional[str] = None,
    split: str = "train",
    use_huggingface: bool = True
) -> List[MultiWOZDialogue]:
    """
    Load MultiWOZ dataset from file or HuggingFace.
    
    Args:
        data_path: Path to local JSON file (if not using HuggingFace)
        split: Dataset split ("train", "validation", "test")
        use_huggingface: Whether to load from HuggingFace datasets
    
    Returns:
        List of MultiWOZDialogue objects
    """
    if use_huggingface:
        print(f"[{time.strftime('%H:%M:%S')}] Loading MultiWOZ dataset from HuggingFace (split: {split})...")
        start_time = time.time()
        
        # This call can be slow if:
        # 1. Dataset not cached (downloading) - network I/O bottleneck
        # 2. Running custom dataset script (trust_remote_code=True generates splits) - CPU compute
        # 3. Processing/parsing raw data - CPU compute
        # Note: After first load, dataset is cached at ~/.cache/huggingface/datasets/
        # Subsequent runs should be faster unless cache is cleared
        dataset = load_dataset("multi_woz_v22", split=split, trust_remote_code=True)
        
        load_time = time.time() - start_time
        print(f"[{time.strftime('%H:%M:%S')}] Dataset loaded in {load_time:.2f}s, processing {len(dataset)} dialogues...")
        
        process_start = time.time()
        dialogues = []
        for idx, item in enumerate(dataset):
            if (idx + 1) % 1000 == 0:
                elapsed = time.time() - process_start
                print(f"[{time.strftime('%H:%M:%S')}]   Processed {idx + 1}/{len(dataset)} dialogues... ({elapsed:.2f}s elapsed)")
            dialogue = MultiWOZDialogue.from_dict(item)
            dialogues.append(dialogue)
        
        process_time = time.time() - process_start
        total_time = time.time() - start_time
        print(f"[{time.strftime('%H:%M:%S')}] Finished processing {len(dialogues)} dialogues")
        print(f"  Timing: load_dataset={load_time:.2f}s, processing={process_time:.2f}s, total={total_time:.2f}s")
        return dialogues
    else:
        if data_path is None:
            raise ValueError("data_path must be provided when not using HuggingFace")
        import json
        with open(data_path, "r") as f:
            data = json.load(f)
        dialogues = []
        for dialogue_id, dialogue_data in data.items():
            dialogue_data["dialogue_id"] = dialogue_id
            dialogue = MultiWOZDialogue.from_dict(dialogue_data)
            dialogues.append(dialogue)
        return dialogues


def get_dialogue_by_id(
    dialogues: List[MultiWOZDialogue],
    dialogue_id: str
) -> Optional[MultiWOZDialogue]:
    """Retrieve specific dialogue by ID."""
    for dialogue in dialogues:
        if dialogue.dialogue_id == dialogue_id:
            return dialogue
    return None


def filter_dialogues_by_domain(
    dialogues: List[MultiWOZDialogue],
    domain: str
) -> List[MultiWOZDialogue]:
    """Filter dialogues by domain (restaurant, hotel, etc.)."""
    filtered = []
    for dialogue in dialogues:
        if domain in dialogue.goal:
            filtered.append(dialogue)
    return filtered

