import os
import json
from typing import List, Dict, Union


class LibriSpeechParser:
    """
    Handles parsing and organization of the LibriSpeech dataset with variable directory structures.
    Supports train-clean-100, dev-clean, and test-clean subsets.
    """

    def __init__(self):
        self.dataset_structure = {
            'train': 'P:/PycharmProjects/pythonProject1/dataset',
            'dev': 'P:/PycharmProjects/pythonProject1/devDataset',
            'test': 'P:/PycharmProjects/pythonProject1/testDataset'
        }
        self.extensions = ['.flac', '.wav']

    def _find_audio_files(self, root_dir: str) -> List[str]:
        """Recursively find all audio files in nested directories"""
        audio_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.extensions):
                    audio_paths.append(os.path.join(root, file))
        return audio_paths

    def _extract_speaker_id(self, file_path: str, root_dir: str) -> str:
        """Extract speaker ID from file path (handles variable directory structures)"""
        rel_path = os.path.relpath(file_path, root_dir)
        parts = rel_path.split(os.sep)

        # Handle standard structure: speaker_id/chapter_id/utterance.flac
        if len(parts) >= 3 and parts[0].isdigit():
            return parts[0]

        # Fallback: use parent directory name as speaker ID
        parent_dir = os.path.dirname(file_path)
        return os.path.basename(parent_dir)

    def parse_split(self, split: str) -> List[Dict[str, str]]:
        """
        Parse a data split (train/dev/test) and return metadata

        Args:
            split: One of 'train', 'dev', or 'test'

        Returns:
            List of dictionaries with keys: 'audio_path', 'speaker_id'
        """
        if split not in self.dataset_structure:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'dev', or 'test'")

        root_dir = self.dataset_structure[split]
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory not found: {root_dir}")

        audio_files = self._find_audio_files(root_dir)
        metadata = []

        for audio_path in audio_files:
            speaker_id = self._extract_speaker_id(audio_path, root_dir)
            metadata.append({
                'audio_path': audio_path,
                'speaker_id': speaker_id
            })

        return metadata

    def save_metadata(self, metadata: List[Dict[str, str]], output_path: str):
        """Save metadata to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    def load_metadata(self, metadata_path: str) -> List[Dict[str, str]]:
        """Load metadata from JSON file"""
        with open(metadata_path, 'r') as f:
            return json.load(f)

    def get_speaker_stats(self, metadata: List[Dict[str, str]]) -> Dict[str, Union[int, List[str]]]:
        """Calculate statistics about speakers in metadata"""
        speakers = [item['speaker_id'] for item in metadata]
        unique_speakers = sorted(set(speakers))
        return {
            'total_utterances': len(metadata),
            'unique_speakers': len(unique_speakers),
            'speaker_ids': unique_speakers
        }


# Usage example (can be placed in separate test file)
if __name__ == "__main__":
    parser = LibriSpeechParser()

    # Parse all splits
    train_metadata = parser.parse_split('train')
    dev_metadata = parser.parse_split('dev')
    test_metadata = parser.parse_split('test')

    # Save metadata
    parser.save_metadata(train_metadata, 'train_metadata.json')
    parser.save_metadata(dev_metadata, 'dev_metadata.json')
    parser.save_metadata(test_metadata, 'test_metadata.json')

    # Generate statistics
    train_stats = parser.get_speaker_stats(train_metadata)
    print(f"Train statistics: {train_stats}")