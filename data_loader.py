"""
Dataset loader for embedding stress testing using MS MARCO dataset.
MS MARCO is ideal for embedding benchmarks as it contains real Bing search queries and passages.
"""

from datasets import load_dataset
from typing import List
import json
import structlog

logger = structlog.get_logger()


def load_test_sentences(num_samples: int = 100000, cache_dir: str = "./data_cache") -> List[str]:
    """
    Load passages from MS MARCO dataset for embedding testing.

    Args:
        num_samples: Number of passages to load
        cache_dir: Directory to cache the dataset

    Returns:
        List of text passages
    """
    logger.info("loading_ms_marco_dataset", num_samples=num_samples)

    # Load MS MARCO passage corpus
    # This contains ~8.8M passages from Bing search
    dataset = load_dataset(
        "microsoft/ms_marco",
        "v2.1",
        split="train",
        cache_dir=cache_dir,
        streaming=True  # Stream to avoid loading entire dataset
    )

    sentences = []
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        # Extract the passage text (MS MARCO v2.1 uses "passages" field)
        # Each item has passages which contain the text
        if "passages" in item and "passage_text" in item["passages"]:
            passages = item["passages"]["passage_text"]
            # Take the first passage or concatenate if multiple
            text = passages[0] if passages else ""
        elif "passage" in item:
            text = item["passage"]
        elif "text" in item:
            text = item["text"]
        else:
            # Fallback: try to find any text field
            text = str(item)

        if text:  # Only add non-empty texts
            sentences.append(text)

        if (i + 1) % 10000 == 0:
            logger.info("loading_progress", passages_loaded=i + 1)

    logger.info("loaded_ms_marco_dataset", num_passages=len(sentences))
    return sentences


def save_test_data(sentences: List[str], output_file: str = "test_sentences.json"):
    """Save test sentences to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(sentences, f, indent=2)
    logger.info("saved_test_data", num_sentences=len(sentences), output_file=output_file)


def load_test_data(input_file: str = "test_sentences.json") -> List[str]:
    """Load test sentences from a JSON file."""
    with open(input_file, "r") as f:
        sentences = json.load(f)
    logger.info("loaded_test_data", num_sentences=len(sentences), input_file=input_file)
    return sentences


if __name__ == "__main__":
    # Test the data loader
    logger.info("testing_ms_marco_data_loader")
    sentences = load_test_sentences(num_samples=100000)

    avg_length = sum(len(s) for s in sentences) / len(sentences)
    logger.info(
        "dataset_statistics",
        total_passages=len(sentences),
        average_length_chars=round(avg_length, 1),
    )

    sample_passages = [sent[:150] for sent in sentences[:3]]
    logger.info("sample_passages", samples=sample_passages)

    # Save for later use
    save_test_data(sentences, "test_sentences.json")
