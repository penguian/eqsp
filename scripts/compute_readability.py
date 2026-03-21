"""
Compute aggregated readability metrics for a list of Markdown files using Vale.
"""

import json
import subprocess
import sys


def get_metrics(file_list):
    """
    Get aggregated raw metrics for a list of files using 'vale ls-metrics'.
    """
    total_metrics = {
        "words": 0,
        "sentences": 0,
        "syllables": 0,
        "complex_words": 0,
        "characters": 0,
    }

    for file_path in file_list:
        try:
            result = subprocess.run(
                ["vale", "ls-metrics", file_path],
                capture_output=True,
                text=True,
                check=True,
            )
            metrics_dict = json.loads(result.stdout)
            for key in total_metrics:
                if key in metrics_dict:
                    total_metrics[key] += metrics_dict[key]
        except subprocess.CalledProcessError as err:
            print(f"Error processing {file_path}: {err}", file=sys.stderr)
            continue

    return total_metrics


def calculate_scores(metrics_data):
    """
    Calculate readability scores based on aggregate metrics.
    """
    words_count = metrics_data["words"]
    sentences_count = metrics_data["sentences"]
    syllables_count = metrics_data["syllables"]
    complex_count = metrics_data["complex_words"]
    chars_count = metrics_data["characters"]

    if words_count == 0 or sentences_count == 0:
        return {}

    # Flesch-Kincaid Grade Level
    fkgl = (0.39 * (words_count / sentences_count)) + (
        11.8 * (syllables_count / words_count)
    ) - 15.59

    # Flesch Reading Ease
    fre = (
        206.835
        - (1.015 * (words_count / sentences_count))
        - (84.6 * (syllables_count / words_count))
    )

    # Gunning Fog Index
    gfi = 0.4 * ((words_count / sentences_count) + 100 * (complex_count / words_count))

    # Automated Readability Index (ARI)
    ari = (
        4.71 * (chars_count / words_count)
        + 0.5 * (words_count / sentences_count)
        - 21.43
    )

    return {
        "Flesch-Kincaid Grade Level": round(fkgl, 1),
        "Flesch Reading Ease": round(fre, 1),
        "Gunning Fog Index": round(gfi, 1),
        "Automated Readability Index": round(ari, 1),
        "Total Words": words_count,
        "Total Sentences": sentences_count,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_readability.py <label> <file1> <file2> ...")
        sys.exit(1)

    LABEL = sys.argv[1]
    FILES = sys.argv[2:]

    METRICS = get_metrics(FILES)
    SCORES = calculate_scores(METRICS)

    print(f"\n### Readability Report: {LABEL}")
    print(json.dumps(SCORES, indent=4))
