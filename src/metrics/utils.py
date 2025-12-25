# Based on seminar materials

# Don't forget to support cases when target_text == ''

from Levenshtein import distance

def calc_cer(target_text, predicted_text) -> float:
    """
    Character Error Rate
    """
    if target_text is None:
        target_text = ""
    if predicted_text is None:
        predicted_text = ""

    
    if len(target_text) == 0:
        return 1.0
    
    # Calculate Levenshtein distance
    edit_distance = distance(target_text, predicted_text)
    # Calculate CER
    cer = edit_distance / len(target_text)
    return cer


def calc_wer(target_text, predicted_text) -> float:
    """
    Word Error Rate
    """
    if target_text is None:
        target_text = ""
    if predicted_text is None:
        predicted_text = ""

    target_words = target_text.split()
    predicted_words = predicted_text.split()

    # Если вдруг ошибка и нет слов в target_text
    if len(target_words) == 0:
        return 1.0
    
    substitutions = sum(1 for target, predicted in zip(target_words, predicted_words) if target != predicted)
    deletions = len(target_words) - len(predicted_words)
    insertions = len(predicted_words) - len(target_words)
    total_words = len(target_words)

    wer = (substitutions + deletions + insertions) / total_words
    return wer
