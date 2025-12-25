import editdistance


def calc_cer(target_text: str, pred_text: str):
    """
    Character Error Rate
    """
    if target_text is None:
        target_text = ""
    if pred_text is None:
        pred_text = ""

    if len(target_text) == 0:
        return 1.0

    return editdistance.eval(target_text, pred_text) / len(target_text)


def calc_wer(target_text: str, pred_text: str) -> float:
    """
    Word Error Rate
    """
    if target_text is None:
        target_text = ""
    if pred_text is None:
        pred_text = ""

    target_words = target_text.split()
    pred_words = pred_text.split()

    if len(target_words) == 0:
        return 1.0

    return editdistance.eval(pred_words, target_words) / len(target_words)