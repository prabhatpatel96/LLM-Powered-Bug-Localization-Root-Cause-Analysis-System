
def localization_accuracy(predicted, actual):
    return int(set(predicted) == set(actual))

def hallucination_rate(output_text, ground_truth_fix):
    return int(ground_truth_fix not in output_text)
