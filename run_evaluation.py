import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score

# Load data
with open("evaluation_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Number of methods (4 methods in your example)
NUM_METHODS = 4

# Evaluation Functions
def compute_bleu(pred, ref):
    smooth = SmoothingFunction().method1
    return sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth)

def compute_rouge(pred, ref):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(ref, pred)['rougeL'].fmeasure

def run_bert_score(predictions, references, lang="fa"):
    _, _, F1 = bert_score.score(predictions, references, lang=lang, verbose=False)
    return F1

# Hallucination Score Function
def estimate_hallucination_score(bleu, rouge_l, bert_f1, precision, correctness):
    # Weighting the automatic evaluations (BLEU, ROUGE, BERTScore)
    weighted_auto_score = (0.4 * bleu + 0.3 * rouge_l + 0.3 * bert_f1)  # Suggested weights for automatic evals
    
    # Weighting the human evaluations (precision and correctness)
    weighted_human_score = (0.5 * precision + 0.5 * correctness)  # Suggested weights for human evaluations
    
    # Combining both to estimate hallucination
    hallucination_score = (1 - weighted_auto_score) * (1 - weighted_human_score) * 100
    
    return round(hallucination_score, 2)

# Initialize results
bleu_results = [[] for _ in range(NUM_METHODS)]
rouge_results = [[] for _ in range(NUM_METHODS)]
bert_inputs = [[] for _ in range(NUM_METHODS)]
bert_refs = [[] for _ in range(NUM_METHODS)]

# Collect all predictions and references per method
for item in data:
    ref = item['reference']
    preds = item['predictions']
    for i in range(len(preds)):
        pred = preds[i]
        if "پاسخی موجود نیست" in pred or "پاسخی تولید نشده" in pred:
            continue
        bleu_results[i].append(compute_bleu(pred, ref))
        rouge_results[i].append(compute_rouge(pred, ref))
        bert_inputs[i].append(pred)
        bert_refs[i].append(ref)

# Final Results
print("\n Evaluation Results for Each Method:\n")
for i in range(NUM_METHODS):
    bleu = sum(bleu_results[i]) / len(bleu_results[i]) if bleu_results[i] else 0
    rouge = sum(rouge_results[i]) / len(rouge_results[i]) if rouge_results[i] else 0
    if bert_inputs[i]:
        bert_f1 = run_bert_score(bert_inputs[i], bert_refs[i])
        bert_f1_avg = float(bert_f1.mean())
    else:
        bert_f1_avg = 0
    
    # Example precision and correctness values (can be replaced with actual data if available)
    precision = 0.85  # Example precision (replace with actual calculation if available)
    correctness = 0.90  # Example correctness (replace with actual calculation if available)
    
    hallucination_score = estimate_hallucination_score(bleu, rouge, bert_f1_avg, precision, correctness)
    
    # Print Evaluation Results
    print(f"Method {i+1}:")
    print(f"  - BLEU:  {bleu:.4f}")
    print(f"  - ROUGE-L: {rouge:.4f}")
    print(f"  - BERTScore F1: {bert_f1_avg:.4f}")
    print(f"  - Hallucination Score: {hallucination_score}\n")
