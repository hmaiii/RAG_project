from sentence_transformers import SentenceTransformer, util
import json

# Load model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Load data
with open("evaluation_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def compute_semantic_mrr_and_recall3(data, threshold=0.7):
    mrr_total = 0
    recall_at_3_total = 0
    valid_samples = 0

    for item in data:
        reference = item["reference"]
        retrieved_docs = item["predictions"]

        reference_emb = model.encode(reference, convert_to_tensor=True)
        retrieved_embs = model.encode(retrieved_docs, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(reference_emb, retrieved_embs)[0]

        sorted_indices = similarities.argsort(descending=True)
        found = False

        for rank, idx in enumerate(sorted_indices):
            sim = similarities[idx].item()
            if sim >= threshold:
                mrr_total += 1 / (rank + 1)
                if rank < 3:
                    recall_at_3_total += 1
                found = True
                break

        if found:
            valid_samples += 1

    mrr = mrr_total / valid_samples if valid_samples else 0
    recall_at_3 = recall_at_3_total / valid_samples if valid_samples else 0

    print(" Semantic Retrieval Metrics:")
    print(f" MRR: {mrr:.4f}")
    print(f" Recall@3: {recall_at_3:.4f}")

compute_semantic_mrr_and_recall3(data)
