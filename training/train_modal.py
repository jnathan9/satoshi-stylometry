"""
Train a ModernBERT classifier to distinguish Satoshi from non-Satoshi text.
Runs on Modal with GPU.
"""

import modal

app = modal.App("satoshi-stylometry")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "scikit-learn",
        "pandas",
    )
)

volume = modal.Volume.from_name("satoshi-stylometry-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/data": volume},
)
def train():
    import json
    import torch
    import numpy as np
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    # Load data
    print("Loading data...")
    with open("/data/train.json") as f:
        train_data = json.load(f)
    with open("/data/val.json") as f:
        val_data = json.load(f)
    with open("/data/test.json") as f:
        test_data = json.load(f)
    with open("/data/golden.json") as f:
        golden_data = json.load(f)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}, Golden: {len(golden_data)}")

    # Create HF datasets
    train_ds = Dataset.from_dict({
        "text": [d["text"] for d in train_data],
        "label": [d["label"] for d in train_data],
    })
    val_ds = Dataset.from_dict({
        "text": [d["text"] for d in val_data],
        "label": [d["label"] for d in val_data],
    })
    test_ds = Dataset.from_dict({
        "text": [d["text"] for d in test_data],
        "label": [d["label"] for d in test_data],
    })
    golden_ds = Dataset.from_dict({
        "text": [d["text"] for d in golden_data],
        "label": [d["label"] for d in golden_data],
    })

    # Load model and tokenizer
    model_name = "answerdotai/ModernBERT-base"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "not_satoshi", 1: "satoshi"},
        label2id={"not_satoshi": 0, "satoshi": 1},
    )

    # Tokenize
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)
    golden_ds = golden_ds.map(tokenize, batched=True)

    # Compute class weights for balanced training
    n_satoshi = sum(1 for d in train_data if d["label"] == 1)
    n_not_satoshi = sum(1 for d in train_data if d["label"] == 0)
    total = n_satoshi + n_not_satoshi
    weight_satoshi = total / (2 * n_satoshi) if n_satoshi > 0 else 1.0
    weight_not_satoshi = total / (2 * n_not_satoshi) if n_not_satoshi > 0 else 1.0
    class_weights = torch.tensor([weight_not_satoshi, weight_satoshi]).cuda()
    print(f"Class weights: not_satoshi={weight_not_satoshi:.3f}, satoshi={weight_satoshi:.3f}")

    # Custom trainer with class weights
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    # Training args
    training_args = TrainingArguments(
        output_dir="/data/model",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        fp16=True,
        report_to="none",
        save_total_limit=2,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\n=== TRAINING ===")
    trainer.train()

    # Evaluate on test set
    print("\n=== TEST SET EVALUATION ===")
    test_results = trainer.evaluate(test_ds)
    print(f"Test results: {test_results}")

    # Evaluate on golden set
    print("\n=== GOLDEN SET EVALUATION ===")
    golden_results = trainer.evaluate(golden_ds)
    print(f"Golden results: {golden_results}")

    # Get per-sample predictions on golden set
    print("\n=== PER-SAMPLE GOLDEN PREDICTIONS ===")
    golden_preds = trainer.predict(golden_ds)
    logits = golden_preds.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_labels = np.argmax(logits, axis=-1)
    true_labels = np.array([d["label"] for d in golden_data])

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    print(f"\nConfusion Matrix:")
    print(f"  Predicted:  not_satoshi  satoshi")
    print(f"  not_satoshi:    {cm[0][0]:>5d}    {cm[0][1]:>5d}")
    print(f"  satoshi:        {cm[1][0]:>5d}    {cm[1][1]:>5d}")

    # Per-sample results
    results = []
    for i, d in enumerate(golden_data):
        results.append({
            "text_preview": d["text"][:100],
            "true_label": "satoshi" if d["label"] == 1 else "not_satoshi",
            "pred_label": "satoshi" if pred_labels[i] == 1 else "not_satoshi",
            "satoshi_prob": float(probs[i][1]),
            "correct": bool(pred_labels[i] == d["label"]),
            "source": d.get("source", ""),
            "author": d.get("author", ""),
        })

    # Save results
    with open("/data/golden_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    n_correct = sum(1 for r in results if r["correct"])
    n_total = len(results)
    print(f"\nGolden accuracy: {n_correct}/{n_total} = {n_correct/n_total:.1%}")

    # Breakdown by class
    sat_results = [r for r in results if r["true_label"] == "satoshi"]
    nsat_results = [r for r in results if r["true_label"] == "not_satoshi"]
    sat_correct = sum(1 for r in sat_results if r["correct"])
    nsat_correct = sum(1 for r in nsat_results if r["correct"])
    print(f"  Satoshi: {sat_correct}/{len(sat_results)} = {sat_correct/len(sat_results):.1%}")
    print(f"  Not Satoshi: {nsat_correct}/{len(nsat_results)} = {nsat_correct/len(nsat_results):.1%}")

    # Show misclassified
    print(f"\nMisclassified samples:")
    for r in results:
        if not r["correct"]:
            print(f"  [{r['true_label']} -> {r['pred_label']}] (prob={r['satoshi_prob']:.3f}) {r['text_preview'][:80]}")

    # Show confidence distribution
    sat_probs = [r["satoshi_prob"] for r in results if r["true_label"] == "satoshi"]
    nsat_probs = [r["satoshi_prob"] for r in results if r["true_label"] == "not_satoshi"]
    print(f"\nSatoshi confidence (should be high):")
    print(f"  Mean: {np.mean(sat_probs):.3f}, Median: {np.median(sat_probs):.3f}")
    print(f"  Min: {np.min(sat_probs):.3f}, Max: {np.max(sat_probs):.3f}")
    print(f"\nNon-Satoshi confidence (should be low):")
    print(f"  Mean: {np.mean(nsat_probs):.3f}, Median: {np.median(nsat_probs):.3f}")
    print(f"  Min: {np.min(nsat_probs):.3f}, Max: {np.max(nsat_probs):.3f}")

    # Save full evaluation report
    report = {
        "test_results": test_results,
        "golden_results": golden_results,
        "confusion_matrix": cm.tolist(),
        "golden_accuracy": n_correct / n_total,
        "satoshi_accuracy": sat_correct / len(sat_results),
        "non_satoshi_accuracy": nsat_correct / len(nsat_results),
        "satoshi_prob_mean": float(np.mean(sat_probs)),
        "non_satoshi_prob_mean": float(np.mean(nsat_probs)),
    }
    with open("/data/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Save model
    trainer.save_model("/data/model/best")
    tokenizer.save_pretrained("/data/model/best")
    print("\nModel saved to /data/model/best")

    volume.commit()
    print("Volume committed.")

    return report


@app.local_entrypoint()
def main():
    import json
    result = train.remote()
    print("\n=== FINAL RESULTS ===")
    print(json.dumps(result, indent=2))
