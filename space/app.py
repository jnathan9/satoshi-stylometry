"""
Satoshi Stylometry - Is this Satoshi?
A BERT-based classifier that detects Satoshi Nakamoto's writing style.
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Load model
MODEL_ID = "thestalwart/satoshi-stylometry"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

EXAMPLES = [
    # Actual Satoshi text (from bitcointalk)
    ["""If you don't believe me or don't understand, I don't have time to try to convince you, sorry."""],
    # Actual Satoshi text (whitepaper style)
    ["""The system is secure as long as honest nodes collectively control more CPU power than any cooperating group of attacker nodes. The proof-of-work chain is the solution to the synchronization problem, and to knowing what the globally shared view is without having to trust anyone."""],
    # Non-Satoshi (Hal Finney)
    ["""Bitcoin seems to be a very promising idea. I like the idea of basing security on the assumption that the CPU power of honest participants outweighs that of the attacker. It is a very modern notion that exploits the power of the long tail."""],
    # Non-Satoshi (generic crypto discussion)
    ["""The fundamental problem with proof of stake is that it doesn't actually require any real-world resources to be expended. This means that there's no physical anchor for the security of the system, unlike proof of work which requires electricity and hardware."""],
    # Craig Wright style
    ["""I am Satoshi Nakamoto. I created Bitcoin. The evidence is clear and I have the keys to prove it. Anyone who disagrees is simply wrong and doesn't understand the technology I invented."""],
]


def predict(text):
    if not text or len(text.strip()) < 10:
        return {}, "Please enter a longer text (at least a few sentences)."

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).numpy()[0]

    satoshi_prob = float(probs[1])
    not_satoshi_prob = float(probs[0])

    # Format result
    label_probs = {
        "Satoshi": satoshi_prob,
        "Not Satoshi": not_satoshi_prob,
    }

    # Generate verdict
    if satoshi_prob > 0.9:
        verdict = f"**Very likely Satoshi** ({satoshi_prob:.1%} confidence)\n\nThis text strongly matches Satoshi Nakamoto's distinctive writing style."
    elif satoshi_prob > 0.7:
        verdict = f"**Probably Satoshi** ({satoshi_prob:.1%} confidence)\n\nThis text has significant stylistic similarities to Satoshi's known writings."
    elif satoshi_prob > 0.3:
        verdict = f"**Uncertain** ({satoshi_prob:.1%} Satoshi probability)\n\nThe model can't confidently determine whether this was written by Satoshi."
    elif satoshi_prob > 0.1:
        verdict = f"**Probably not Satoshi** ({not_satoshi_prob:.1%} confidence)\n\nThis text doesn't strongly match Satoshi's writing style."
    else:
        verdict = f"**Very unlikely Satoshi** ({not_satoshi_prob:.1%} confidence)\n\nThis text does not match Satoshi Nakamoto's writing style."

    word_count = len(text.split())
    verdict += f"\n\n*Analysis based on {word_count} words. Longer texts give more reliable results.*"

    return label_probs, verdict


with gr.Blocks(
    title="Satoshi Stylometry",
    theme=gr.themes.Base(
        primary_hue="amber",
        secondary_hue="stone",
        neutral_hue="stone",
    ),
    css="""
    .main-header { text-align: center; margin-bottom: 0.5em; }
    .main-header h1 { font-size: 2.2em; margin-bottom: 0; }
    .subtitle { text-align: center; color: #666; margin-top: 0; margin-bottom: 1.5em; font-size: 1.1em; }
    .verdict-box { font-size: 1.1em; }
    """
) as demo:
    gr.HTML("""
        <div class="main-header">
            <h1>Is this Satoshi?</h1>
        </div>
        <p class="subtitle">
            A BERT model trained on every known Satoshi Nakamoto writing to detect his distinctive style.<br>
            99.1% accuracy on held-out test data.
        </p>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Paste text to analyze",
                placeholder="Enter a paragraph or more of text to check if it matches Satoshi's writing style...",
                lines=8,
            )
            submit_btn = gr.Button("Analyze", variant="primary", size="lg")

        with gr.Column(scale=1):
            label_output = gr.Label(label="Prediction", num_top_classes=2)
            verdict_output = gr.Markdown(label="Verdict", elem_classes=["verdict-box"])

    gr.Examples(
        examples=EXAMPLES,
        inputs=text_input,
        outputs=[label_output, verdict_output],
        fn=predict,
        cache_examples=False,
    )

    submit_btn.click(fn=predict, inputs=text_input, outputs=[label_output, verdict_output])
    text_input.submit(fn=predict, inputs=text_input, outputs=[label_output, verdict_output])

    gr.HTML("""
        <div style="margin-top: 2em; padding-top: 1em; border-top: 1px solid #ddd; color: #888; font-size: 0.9em; text-align: center;">
            <p><strong>How it works:</strong> Fine-tuned ModernBERT on 572 Satoshi writings (BitcoinTalk posts, mailing list emails)
            vs. 1,546 non-Satoshi writings from the same era and community. The model detects writing style, not identity.</p>
            <p><a href="https://github.com/thestalwart/satoshi-stylometry" target="_blank">GitHub</a> |
            Model: <a href="https://huggingface.co/thestalwart/satoshi-stylometry" target="_blank">thestalwart/satoshi-stylometry</a></p>
        </div>
    """)

demo.launch()
