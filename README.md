# Food Recognition & Nutrition Advisor — Simple Demo

A minimal, self-contained version for demoing quickly: **no dataset download, no training.**

- **CNN component:** uses an already-trained Food-101 classifier from Hugging Face
  (`prithivMLmods/Food-101-93M`, 93M params) — downloads once (~a few hundred MB), then it's
  cached locally by Hugging Face and won't re-download.
- **Gen AI component:** Claude generates a recipe and personalized diet advice, grounded in a local
  nutrition CSV (so calorie/macro numbers are never hallucinated).
- **GUI:** plain `tkinter` desktop window — no browser, no local web server.

## Setup (one time)

```bash
cd food_advisor_demo
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and paste your key from https://console.anthropic.com/settings/keys:
```
ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
python3 desktop_app.py
```

First launch downloads the classifier model (one-time). After that it loads from the local
Hugging Face cache instantly. Upload a food photo, and you'll see:

1. Predicted food class + confidence (top-3)
2. Grounded nutrition facts scaled to your chosen portion size
3. A button to generate a Claude-written recipe + personalized diet advice

## Files

```
food_advisor_demo/
├── desktop_app.py       # everything: classifier, nutrition lookup, Gen AI call, GUI
├── nutrition_db.csv     # grounded calorie/macro facts for all 101 Food-101 classes
├── requirements.txt
└── .env.example
```

## Notes for your internship writeup

- Classification uses transfer learning / a pretrained model fine-tuned specifically on Food-101 —
  worth explaining that this sidesteps needing your own GPU/training run while still giving a real,
  working CNN classifier (not a mock).
- Nutrition facts are approximate estimates, not USDA-verified — worth flagging, and a natural
  "future work" item is swapping in a live USDA FoodData Central API lookup.
- The Gen AI layer is deliberately restricted to generating *language* (recipe steps, advice
  phrasing) rather than numbers — this is a grounding/RAG-style design choice worth calling out.
