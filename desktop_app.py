"""
Food Recognition & Nutrition Advisor - simple internship demo.

No dataset download, no training - uses an already-trained Food-101 classifier
(prithivMLmods/Food-101-93M, ~93M params) from Hugging Face for the CNN component,
a local CSV for grounded nutrition facts, and Claude for recipe/diet-advice generation.

Setup (one time):
    pip install -r requirements.txt
    cp .env.example .env        # then paste your Anthropic API key into .env

Run:
    python desktop_app.py

First run downloads the classifier model automatically (one-time, a few hundred MB,
cached by Hugging Face so it won't re-download on future runs).
"""
import csv
import json
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

MODEL_NAME = "prithivMLmods/Food-101-93M"
NUTRITION_CSV = os.path.join(os.path.dirname(__file__), "nutrition_db.csv")
DISPLAY_SIZE = (320, 320)


# ---------------- Nutrition lookup ----------------

def load_nutrition_db(path: str) -> dict:
    db = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            db[row["food"]] = {
                "calories_per_100g": float(row["calories_per_100g"]),
                "protein_g": float(row["protein_g"]),
                "carbs_g": float(row["carbs_g"]),
                "fat_g": float(row["fat_g"]),
            }
    return db


def scaled_nutrition(db: dict, food_label: str, portion_grams: float):
    if food_label not in db:
        return None
    row = db[food_label]
    scale = portion_grams / 100
    return {
        "food": food_label,
        "portion_grams": portion_grams,
        "calories": round(row["calories_per_100g"] * scale, 1),
        "protein_g": round(row["protein_g"] * scale, 1),
        "carbs_g": round(row["carbs_g"] * scale, 1),
        "fat_g": round(row["fat_g"] * scale, 1),
    }


# ---------------- Gen AI: recipe + diet advice ----------------

def generate_recipe_and_advice(client, food_label: str, facts: dict, user_profile: dict) -> dict:
    profile_text = json.dumps(user_profile) if user_profile else "No profile provided - give general advice."

    prompt = f"""You are a nutrition assistant. These nutrition facts are VERIFIED and GROUNDED -
use them exactly as given, do not recalculate or contradict them:

Food: {food_label.replace('_', ' ')}
Portion: {facts['portion_grams']}g
Calories: {facts['calories']} kcal
Protein: {facts['protein_g']}g
Carbs: {facts['carbs_g']}g
Fat: {facts['fat_g']}g

User profile: {profile_text}

Return ONLY valid JSON (no markdown fences, no preamble) matching this schema:
{{
  "recipe": {{
    "ingredients": [string, ...],
    "steps": [string, ...],
    "prep_time_minutes": number
  }},
  "diet_recommendation": {{
    "summary": string,
    "fits_profile": boolean,
    "suggested_tweaks": [string, ...]
  }}
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Could not parse LLM response as JSON", "raw": text}


# ---------------- GUI ----------------

class FoodAdvisorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Food Recognition & Nutrition Advisor")
        self.root.geometry("900x700")

        self.classifier_pipeline = None
        self.nutrition_db = load_nutrition_db(NUTRITION_CSV)
        self.genai_client = None
        self.current_facts = None
        self.current_label = None
        self.current_image_tk = None

        self._build_layout()
        self._load_backend()

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="y", padx=(0, 12))

        self.image_label = ttk.Label(left, text="No image loaded", relief="groove", width=40, anchor="center")
        self.image_label.pack(pady=(0, 8))
        self.image_label.configure(background="#eeeeee")

        self.upload_btn = ttk.Button(left, text="Upload food photo...", command=self.on_upload, state="disabled")
        self.upload_btn.pack(fill="x")

        self.prediction_var = tk.StringVar(value="Prediction: -")
        ttk.Label(left, textvariable=self.prediction_var, font=("Segoe UI", 12, "bold"), wraplength=300).pack(pady=(12, 4))

        self.confidence_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.confidence_var, wraplength=300).pack()

        profile_frame = ttk.LabelFrame(left, text="Your profile", padding=8)
        profile_frame.pack(fill="x", pady=(16, 0))

        ttk.Label(profile_frame, text="Goal:").grid(row=0, column=0, sticky="w")
        self.goal_var = tk.StringVar(value="maintenance")
        ttk.Combobox(profile_frame, textvariable=self.goal_var, state="readonly",
                     values=["maintenance", "weight_loss", "muscle_gain", "diabetic_friendly"]).grid(row=0, column=1, sticky="ew")

        ttk.Label(profile_frame, text="Calorie target:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.calorie_var = tk.IntVar(value=2000)
        ttk.Entry(profile_frame, textvariable=self.calorie_var, width=10).grid(row=1, column=1, sticky="w", pady=(6, 0))

        ttk.Label(profile_frame, text="Portion (g):").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.portion_var = tk.IntVar(value=250)
        ttk.Entry(profile_frame, textvariable=self.portion_var, width=10).grid(row=2, column=1, sticky="w", pady=(6, 0))

        ttk.Label(profile_frame, text="Allergies (comma-sep):").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.allergies_var = tk.StringVar(value="")
        ttk.Entry(profile_frame, textvariable=self.allergies_var, width=20).grid(row=3, column=1, sticky="ew", pady=(6, 0))
        profile_frame.columnconfigure(1, weight=1)

        self.generate_btn = ttk.Button(left, text="Generate recipe & diet advice",
                                        command=self.on_generate, state="disabled")
        self.generate_btn.pack(fill="x", pady=(12, 0))

        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        nutrition_frame = ttk.LabelFrame(right, text="Nutrition facts", padding=8)
        nutrition_frame.pack(fill="x")
        self.nutrition_var = tk.StringVar(value="Upload a photo to see nutrition facts.")
        ttk.Label(nutrition_frame, textvariable=self.nutrition_var, justify="left").pack(anchor="w")

        output_frame = ttk.LabelFrame(right, text="Recipe & diet advice", padding=8)
        output_frame.pack(fill="both", expand=True, pady=(12, 0))

        self.output_text = tk.Text(output_frame, wrap="word", font=("Segoe UI", 10))
        scrollbar = ttk.Scrollbar(output_frame, command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)
        self.output_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.output_text.insert("1.0", "Recipe and diet advice will appear here after you generate them.")
        self.output_text.configure(state="disabled")

        self.status_var = tk.StringVar(value="Loading classifier model (first run downloads it, please wait)...")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w").pack(side="bottom", fill="x")

    def _load_backend(self):
        def load():
            try:
                from transformers import AutoImageProcessor, SiglipForImageClassification

                processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
                model = SiglipForImageClassification.from_pretrained(MODEL_NAME)
                model.eval()
                self.classifier_pipeline = (processor, model)

                self._set_status("Classifier ready.")
                self.root.after(0, lambda: self.upload_btn.configure(state="normal"))

                try:
                    import anthropic
                    api_key = os.environ.get("ANTHROPIC_API_KEY")
                    if not api_key:
                        from dotenv import load_dotenv
                        load_dotenv()
                        api_key = os.environ.get("ANTHROPIC_API_KEY")
                    if not api_key:
                        self._set_status("Classifier ready. No ANTHROPIC_API_KEY found - recipe/advice generation disabled.")
                        return
                    self.genai_client = anthropic.Anthropic(api_key=api_key)
                    self._set_status("Ready.")
                except Exception as e:
                    self._set_status(f"Classifier ready. Gen AI setup failed: {e}")

            except Exception as e:
                self._set_status(f"Error loading classifier: {e}")

        threading.Thread(target=load, daemon=True).start()

    def _set_status(self, text: str):
        self.root.after(0, lambda: self.status_var.set(text))

    def on_upload(self):
        if self.classifier_pipeline is None:
            messagebox.showerror("Not ready", "The classifier hasn't finished loading yet. Check the status bar.")
            return

        path = filedialog.askopenfilename(title="Select a food photo", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path:
            return

        image = Image.open(path).convert("RGB")

        display_image = image.copy()
        display_image.thumbnail(DISPLAY_SIZE)
        self.current_image_tk = ImageTk.PhotoImage(display_image)
        self.image_label.configure(image=self.current_image_tk, text="")

        self._set_status("Classifying...")

        def classify():
            import torch

            processor, model = self.classifier_pipeline
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=1)[0]

            top_probs, top_indices = probs.topk(3)
            predictions = [
                (model.config.id2label[i.item()], p.item())
                for i, p in zip(top_indices, top_probs)
            ]
            top_label, top_confidence = predictions[0]
            facts = scaled_nutrition(self.nutrition_db, top_label, self.portion_var.get())

            def update_ui():
                self.current_label = top_label
                self.current_facts = facts

                self.prediction_var.set(f"Prediction: {top_label.replace('_', ' ').title()}")
                others = ", ".join(f"{l.replace('_', ' ')} ({c:.0%})" for l, c in predictions[1:])
                self.confidence_var.set(f"Confidence: {top_confidence:.1%}  |  Also considered: {others}")

                if facts is None:
                    self.nutrition_var.set(f"No nutrition data available for '{top_label}'.")
                else:
                    self.nutrition_var.set(
                        f"Portion: {facts['portion_grams']}g\n"
                        f"Calories: {facts['calories']:.0f} kcal\n"
                        f"Protein: {facts['protein_g']:.1f} g\n"
                        f"Carbs: {facts['carbs_g']:.1f} g\n"
                        f"Fat: {facts['fat_g']:.1f} g"
                    )
                    if self.genai_client is not None:
                        self.generate_btn.configure(state="normal")

                self._set_status("Ready.")

            self.root.after(0, update_ui)

        threading.Thread(target=classify, daemon=True).start()

    def on_generate(self):
        if self.current_facts is None or self.current_label is None or self.genai_client is None:
            return

        self.generate_btn.configure(state="disabled")
        self._set_status("Asking Claude for a recipe and diet advice...")

        def generate():
            user_profile = {
                "goal": self.goal_var.get(),
                "daily_calorie_target": self.calorie_var.get(),
                "allergies": [a.strip() for a in self.allergies_var.get().split(",") if a.strip()],
            }
            result = generate_recipe_and_advice(self.genai_client, self.current_label, self.current_facts, user_profile)

            def update_ui():
                self.output_text.configure(state="normal")
                self.output_text.delete("1.0", "end")

                if "error" in result:
                    self.output_text.insert("1.0", f"Error: {result['error']}\n\n{result.get('raw', '')}")
                else:
                    lines = [f"Prep time: ~{result['recipe']['prep_time_minutes']} minutes\n", "Ingredients:"]
                    for ing in result["recipe"]["ingredients"]:
                        lines.append(f"  - {ing}")
                    lines.append("\nSteps:")
                    for i, step in enumerate(result["recipe"]["steps"], 1):
                        lines.append(f"  {i}. {step}")
                    lines.append("\nDiet recommendation:")
                    lines.append(result["diet_recommendation"]["summary"])
                    fits = result["diet_recommendation"]["fits_profile"]
                    lines.append(f"\nFits your profile: {'Yes' if fits else 'Not quite'}")
                    if result["diet_recommendation"]["suggested_tweaks"]:
                        lines.append("\nSuggested tweaks:")
                        for tweak in result["diet_recommendation"]["suggested_tweaks"]:
                            lines.append(f"  - {tweak}")
                    self.output_text.insert("1.0", "\n".join(lines))

                self.output_text.configure(state="disabled")
                self.generate_btn.configure(state="normal")
                self._set_status("Ready.")

            self.root.after(0, update_ui)

        threading.Thread(target=generate, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = FoodAdvisorApp(root)
    root.mainloop()
