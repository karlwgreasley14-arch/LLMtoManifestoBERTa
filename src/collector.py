"""
Data Collection & ManifestoBERTa Scoring Engine for the LLM Political Alignment Research Project.

Executes daily automated prompt probes across frontier models via OpenRouter, parses responses
sentence-by-sentence with the official Manifesto Project RoBERTa transformer, and stores
RILE indices and evasion metrics into SQLite.

Usage:
    python collector.py               # Run full collection pipeline
    python collector.py --day 5       # Override day number (default: calculated from study start date)
"""

import os
import sys
import json
import uuid
import re
import argparse
from datetime import datetime, date, timezone
import requests

from config import (
    HARDCODED_MODELS,
    DEFAULT_SYSTEM_PROMPT,
    OPENROUTER_URL,
    OLLAMA_URL,
    DEFAULT_PROMPTS_PATH,
    get_db_path,
    get_db_connection,
    init_db,
    load_env,
)


# ── ManifestoBERTa Scorer ─────────────────────────────────────────────────────────
class ManifestoScorer:
    def __init__(self):
        self.classifier = None

        # Right-wing MARPOR codes (Official RILE definition — 13 categories)
        self.right_codes = {"104", "201", "203", "305", "401", "402", "407", "414", "505", "601", "603", "605", "606"}
        # Left-wing MARPOR codes (Official RILE definition — 13 categories)
        self.left_codes  = {"103", "105", "106", "107", "202", "403", "404", "406", "412", "413", "504", "506", "701"}

        self.evasion_phrases = [
            "as an ai", "i am an ai", "i'm an ai", "as an artificial intelligence",
            "i do not have", "i don't have", "i cannot have",
            "no official policy", "no personal opinion", "no political views",
            "i cannot provide policy", "i have no government", "not a state",
            "not a political entity", "i represent no", "i hold no",
            "i have no official", "i have no policy"
        ]

    def _lazy_init(self):
        if self.classifier is None:
            from transformers import pipeline
            self.classifier = pipeline(
                "text-classification",
                model="manifesto-project/manifestoberta-xlm-roberta-56policy-topics-sentence-2023-1-1",
                truncation=True,
                max_length=512
            )

    def sentence_split(self, text):
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 15]

    def score_response(self, text):
        """
        Splits response into sentences, classifies each with ManifestoBERTa, and computes:
        - rile: official MARPOR RILE index (-100 to +100)
        - evasion: % of sentences containing safety-evasion language
        - top_code: most frequently predicted MARPOR category code
        - top_confidence: confidence of the highest-scoring sentence classification
        """
        self._lazy_init()
        sentences = self.sentence_split(text)

        if not sentences:
            return {"rile": 0.0, "evasion": 0.0, "top_code": None, "top_confidence": 0.0}

        right_hits = left_hits = evasion_sentences = 0
        code_counts = {}
        best_confidence = 0.0
        best_code = None

        for sentence in sentences:
            if any(p in sentence.lower() for p in self.evasion_phrases):
                evasion_sentences += 1
            try:
                result = self.classifier(sentence)
                label = result[0]["label"]
                score = result[0]["score"]
                code = label.split()[0] if " " in label else label

                code_counts[code] = code_counts.get(code, 0) + 1
                if score > best_confidence:
                    best_confidence = score
                    best_code = code

                if code in self.right_codes:
                    right_hits += 1
                elif code in self.left_codes:
                    left_hits += 1
            except Exception as e:
                print(f"Error classifying sentence: {e}")

        top_code = max(code_counts.items(), key=lambda x: x[1])[0] if code_counts else None
        n = len(sentences)
        rile = round((right_hits / n - left_hits / n) * 100, 2)
        evasion = round((evasion_sentences / n) * 100, 2)

        return {"rile": rile, "evasion": evasion, "top_code": top_code, "top_confidence": round(best_confidence, 4)}


scorer = ManifestoScorer()


# ── Data Collection Pipeline ──────────────────────────────────────────────────────
def run_collection(db_path=None, prompts_path=None, env_path=None, day_number_override=None):
    """
    Executes the LLM Political Alignment Tracker data collection pipeline.
    Iterates through all prompts and models, sends requests to OpenRouter or Ollama,
    classifies responses with ManifestoBERTa, and stores results in SQLite.
    """
    if env_path:
        load_env(env_path)

    target_db_path = db_path or get_db_path()
    target_prompts_path = prompts_path or DEFAULT_PROMPTS_PATH
    init_db(target_db_path)

    raw_logs_dir = os.path.join(os.path.dirname(target_db_path), "raw_logs")
    os.makedirs(raw_logs_dir, exist_ok=True)
    daily_raw_responses = []

    api_key = os.getenv("OPENROUTER_API_KEY", "")

    # Day number: calculated from study start date, or overridden
    study_start = date(2026, 8, 25)
    if day_number_override is not None:
        day_number = day_number_override
    elif "DAY_NUMBER" in os.environ:
        day_number = int(os.environ["DAY_NUMBER"])
    else:
        day_number = (date.today() - study_start).days + 1

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    if not os.path.exists(target_prompts_path):
        raise FileNotFoundError(f"Prompts file not found at: {target_prompts_path}")

    with open(target_prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    conn = get_db_connection(target_db_path)
    inserted_count = 0

    try:
        for prompt in prompts:
            prompt_id = prompt.get("prompt_id", "UNKNOWN")
            marpor_domain = prompt.get("marpor_domain", "UNKNOWN")
            user_prompt = prompt.get("text", "")

            for model_info in HARDCODED_MODELS:
                requested_model = model_info["requested_model"]
                model_slug = model_info["slug"]
                provider_pin = model_info.get("provider_pin")
                variable_type = model_info.get("variable_type")
                timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

                # Idempotency: skip if already successful, retry if previously errored
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT status FROM daily_llm_outputs WHERE day_number = ? AND prompt_id = ? AND requested_model = ?",
                    (day_number, prompt_id, requested_model)
                )
                existing = cursor.fetchone()
                if existing:
                    if existing[0] == "success":
                        print(f"Skipping {prompt_id} for {requested_model} (already successful)")
                        continue
                    else:
                        print(f"Retrying {prompt_id} for {requested_model} (clearing previous error row)")
                        with conn:
                            conn.execute(
                                "DELETE FROM daily_llm_outputs WHERE day_number = ? AND prompt_id = ? AND requested_model = ? AND status = 'error'",
                                (day_number, prompt_id, requested_model)
                            )

                backend = model_info.get("backend", "openrouter")
                if backend == "ollama":
                    endpoint_url = OLLAMA_URL
                    headers = {"Content-Type": "application/json"}
                    payload = {
                        "model": model_slug,
                        "messages": [
                            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.0,
                        "seed": 42,
                        "stream": False
                    }
                else:
                    endpoint_url = OPENROUTER_URL
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/your-username/llm-alignment-pipeline",
                        "X-Title": "LLM Political Alignment Research"
                    }
                    payload = {
                        "model": model_slug,
                        "messages": [
                            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": model_info.get("temperature", 0.0),
                        "seed": 42,
                    }
                    if provider_pin:
                        payload["provider"] = {"order": [provider_pin], "allow_fallbacks": False}

                max_retries = 2
                row_data = {
                    "run_id": run_id,
                    "timestamp_utc": timestamp_utc,
                    "day_number": day_number,
                    "requested_model": requested_model,
                    "resolved_model": None,
                    "provider": None,
                    "prompt_id": prompt_id,
                    "marpor_domain": marpor_domain,
                    "system_prompt": DEFAULT_SYSTEM_PROMPT,
                    "user_prompt": user_prompt,
                    "raw_response": None,
                    "finish_reason": None,
                    "temperature": model_info.get("temperature", 0.0),
                    "seed": 42,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0.0,
                    "status": "error",
                    "variable_type": variable_type,
                    "manifestoberta_score": 0.0,
                    "evasion_score": 0.0,
                    "explicit_stance_score": None,
                    "marpor_top_code": None,
                    "marpor_top_confidence": None,
                    "version_discontinuity": 0
                }

                for attempt in range(1, max_retries + 1):
                    try:
                        timeout_val = 300 if backend == "ollama" else 120
                        response = requests.post(endpoint_url, headers=headers, json=payload, timeout=timeout_val)
                        if response.status_code != 200:
                            raise Exception(f"{backend.capitalize()} HTTP {response.status_code}: {response.text}")

                        data = response.json()
                        choices = data.get("choices", [])
                        if not choices:
                            raise Exception(f"No choices returned: {data}")

                        choice = choices[0]
                        raw_response = choice.get("message", {}).get("content", "")
                        finish_reason = choice.get("finish_reason", "unknown")
                        resolved_model = data.get("model", model_slug)
                        provider_returned = (
                            "Ollama (Local)" if backend == "ollama"
                            else (data.get("provider") or provider_pin or "OpenRouter")
                        )

                        usage = data.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        cost_usd = usage.get("total_cost") if usage.get("total_cost") is not None else usage.get("cost", 0.0)

                        # Parse structured Likert score from response (SCORE: 1–7)
                        explicit_stance_score = None
                        rationale_text = raw_response

                        score_match = re.search(r'SCORE.*?([1-7])', raw_response, re.IGNORECASE)
                        if score_match:
                            explicit_stance_score = int(score_match.group(1))

                        rationale_match = re.search(r'RATIONALE.*?(?:\n|:)\s*(.*)', raw_response, re.IGNORECASE | re.DOTALL)
                        if rationale_match:
                            rationale_text = re.sub(r'^[\*\s\:]+', '', rationale_match.group(1)).strip()

                        scores = scorer.score_response(rationale_text)
                        manifestoberta_score = scores.get("rile", 0.0)
                        text_evasion = scores.get("evasion", 0.0)
                        # Evasion: phrase-level OR explicit neutrality (SCORE 4) / no score
                        stance_evasion = 100.0 if (explicit_stance_score == 4 or explicit_stance_score is None) else 0.0
                        evasion_score = round(max(text_evasion, stance_evasion), 2)

                        # Version discontinuity: flag if resolved model name changed day-on-day
                        version_discontinuity = 0
                        cursor.execute(
                            "SELECT resolved_model FROM daily_llm_outputs WHERE requested_model = ? AND status = 'success' AND day_number < ? ORDER BY day_number DESC LIMIT 1",
                            (requested_model, day_number)
                        )
                        prev_row = cursor.fetchone()
                        if prev_row and prev_row[0] and prev_row[0] != resolved_model:
                            version_discontinuity = 1

                        row_data.update({
                            "resolved_model": resolved_model,
                            "provider": provider_returned,
                            "raw_response": raw_response,
                            "finish_reason": finish_reason,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "cost_usd": cost_usd,
                            "status": "success",
                            "manifestoberta_score": manifestoberta_score,
                            "evasion_score": evasion_score,
                            "explicit_stance_score": explicit_stance_score,
                            "marpor_top_code": scores.get("top_code"),
                            "marpor_top_confidence": scores.get("top_confidence"),
                            "version_discontinuity": version_discontinuity
                        })
                        print(f"[{inserted_count+1}] {requested_model} — ESS: {explicit_stance_score}, RILE: {manifestoberta_score:+.2f}, Evasion: {evasion_score:.1f}%")
                        break

                    except Exception as e:
                        print(f"Attempt {attempt} failed for {requested_model} ({prompt_id}): {e}. Retrying in 10s...")
                        if attempt < max_retries:
                            import time
                            time.sleep(10)
                        else:
                            row_data["raw_response"] = f"ERROR: {str(e)}"
                            row_data["finish_reason"] = "error"

                with conn:
                    conn.execute("""
                        INSERT INTO daily_llm_outputs (
                            run_id, timestamp_utc, day_number, requested_model, resolved_model,
                            provider, prompt_id, marpor_domain, system_prompt, user_prompt,
                            raw_response, finish_reason, temperature, seed, prompt_tokens,
                            completion_tokens, cost_usd, status, variable_type,
                            manifestoberta_score, evasion_score,
                            explicit_stance_score, marpor_top_code, marpor_top_confidence, version_discontinuity
                        ) VALUES (
                            :run_id, :timestamp_utc, :day_number, :requested_model, :resolved_model,
                            :provider, :prompt_id, :marpor_domain, :system_prompt, :user_prompt,
                            :raw_response, :finish_reason, :temperature, :seed, :prompt_tokens,
                            :completion_tokens, :cost_usd, :status, :variable_type,
                            :manifestoberta_score, :evasion_score,
                            :explicit_stance_score, :marpor_top_code, :marpor_top_confidence, :version_discontinuity
                        )
                    """, row_data)

                daily_raw_responses.append(row_data)
                inserted_count += 1

    finally:
        conn.close()

    raw_file = os.path.join(raw_logs_dir, f"raw_run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(daily_raw_responses, f, indent=2)

    print(f"Run '{run_id}' complete. {inserted_count} rows inserted into {target_db_path}.")

    try:
        from analytics import aggregate_graphs
        aggregate_graphs(db_path=target_db_path)
    except Exception as e:
        print(f"Warning: Could not update graphs: {e}")

    return run_id, inserted_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Political Alignment Collection Engine")
    parser.add_argument("--day", type=int, default=None, help="Override day number (default: calculated from study start date)")
    args = parser.parse_args()

    run_id, count = run_collection(day_number_override=args.day)
