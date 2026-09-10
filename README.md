# LLM → ManifestoBERTa Political Alignment Pipeline

Data collection backend for my thesis — probes GPT, Claude, Gemini, Grok and a local Llama control across 7 MARPOR policy domains daily, scores responses with [ManifestoBERTa](https://huggingface.co/manifesto-project/manifestoberta-xlm-roberta-56policy-topics-sentence-2023-1-1) and stores RILE indices + evasion metrics in SQLite.

Live data dashboard → **[research.tinknet.co.uk](https://research.tinknet.co.uk)**

---

### Setup

```bash
cp .env.example .env
# add your OPENROUTER_API_KEY

pip install -r requirements.txt
```

Needs Ollama running locally with `llama3.2:3b` pulled for the control variable.

### Run

```bash
python src/collector.py          # run today's collection
python src/analytics.py          # dataset summary
python src/analytics.py --audit  # sentence-level classification breakdown
```

### Test

```bash
python -m pytest tests/ -v
```

### Structure

```
src/collector.py   — collection + ManifestoBERTa scoring
src/config.py      — models, DB schema, MARPOR domains
src/analytics.py   — aggregation + diagnostics
data/prompts.json  — prompt bank (7 domains)
tests/             — integration tests
```
