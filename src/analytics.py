"""
Analytics, Time-Series Graph Aggregator, and Statistical Auditor for the LLM Political Alignment Research Project.

Usage:
    python analytics.py                 # Prints full dataset telemetry & summary table
    python analytics.py --audit         # Inspects a random prompt output sentence-by-sentence
    python analytics.py --update-graphs # Recompiles static/graph_data.json
"""

import os
import sys
import json
import sqlite3
import argparse
import shutil

from config import (
    MODEL_LABELS,
    ALL_DOMAIN_KEYS,
    GRAPH_JSON_PATH,
    get_db_path,
)


def get_db_connection(db_path=None):
    target = db_path or get_db_path()
    if not os.path.exists(target):
        raise FileNotFoundError(f"Database not found at {target}")
    
    # Use temporary snapshot to prevent SQLite WAL locking during heavy read analytics
    snapshot = "/tmp/analytics_db_snapshot.db"
    shutil.copy2(target, snapshot)
    conn = sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def aggregate_graphs(db_path=None, output_json=None):
    """
    Reads all successful observations and compiles rolling weekly RILE indices,
    safety evasion percentages, and 7-domain breakdowns into graph_data.json for Chart.js.
    """
    target_db = db_path or get_db_path()
    target_out = output_json or GRAPH_JSON_PATH
    
    conn = get_db_connection(target_db)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT day_number, requested_model, marpor_domain, manifestoberta_score, evasion_score, explicit_stance_score, status
            FROM daily_llm_outputs
            WHERE status = 'success'
        """)
        rows = cur.fetchall()
    finally:
        conn.close()

    data_by_day = {}
    all_domains = ALL_DOMAIN_KEYS

    for row in rows:
        day = row["day_number"]
        model = row["requested_model"]
        domain = row["marpor_domain"]
        
        if day not in data_by_day:
            data_by_day[day] = {}
        if model not in data_by_day[day]:
            data_by_day[day][model] = {
                "rile_sum": 0, "evasion_sum": 0, "ess_sum": 0, "count": 0, "ess_count": 0,
                "domains": {d: {"rile_sum": 0, "ess_sum": 0, "count": 0, "ess_count": 0} for d in all_domains}
            }
            
        rile = row["manifestoberta_score"] or 0.0
        evasion = row["evasion_score"] or 0.0
        ess = row["explicit_stance_score"]
        
        data_by_day[day][model]["rile_sum"] += rile
        data_by_day[day][model]["evasion_sum"] += evasion
        data_by_day[day][model]["count"] += 1
        
        if ess is not None:
            data_by_day[day][model]["ess_sum"] += ess
            data_by_day[day][model]["ess_count"] += 1
        
        if domain in data_by_day[day][model]["domains"]:
            data_by_day[day][model]["domains"][domain]["rile_sum"] += rile
            data_by_day[day][model]["domains"][domain]["count"] += 1
            if ess is not None:
                data_by_day[day][model]["domains"][domain]["ess_sum"] += ess
                data_by_day[day][model]["domains"][domain]["ess_count"] += 1

    days = sorted(list(data_by_day.keys()))
    
    models = set()
    for d in days:
        models.update(data_by_day[d].keys())

    graph_data = {
        "days": days,
        "models": {}
    }

    for model in models:
        label = MODEL_LABELS.get(model, model)
        graph_data["models"][label] = {
            "rile": [], 
            "evasion": [],
            "ess": [],
            "domains": {d: [] for d in all_domains},
            "domains_ess": {d: [] for d in all_domains}
        }
        
        for d in days:
            if model in data_by_day[d]:
                m_data = data_by_day[d][model]
                count = m_data["count"]
                ess_count = m_data["ess_count"]
                
                rile = round(m_data["rile_sum"] / count, 2) if count > 0 else None
                evasion_pct = round(m_data["evasion_sum"] / count, 2) if count > 0 else None
                ess = round(m_data["ess_sum"] / ess_count, 2) if ess_count > 0 else None
                
                graph_data["models"][label]["rile"].append(rile)
                graph_data["models"][label]["evasion"].append(round(evasion_pct, 1) if evasion_pct is not None else None)
                graph_data["models"][label]["ess"].append(ess)
                
                for dom in all_domains:
                    d_count = m_data["domains"][dom]["count"]
                    d_ess_count = m_data["domains"][dom]["ess_count"]
                    d_rile = round(m_data["domains"][dom]["rile_sum"] / d_count, 2) if d_count > 0 else None
                    d_ess = round(m_data["domains"][dom]["ess_sum"] / d_ess_count, 2) if d_ess_count > 0 else None
                    
                    graph_data["models"][label]["domains"][dom].append(d_rile)
                    graph_data["models"][label]["domains_ess"][dom].append(d_ess)
            else:
                graph_data["models"][label]["rile"].append(None)
                graph_data["models"][label]["evasion"].append(None)
                graph_data["models"][label]["ess"].append(None)
                for dom in all_domains:
                    graph_data["models"][label]["domains"][dom].append(None)
                    graph_data["models"][label]["domains_ess"][dom].append(None)

    os.makedirs(os.path.dirname(os.path.abspath(target_out)), exist_ok=True)
    with open(target_out, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"Graph data generated successfully at {target_out}")


def show_summary(db_path=None):
    """Prints a formatted statistical summary table across models and observations."""
    conn = get_db_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                COUNT(*) as total_rows,
                COALESCE(SUM(cost_usd), 0.0) as total_cost,
                MIN(timestamp_utc) as first_run,
                MAX(timestamp_utc) as last_run,
                COUNT(DISTINCT day_number) as total_days
            FROM daily_llm_outputs
            WHERE status = 'success'
        """)
        summary = cur.fetchone()
        
        print("\n" + "=" * 90)
        print("  📊 LLM POLITICAL ALIGNMENT RESEARCH — TELEMETRY SUMMARY")
        print("=" * 90)
        print(f"  Total Observations: {summary['total_rows']:,}")
        print(f"  Total Days Recorded: {summary['total_days']} days")
        print(f"  Date Range:         {summary['first_run']} -> {summary['last_run']}")
        print(f"  Cumulative Cost:    ${summary['total_cost']:.4f} USD")
        print("-" * 90)
        
        print(f"  {'Model':<16} | {'Samples':<8} | {'Avg ESS':<7} | {'Avg RILE':<10} | {'Avg Evasion':<12} | {'Cost ($)':<10}")
        print("  " + "-" * 86)
        
        cur.execute("""
            SELECT 
                requested_model,
                COUNT(*) as count,
                AVG(explicit_stance_score) as avg_ess,
                AVG(manifestoberta_score) as avg_rile,
                AVG(evasion_score) as avg_evasion,
                SUM(cost_usd) as cost
            FROM daily_llm_outputs
            WHERE status = 'success'
            GROUP BY requested_model
            ORDER BY avg_rile DESC
        """)
        for r in cur.fetchall():
            m_label = MODEL_LABELS.get(r['requested_model'], r['requested_model'])
            avg_ess = f"{r['avg_ess']:.2f}" if r['avg_ess'] is not None else "N/A"
            avg_r = f"{r['avg_rile']:+.2f}" if r['avg_rile'] is not None else "N/A"
            avg_e = f"{r['avg_evasion']:.2f}%" if r['avg_evasion'] is not None else "N/A"
            cost = f"${r['cost']:.4f}"
            print(f"  {m_label:<16} | {r['count']:<8} | {avg_ess:<7} | {avg_r:<10} | {avg_e:<12} | {cost:<10}")
            
        print("=" * 90 + "\n")
    finally:
        conn.close()


def audit_sample(db_path=None):
    """Pulls a random observation and performs a sentence-level ManifestoBERTa classification breakdown."""
    from collector import scorer
    
    conn = get_db_connection(db_path)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT day_number, requested_model, user_prompt, raw_response,
                   manifestoberta_score, evasion_score, prompt_id, marpor_domain
            FROM daily_llm_outputs 
            WHERE status = 'success' AND (manifestoberta_score < -30 OR manifestoberta_score > 30 OR evasion_score > 10)
            ORDER BY RANDOM() LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            print("No records found matching audit criteria.")
            return

        print("\n" + "=" * 78)
        print(f"  AUDIT SAMPLE: {MODEL_LABELS.get(row['requested_model'], row['requested_model'])} (Day {row['day_number']})")
        print(f"  Domain: {row['marpor_domain']} | Prompt ID: {row['prompt_id']}")
        print(f"  Prompt: \"{row['user_prompt']}\"")
        print(f"  Recorded RILE: {row['manifestoberta_score']:+.2f} | Evasion: {row['evasion_score']:.1f}%")
        print("=" * 78)
        
        scorer._lazy_init()
        sentences = scorer.sentence_split(row["raw_response"])
        
        for i, sentence in enumerate(sentences):
            print(f"\n[Sentence {i+1}] {sentence}")
            s_lower = sentence.lower()
            if any(p in s_lower for p in scorer.evasion_phrases):
                print("  🚩 Evasion Trigger Detected")
            try:
                res = scorer.classifier(sentence)
                label = res[0]['label']
                code = label.split()[0] if " " in label else label
                if code in scorer.right_codes:
                    print(f"  👉 Right-wing Category: {label}")
                elif code in scorer.left_codes:
                    print(f"  👈 Left-wing Category:  {label}")
                else:
                    print(f"  ⚪ Neutral/Other:       {label}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
                
        print("\n" + "=" * 78 + "\n")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Political Alignment Research Diagnostics")
    parser.add_argument("--audit", action="store_true", help="Run sentence-level classification audit on a random sample")
    parser.add_argument("--update-graphs", action="store_true", help="Recompile static/graph_data.json")
    
    args = parser.parse_args()
    
    if args.audit:
        audit_sample()
    elif args.update_graphs:
        aggregate_graphs()
    else:
        show_summary()
