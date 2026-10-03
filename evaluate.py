"""Evaluate the RAG pipeline on questions whose answers exist only in the PDFs.

Usage: .venv/bin/python evaluate.py --mode rag|agent|multi [--limit N]

Each case checks two things:
  * retrieval - did the top-k chunks include the file that holds the answer?
  * answer    - does the answer contain every expected fact (case-insensitive)?
Unanswerable cases pass only if the model says it doesn't know.
"""

import argparse
import uuid

from rag import NO_ANSWER, RagPipeline

# (question, expected source files, facts the answer must contain)
# Empty sources = unanswerable from the documents; the model should decline.
EVAL_CASES = [
    ("How many days of paid time off do full-time employees get?",
     ["employee_handbook.pdf"], ["27"]),
    ("Which days must employees be in the office?",
     ["employee_handbook.pdf"], ["tuesday", "thursday"]),
    ("How much is the on-call pay for a weekend day?",
     ["employee_handbook.pdf"], ["160"]),
    ("Who is the lead engineer of Project Kestrel?",
     ["project_kestrel_spec.pdf"], ["ines varga"]),
    ("What is the battery capacity and flight time of the Kestrel drone?",
     ["project_kestrel_spec.pdf"], ["41.5", "34"]),
    ("When is Kestrel planned to become generally available?",
     ["project_kestrel_spec.pdf"], ["14 march 2027|march 14, 2027"]),
    ("What was total revenue in Q3 2026?",
     ["q3_2026_sales_report.pdf"], ["18.4"]),
    ("Which customer ordered the most units in Q3, and how many?",
     ["q3_2026_sales_report.pdf"], ["bellmark", "212"]),
    ("What caused the INC-0917 telemetry outage?",
     ["incident_postmortem_inc0917.pdf"], ["certificate", "tg-east-2"]),
    ("How long did the telemetry outage last, and how many customers were affected?",
     ["incident_postmortem_inc0917.pdf"], ["47", "fourteen|14"]),
    # Cross-document question: needs the spec and the sales report.
    ("What is the list price of Kestrel, and what Q4 2026 revenue range does sales expect?",
     ["project_kestrel_spec.pdf", "q3_2026_sales_report.pdf"], ["12,900|12900", "19", "21"]),
    # Unanswerable - not in any document.
    ("What is the CEO's annual salary?", [], []),
    ("How many employees does Halvorsen Robotics have?", [], []),
]


def contains_fact(answer: str, fact: str) -> bool:
    # "a|b" means either spelling is accepted.
    return any(option in answer for option in fact.split("|"))


def score_case(expected_sources: list[str], facts: list[str], answer: str,
               retrieved_sources: set[str]) -> dict:
    """Score one answer. Shared by this script, the web UI and the LangSmith evaluators."""
    answer_lc = answer.lower()
    if expected_sources:
        missing = [f for f in facts if not contains_fact(answer_lc, f)]
        return {
            "retrieval_ok": set(expected_sources) <= retrieved_sources,
            "answer_ok": not missing and NO_ANSWER.lower() not in answer_lc,
            "missing": missing,
        }
    # Unanswerable: nothing to retrieve, and the model must decline.
    return {"retrieval_ok": True, "answer_ok": "don't know" in answer_lc, "missing": []}


def make_ask(mode: str):
    """Return ask(question) -> (answer, set of retrieved source files) for the mode."""
    rag = RagPipeline()
    print(f"Indexed {rag.num_chunks} chunks - mode: {mode}\n")
    if mode == "rag":
        def ask(question):
            answer, retrieved = rag.ask(question)
            return answer, {d.metadata["source"] for d in retrieved}
        return ask

    from agents import build_agent, build_multi_agent
    runner = (build_agent if mode == "agent" else build_multi_agent)(rag.vector_store)

    def ask(question):
        result = runner.ask(question, thread_id=str(uuid.uuid4()))  # fresh memory per question
        return result.answer, {s["source"] for s in result.sources}
    return ask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["rag", "agent", "multi"], default="rag")
    parser.add_argument("--limit", type=int, default=None,
                        help="only run the first N questions (saves API tokens)")
    args = parser.parse_args()
    cases = EVAL_CASES[:args.limit]
    ask = make_ask(args.mode)

    retrieval_hits = answer_hits = 0
    for i, (question, expected_sources, facts) in enumerate(cases, start=1):
        answer, retrieved_sources = ask(question)
        score = score_case(expected_sources, facts, answer, retrieved_sources)
        retrieval_hits += score["retrieval_ok"]
        answer_hits += score["answer_ok"]
        status = "PASS" if score["answer_ok"] and score["retrieval_ok"] else "FAIL"
        print(f"[{status}] Q{i}: {question}")
        print(f"    answer:    {answer.strip()}")
        print(f"    retrieved: {sorted(retrieved_sources)}"
              + ("" if score["retrieval_ok"] else f"  (expected {expected_sources})"))
        if score["missing"]:
            print(f"    missing facts: {score['missing']}")
        print()

    n = len(cases)
    print(f"Retrieval accuracy: {retrieval_hits}/{n}")
    print(f"Answer accuracy:    {answer_hits}/{n}")


if __name__ == "__main__":
    main()
