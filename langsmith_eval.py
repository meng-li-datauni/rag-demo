"""Run the evaluation as a LangSmith experiment.

  .venv/bin/python langsmith_eval.py --upload-only               # dataset only, no Claude calls
  .venv/bin/python langsmith_eval.py --mode agent --limit 3      # experiment on 3 questions
  .venv/bin/python langsmith_eval.py --mode multi --judge        # + LLM-as-judge (extra tokens)

Results show up in LangSmith under Datasets & Experiments -> halvorsen-rag-demo, where
experiments can be compared side by side.
"""

import argparse
import os
import sys

from evaluate import EVAL_CASES, make_ask, score_case  # importing rag also loads .env
from langchain_anthropic import ChatAnthropic
from langsmith import Client
from rag import TOP_K

DATASET_NAME = "halvorsen-rag-demo"
JUDGE_MODEL = "claude-haiku-4-5"


def ensure_dataset(client: Client) -> None:
    """Create the dataset from EVAL_CASES the first time. Later edits happen in LangSmith,
    e.g. adding examples from production traces with "Add to dataset"."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Dataset '{DATASET_NAME}' already exists")
        return
    dataset = client.create_dataset(
        DATASET_NAME,
        description="Questions about fictional Halvorsen Robotics PDFs; answers exist only "
                    "in the documents. Empty expected_sources = should decline.",
    )
    client.create_examples(dataset_id=dataset.id, examples=[
        {"inputs": {"question": q},
         "outputs": {"expected_sources": sources, "facts": facts}}
        for q, sources, facts in EVAL_CASES
    ])
    print(f"Created dataset '{DATASET_NAME}' with {len(EVAL_CASES)} examples")


# --- Evaluators ----------------------------------------------------------------------

def code_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """Code-based evaluator: deterministic keyword + source checks (free, no LLM)."""
    score = score_case(reference_outputs["expected_sources"], reference_outputs["facts"],
                       outputs["answer"], set(outputs["sources"]))
    return {"results": [
        {"key": "answer_correct", "score": score["answer_ok"],
         "comment": f"missing: {score['missing']}" if score["missing"] else None},
        {"key": "retrieval_hit", "score": score["retrieval_ok"]},
    ]}


JUDGE_PROMPT = """You grade answers from a company-document Q&A assistant.

Question: {question}
Facts the answer must contain: {facts}
(If the facts list is empty, the documents don't contain the answer and the assistant
must say it doesn't know.)

Assistant's answer: {answer}

Is the answer correct and free of invented facts? Reply with exactly one word,
CORRECT or INCORRECT, then a one-sentence reason."""

_judge_llm = None


def llm_judge(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LLM-as-judge evaluator: tolerant of paraphrases, but costs tokens and can be wrong."""
    global _judge_llm
    _judge_llm = _judge_llm or ChatAnthropic(model=JUDGE_MODEL, max_tokens=200)
    facts = [f.replace("|", " or ") for f in reference_outputs["facts"]]
    verdict = _judge_llm.invoke(JUDGE_PROMPT.format(
        question=inputs["question"], facts=facts or "(none)", answer=outputs["answer"],
    )).text.strip()
    return {"key": "judge_correct", "score": verdict.upper().startswith("CORRECT"),
            "comment": verdict}


# --- Main ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["rag", "agent", "multi"], default="rag")
    parser.add_argument("--limit", type=int, default=None,
                        help="only run N examples (saves API tokens)")
    parser.add_argument("--judge", action="store_true",
                        help="add the LLM-as-judge evaluator (uses extra tokens)")
    parser.add_argument("--upload-only", action="store_true",
                        help="create the dataset and stop (no Claude calls)")
    args = parser.parse_args()

    if not os.environ.get("LANGSMITH_API_KEY"):
        sys.exit("Set LANGSMITH_API_KEY in .env first (https://smith.langchain.com -> Settings -> API Keys)")

    client = Client()
    ensure_dataset(client)
    if args.upload_only:
        return

    ask = make_ask(args.mode)

    def target(inputs: dict) -> dict:
        answer, sources = ask(inputs["question"])
        return {"answer": answer, "sources": sorted(sources)}

    results = client.evaluate(
        target,
        data=client.list_examples(dataset_name=DATASET_NAME, limit=args.limit),
        evaluators=[code_evaluator, *([llm_judge] if args.judge else [])],
        experiment_prefix=f"{args.mode}-k{TOP_K}",
        metadata={"mode": args.mode, "top_k": TOP_K},
        max_concurrency=1,
    )
    print(f"\nExperiment: {results.experiment_name}")
    print(f"Open in LangSmith: {results.url}")


if __name__ == "__main__":
    main()
