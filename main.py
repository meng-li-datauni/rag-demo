"""Interactive Q&A over the PDFs in docs/."""

from rag import RagPipeline


def main() -> None:
    rag = RagPipeline()
    print(f"Indexed {rag.num_chunks} chunks. Ask a question (empty line to quit).")
    while question := input("\n> ").strip():
        answer, retrieved = rag.ask(question)
        print(answer)
        print("  sources:", sorted({d.metadata["source"] for d in retrieved}))


if __name__ == "__main__":
    main()
