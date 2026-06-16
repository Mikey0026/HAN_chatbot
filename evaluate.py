import csv
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

from src.rag.chat import answer_question, INTERNSHIP_GI
from src.utils.config import load_config, project_root

cfg = load_config()
root = project_root()

store_dir = root / cfg["vector_store_dir"]

print("Loading vector store...")
embeddings = OllamaEmbeddings(model=cfg["embedding_model"])

vector_store = Chroma(
    collection_name=cfg["collection_name"],
    embedding_function=embeddings,
    persist_directory=str(store_dir),
)

print("Loading LLM...")
llm = ChatOllama(
    model=cfg["llm_model"],
    temperature=cfg["temperature"],
)

EVAL_FILE = Path("data/evaluation/questions.csv")


def run_evaluation():

    total = 0
    passed = 0

    with open(EVAL_FILE, encoding="utf-8-sig") as f:

        reader = csv.DictReader(f)

        for row in reader:

            question = row["question"]

            expected_keywords = [
                k.strip().lower()
                for k in row["expected_keywords"].split(";")
            ]

            answer, _ = answer_question(
                question=question,
                history=[],
                vector_store=vector_store,
                llm=llm,
                k=cfg["retrieval_k"],
                internship_type=INTERNSHIP_GI,
            )

            answer_lower = answer.lower()

            matches = sum(
                keyword in answer_lower
                for keyword in expected_keywords
            )

            score = matches / len(expected_keywords)

            success = score >= 0.5

            total += 1

            if success:
                passed += 1

            print("=" * 80)
            print(question)
            print()
            print("ANSWER:")
            print(answer)
            print()

            print(f"SCORE: {score:.0%}")
            print("PASS" if success else "FAIL")
            print()

    print("=" * 80)
    print(f"FINAL SCORE: {passed}/{total}")
    print(f"ACCURACY: {(passed/total)*100:.1f}%")


if __name__ == "__main__":
    run_evaluation()