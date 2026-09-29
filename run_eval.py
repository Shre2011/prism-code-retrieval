import json

import mteb

from encoder import PrePostPipelineEncoder, RUN_ID


def main() -> None:
    model = PrePostPipelineEncoder()
    task = mteb.get_task("AppsRetrieval")
    result = mteb.evaluate(model, [task], encode_kwargs={"batch_size": 64})

    task_result = list(result.task_results)[0]
    out = f"results_run{RUN_ID}.json"
    with open(out, "w") as f:
        json.dump(task_result.to_dict(), f, indent=2, default=str)
    print("Saved", out)
    s = task_result.to_dict()["scores"]["test"][0]
    print("NDCG@10:", s["ndcg_at_10"], "MRR@10:", s["mrr_at_10"])


if __name__ == "__main__":
    main()
