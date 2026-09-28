import json

import mteb

from encoder import PrePostPipelineEncoder


def main() -> None:
    model = PrePostPipelineEncoder()
    task = mteb.get_task("AppsRetrieval")
    result = mteb.evaluate(model, [task], encode_kwargs={"batch_size": 64})

    task_result = list(result.task_results)[0]
    with open("appsretrieval_results.json", "w") as f:
        json.dump(task_result.to_dict(), f, indent=2, default=str)
    print("Saved appsretrieval_results.json")


if __name__ == "__main__":
    main()
