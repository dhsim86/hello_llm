if __name__ == '__main__':
    from utils.evaluation import change_jsonl_to_csv

    change_jsonl_to_csv("results/text2sql_evaluation_result_pretrained.jsonl",
                        "results/text2sql_evaluation_result_pretrained.csv")