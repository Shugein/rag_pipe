from datasets import load_dataset
ds = load_dataset("IlyaGusev/ru_news", split="train").shuffle(seed=7).select(range(5000))
