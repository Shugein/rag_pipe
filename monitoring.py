import requests

def parse_metrics(url="http://localhost:8080/metrics"):
    resp = requests.get(url)
    lines = resp.text.splitlines()
    metrics = {}

    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.split(" ")
        if len(parts) != 2:
            continue
        name, value = parts
        try:
            metrics[name] = float(value)
        except ValueError:
            continue
    return metrics

def get_stats(metrics):
    stats = {}

    # токены
    stats["prompt_tokens"] = metrics.get("vllm:prompt_tokens_total", 0)
    stats["generated_tokens"] = metrics.get("vllm:generation_tokens_total", 0)

    # latency до первого токена
    ttfb_sum = metrics.get("vllm:time_to_first_token_seconds_sum", 0)
    ttfb_count = metrics.get("vllm:time_to_first_token_seconds_count", 1)
    stats["avg_time_to_first_token"] = ttfb_sum / ttfb_count

    # скорость генерации токенов
    tpt_sum = metrics.get("vllm:time_per_output_token_seconds_sum", 0)
    tpt_count = metrics.get("vllm:time_per_output_token_seconds_count", 1)
    if tpt_sum > 0:
        avg_time_per_token = tpt_sum / tpt_count
        stats["tokens_per_sec"] = 1.0 / avg_time_per_token
    else:
        stats["tokens_per_sec"] = 0

    # общее время запроса
    e2e_sum = metrics.get("vllm:e2e_request_latency_seconds_sum", 0)
    e2e_count = metrics.get("vllm:e2e_request_latency_seconds_count", 1)
    stats["avg_request_latency"] = e2e_sum / e2e_count

    # загрузка GPU cache
    stats["gpu_cache_usage"] = metrics.get("vllm:gpu_cache_usage_perc", 0) * 100

    # активные/ожидающие запросы
    stats["running_requests"] = metrics.get("vllm:num_requests_running", 0)
    stats["waiting_requests"] = metrics.get("vllm:num_requests_waiting", 0)

    return stats


if __name__ == "__main__":
    metrics = parse_metrics()
    stats = get_stats(metrics)
    print("📊 vLLM Monitoring:")
    for k, v in stats.items():
        print(f"{k}: {v}")
