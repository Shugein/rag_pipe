import streamlit as st
import requests
import time
import pynvml
from prometheus_client.parser import text_string_to_metric_families

VLLM_METRICS_URL = "http://localhost:8080/metrics"

# Инициализация NVML
pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()

def get_vllm_metrics():
    """Парсит метрики из vLLM Prometheus endpoint"""
    try:
        resp = requests.get(VLLM_METRICS_URL, timeout=5)
        resp.raise_for_status()
        metrics = {}
        for family in text_string_to_metric_families(resp.text):
            for sample in family.samples:
                metrics[sample.name] = sample.value
        return metrics
    except Exception as e:
        st.error(f"Ошибка получения метрик: {e}")
        return {}

def get_gpu_stats():
    """Собираем загрузку GPU и память"""
    stats = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        # В новых версиях NVML строка уже str, в старых — bytes
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        stats.append({
            "id": i,
            "name": name,
            "gpu_util": util.gpu,
            "mem_used": mem_info.used / (1024**3),
            "mem_total": mem_info.total / (1024**3),
        })
    return stats


# --- Streamlit Dashboard ---
st.set_page_config(page_title="vLLM GPU Dashboard", layout="wide")
st.title("📊 vLLM + GPU Monitoring Dashboard")

placeholder = st.empty()

while True:
    with placeholder.container():
        # vLLM metrics
        metrics = get_vllm_metrics()

        col1, col2, col3 = st.columns(3)
        col1.metric("⚡ Tokens Generated", f"{metrics.get('vllm:generation_tokens_total', 0):.0f}")
        col2.metric("📝 Prompt Tokens", f"{metrics.get('vllm:prompt_tokens_total', 0):.0f}")
        col3.metric("✅ Successful Requests", f"{metrics.get('vllm:request_success_total', 0):.0f}")

        col1, col2, col3 = st.columns(3)
        # Скорость генерации токенов
        gen_time = metrics.get("vllm:time_per_output_token_seconds_sum", 0.0)
        gen_count = metrics.get("vllm:time_per_output_token_seconds_count", 1)
        tok_per_sec = (gen_count / gen_time) if gen_time > 0 else 0
        col1.metric("🚀 Tokens/sec", f"{tok_per_sec:.2f}")

        # Latency до первого токена
        ttfb_sum = metrics.get("vllm:time_to_first_token_seconds_sum", 0.0)
        ttfb_count = metrics.get("vllm:time_to_first_token_seconds_count", 1)
        ttfb = (ttfb_sum / ttfb_count) if ttfb_count > 0 else 0
        col2.metric("⏱️ Time to First Token", f"{ttfb:.3f} s")

        # Очередь и активные запросы
        running = metrics.get("vllm:num_requests_running", 0)
        waiting = metrics.get("vllm:num_requests_waiting", 0)
        col3.metric("🔄 Requests", f"Running: {running}, Waiting: {waiting}")

        st.subheader("🎮 GPU Stats")
        gpu_stats = get_gpu_stats()
        for gpu in gpu_stats:
            st.progress(gpu["gpu_util"] / 100, text=f"GPU {gpu['id']} {gpu['name']} - {gpu['gpu_util']}%")
            st.write(f"Memory: {gpu['mem_used']:.2f} / {gpu['mem_total']:.2f} GB")

        st.divider()
        st.caption("Данные обновляются каждые 2 секунды")

    time.sleep(2)
