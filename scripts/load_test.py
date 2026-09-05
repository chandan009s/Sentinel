import argparse
import json
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen


def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)
    index = (len(values) - 1) * p / 100
    lower = int(index)
    upper = min(lower + 1, len(values))

    if lower == upper:
        return values[lower]

    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def send_request(base_url, i):
    payload = {
        "event_id": f"LOAD-{uuid.uuid4()}",
        "merchant_id": f"LOAD-MERCHANT-{i % 100}",
        "timestamp": time.time(),
        "velocity_ratio": 1.0,
        "amount_ratio": 1.0,
        "velocity_acceleration_1m": 0.1,
        "amount_acceleration_1m": 0.1,
    }

    request = Request(
        f"{base_url}/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()

    try:
        with urlopen(request, timeout=15) as response:
            response.read()

        return True, time.perf_counter() - start

    except Exception:
        return False, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)

    args = parser.parse_args()

    start = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(send_request, args.url, i)
            for i in range(args.requests)
        ]

        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.perf_counter() - start

    latencies = [latency for success, latency in results]
    successful = sum(success for success, _ in results)
    failed = len(results) - successful

    print()
    print("AI Risk Manager Load Test")
    print("=" * 40)
    print(f"Requests    : {args.requests}")
    print(f"Concurrency : {args.concurrency}")
    print(f"Successful  : {successful}")
    print(f"Failed      : {failed}")
    print(f"Elapsed     : {elapsed:.3f}s")
    print(f"Throughput  : {len(results) / elapsed:.2f} req/s")

    if latencies:
        print()
        print(f"Mean        : {statistics.mean(latencies) * 1000:.2f} ms")
        print(f"Median      : {statistics.median(latencies) * 1000:.2f} ms")
        print(f"P95         : {percentile(latencies, 95) * 1000:.2f} ms")
        print(f"P99         : {percentile(latencies, 99) * 1000:.2f} ms")
        print(f"Max         : {max(latencies) * 1000:.2f} ms")


if __name__ == "__main__":
    main()