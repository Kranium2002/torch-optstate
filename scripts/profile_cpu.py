import argparse
import time
import torch

from torch.optim import AdamW, SGD
from torch_optstate import wrap, WarmupPolicy
from torch_optstate.codecs import FP16Codec, BF16Codec, Int8MomentumCodec


def build_model(hidden_dim: int, layers: int) -> torch.nn.Module:
    modules = []
    for i in range(layers):
        modules.append(torch.nn.Linear(hidden_dim, hidden_dim))
        if i < layers - 1:
            modules.append(torch.nn.ReLU())
    return torch.nn.Sequential(*modules)


def build_policy(policy_name: str, steps: int, opt_name: str) -> WarmupPolicy:
    if opt_name == "adamw":
        momentum_key = "exp_avg"
        variance_key = "exp_avg_sq"
    else:
        momentum_key = "momentum_buffer"
        variance_key = "unused"

    if policy_name == "fp32":
        warmup_steps = steps + 1
        variance_codec = None
    elif policy_name == "int8":
        warmup_steps = 0
        variance_codec = None
    elif policy_name == "int8_all":
        warmup_steps = 0
        variance_codec = Int8MomentumCodec()
    elif policy_name == "fp16":
        warmup_steps = 0
        variance_codec = FP16Codec()
    elif policy_name == "bf16":
        warmup_steps = 0
        variance_codec = BF16Codec()
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    return WarmupPolicy(
        warmup_steps=warmup_steps,
        momentum_key=momentum_key,
        variance_key=variance_key,
        variance_codec=variance_codec,
    )


def merge_stats(total, step_stats):
    for key, value in step_stats.items():
        if key.endswith("_by_codec"):
            by_codec = total.setdefault(key, {})
            for codec_name, codec_stats in value.items():
                entry = by_codec.setdefault(codec_name, {"time_s": 0.0, "tensors": 0, "bytes": 0, "elements": 0})
                entry["time_s"] += codec_stats.get("time_s", 0.0)
                entry["tensors"] += codec_stats.get("tensors", 0)
                entry["bytes"] += codec_stats.get("bytes", 0)
                entry["elements"] += codec_stats.get("elements", 0)
        elif isinstance(value, (int, float)):
            total[key] = total.get(key, 0.0) + value


def print_summary(total, steps):
    def ms_per_step(val):
        return (val / steps) * 1000.0

    print("\nPer-step averages (ms):")
    for key in ("materialize", "decode", "materialize_overhead", "step", "commit", "encode", "commit_overhead"):
        if key in total:
            print(f"  {key:>22}: {ms_per_step(total[key]):8.3f}")

    if "encode" in total and total.get("encode", 0.0) > 0:
        enc_mb = (total.get("encode_bytes", 0) / steps) / (1024 * 1024)
        enc_tensors = total.get("encode_tensors", 0) / steps
        print(f"  {'encode_tensors':>22}: {enc_tensors:8.2f} / step")
        print(f"  {'encode_mb':>22}: {enc_mb:8.2f} / step")

    if "decode" in total and total.get("decode", 0.0) > 0:
        dec_mb = (total.get("decode_bytes", 0) / steps) / (1024 * 1024)
        dec_tensors = total.get("decode_tensors", 0) / steps
        print(f"  {'decode_tensors':>22}: {dec_tensors:8.2f} / step")
        print(f"  {'decode_mb':>22}: {dec_mb:8.2f} / step")

    for section in ("encode_by_codec", "decode_by_codec"):
        if section in total:
            print(f"\n{section}:")
            items = list(total[section].items())
            items.sort(key=lambda x: x[1].get("time_s", 0.0), reverse=True)
            for name, entry in items:
                ms = ms_per_step(entry.get("time_s", 0.0))
                tensors = entry.get("tensors", 0) / steps
                mb = (entry.get("bytes", 0) / steps) / (1024 * 1024)
                print(f"  {name:>22}: {ms:8.3f} ms, {tensors:8.2f} tensors, {mb:8.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="CPU profiling for torch-optstate encode/decode hot spots.")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--burnin", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--policy", choices=["fp32", "int8", "int8_all", "fp16", "bf16"], default="int8")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--initial-chunk-size", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cpu")
    model = build_model(args.hidden_dim, args.layers).to(device)

    if args.optimizer == "adamw":
        base_opt = AdamW(model.parameters(), lr=1e-3)
    else:
        base_opt = SGD(model.parameters(), lr=1e-3, momentum=0.9)

    policy = build_policy(args.policy, args.steps, args.optimizer)

    opt = wrap(
        base_opt,
        policy=policy,
        chunk_size=args.chunk_size,
        initial_chunk_size=args.initial_chunk_size,
        profile=True,
    )

    x = torch.randn(args.batch, args.hidden_dim, device=device)
    y = torch.randn(args.batch, args.hidden_dim, device=device)

    # Burn-in to reduce compile/alloc noise.
    for _ in range(args.burnin):
        opt.zero_grad(set_to_none=True)
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()

    totals = {}
    t0 = time.perf_counter()
    for _ in range(args.steps):
        opt.zero_grad(set_to_none=True)
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
        merge_stats(totals, opt.last_step_timings)
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / max(1, args.steps)) * 1000.0
    print(f"\nAvg end-to-end step time: {avg_ms:.3f} ms")
    print_summary(totals, args.steps)


if __name__ == "__main__":
    main()
