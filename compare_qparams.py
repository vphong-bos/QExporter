#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch


QPARAM_TOKENS = ("scale", "zero_point", "zeropoint")


def load_pt(path):
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        return data
    if hasattr(data, "state_dict"):
        return data.state_dict()
    raise TypeError(f"Unsupported .pt content in {path}: {type(data)!r}")


def is_qparam_key(key):
    lowered = key.lower()
    return any(token in lowered for token in QPARAM_TOKENS)


def compare_values(ref, cand, atol, rtol):
    if isinstance(ref, torch.Tensor) and isinstance(cand, torch.Tensor):
        if ref.shape != cand.shape:
            return {
                "status": "shape_mismatch",
                "ref_shape": tuple(ref.shape),
                "cand_shape": tuple(cand.shape),
            }

        ref_cpu = ref.detach().cpu()
        cand_cpu = cand.detach().cpu()

        if ref_cpu.dtype != cand_cpu.dtype:
            dtype_status = {
                "ref_dtype": str(ref_cpu.dtype),
                "cand_dtype": str(cand_cpu.dtype),
            }
        else:
            dtype_status = {}

        if ref_cpu.numel() == 0:
            return {"status": "match", "max_abs_diff": 0.0, "mean_abs_diff": 0.0, **dtype_status}

        if ref_cpu.is_floating_point() or cand_cpu.is_floating_point():
            ref_cmp = ref_cpu.to(torch.float32)
            cand_cmp = cand_cpu.to(torch.float32)
            abs_diff = (cand_cmp - ref_cmp).abs()
            allclose = torch.allclose(cand_cmp, ref_cmp, atol=atol, rtol=rtol)
            return {
                "status": "match" if allclose else "value_mismatch",
                "max_abs_diff": abs_diff.max().item(),
                "mean_abs_diff": abs_diff.mean().item(),
                **dtype_status,
            }

        equal = torch.equal(ref_cpu, cand_cpu)
        diff_count = (ref_cpu != cand_cpu).sum().item()
        return {
            "status": "match" if equal else "value_mismatch",
            "num_different": int(diff_count),
            **dtype_status,
        }

    if ref == cand:
        return {"status": "match"}
    return {
        "status": "value_mismatch",
        "ref_repr": repr(ref),
        "cand_repr": repr(cand),
    }


def compare_state_dicts(ref_state, cand_state, atol, rtol):
    ref_keys = set(ref_state)
    cand_keys = set(cand_state)
    shared_keys = sorted(ref_keys & cand_keys)

    results = {}
    for key in shared_keys:
        results[key] = compare_values(ref_state[key], cand_state[key], atol=atol, rtol=rtol)

    return {
        "missing_in_candidate": sorted(ref_keys - cand_keys),
        "missing_in_reference": sorted(cand_keys - ref_keys),
        "results": results,
    }


def summarize(comparison):
    results = comparison["results"]
    matched = [key for key, item in results.items() if item["status"] == "match"]
    mismatched = [key for key, item in results.items() if item["status"] != "match"]

    qparam_results = {key: item for key, item in results.items() if is_qparam_key(key)}
    qparam_matched = [key for key, item in qparam_results.items() if item["status"] == "match"]
    qparam_mismatched = [key for key, item in qparam_results.items() if item["status"] != "match"]

    return {
        "shared_keys": len(results),
        "matched": len(matched),
        "mismatched": len(mismatched),
        "only_in_reference": len(comparison["missing_in_candidate"]),
        "only_in_candidate": len(comparison["missing_in_reference"]),
        "qparam_shared": len(qparam_results),
        "qparam_matched": len(qparam_matched),
        "qparam_mismatched": len(qparam_mismatched),
    }


def print_report(comparison, summary, limit, reference_label, candidate_label):
    print("Comparison Summary")
    print(f"  Reference: {reference_label}")
    print(f"  Candidate: {candidate_label}")
    print(f"  Shared keys: {summary['shared_keys']}")
    print(f"  Matched: {summary['matched']}")
    print(f"  Mismatched: {summary['mismatched']}")
    print(f"  Only in reference: {summary['only_in_reference']}")
    print(f"  Only in candidate: {summary['only_in_candidate']}")
    print(f"  Qparam shared: {summary['qparam_shared']}")
    print(f"  Qparam matched: {summary['qparam_matched']}")
    print(f"  Qparam mismatched: {summary['qparam_mismatched']}")

    if comparison["missing_in_candidate"]:
        print(f"\nOnly In Reference ({reference_label})")
        for key in comparison["missing_in_candidate"][:limit]:
            print(f"  - {key}")

    if comparison["missing_in_reference"]:
        print(f"\nOnly In Candidate ({candidate_label})")
        for key in comparison["missing_in_reference"][:limit]:
            print(f"  - {key}")

    mismatches = [(key, item) for key, item in comparison["results"].items() if item["status"] != "match"]
    if mismatches:
        print("\nMismatches")
        for key, item in mismatches[:limit]:
            line = f"  - {key}: {item['status']}"
            if "max_abs_diff" in item:
                line += f", max_abs_diff={item['max_abs_diff']:.6g}, mean_abs_diff={item['mean_abs_diff']:.6g}"
            if "num_different" in item:
                line += f", num_different={item['num_different']}"
            if "ref_shape" in item:
                line += f", ref_shape={item['ref_shape']}, cand_shape={item['cand_shape']}"
            if "ref_dtype" in item:
                line += f", ref_dtype={item['ref_dtype']}, cand_dtype={item['cand_dtype']}"
            print(line)


def main():
    parser = argparse.ArgumentParser(description="Compare two .pt files containing state_dict-like data.")
    parser.add_argument("reference", type=Path, help="Reference .pt file")
    parser.add_argument("candidate", type=Path, help="Candidate .pt file")
    parser.add_argument("--atol", type=float, default=0.001, help="Absolute tolerance for floating-point tensors")
    parser.add_argument("--rtol", type=float, default=0.001, help="Relative tolerance for floating-point tensors")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of missing/mismatch entries to print")
    args = parser.parse_args()

    ref_state = load_pt(args.reference)
    cand_state = load_pt(args.candidate)

    comparison = compare_state_dicts(ref_state, cand_state, atol=args.atol, rtol=args.rtol)
    summary = summarize(comparison)
    print_report(
        comparison,
        summary,
        limit=args.limit,
        reference_label=str(args.reference),
        candidate_label=str(args.candidate),
    )

    has_diff = (
        summary["mismatched"] > 0
        or summary["only_in_reference"] > 0
        or summary["only_in_candidate"] > 0
    )
    raise SystemExit(1 if has_diff else 0)


if __name__ == "__main__":
    main()
