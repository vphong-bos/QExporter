#!/usr/bin/env python3
import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import numpy_helper

import torch


class QuantParamFp32Mapper:
    def __init__(self, fp32_path, qparam_path):
        self.fp32_path = str(fp32_path)
        self.qparam_path = str(qparam_path)

        self.fp32_raw = self._load_checkpoint(self.fp32_path)
        self.qparam_raw = self._load_encodings(self.qparam_path)

        if "model" in self._extract_state_dict(self.fp32_raw).keys():
            self.fp32_state = self._extract_state_dict(self.fp32_raw)["model"]
        else:
            self.fp32_state = self._extract_state_dict(self.fp32_raw)

        self.activation_state, self.qparam_state = self._extract_encodings(self.qparam_raw)

    @staticmethod
    def _load_checkpoint(path):
        suffix = Path(path).suffix.lower()

        if suffix == ".pkl":
            with open(path, "rb") as f:
                return pickle.load(f)
            
        if suffix == ".onnx":
            model = onnx.load(path)

            state_dict = {}
            for initializer in model.graph.initializer:
                arr = numpy_helper.to_array(initializer)
                state_dict[initializer.name] = torch.from_numpy(arr.copy())

            return state_dict

        return torch.load(path, map_location="cpu", weights_only=False)

    @staticmethod
    def _load_encodings(path):
        import json

        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if not isinstance(obj, dict):
            raise TypeError(f"Encodings must be dict, got: {type(obj)}")

        if "activation_encodings" not in obj and "param_encodings" not in obj:
            raise ValueError(
                "Invalid encodings file: missing 'activation_encodings' and 'param_encodings'"
            )

        return obj

    @staticmethod
    def _extract_state_dict(obj):
        if isinstance(obj, dict):
            if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                return obj["state_dict"]
            return obj
        raise TypeError(f"Unsupported checkpoint format: {type(obj)}")

    @staticmethod
    def _extract_encodings(obj):
        if not isinstance(obj, dict):
            raise TypeError(f"Encodings must be dict, got: {type(obj)}")

        activation_encodings = obj.get("activation_encodings", {})
        param_encodings = obj.get("param_encodings", {})

        flat_param_encodings = {}
        for name, enc in param_encodings.items():
            flat_param_encodings[name] = enc

        return activation_encodings, flat_param_encodings

    @staticmethod
    def _encoding_shape(value):
        if isinstance(value, list):
            return (len(value),)
        return None

    @staticmethod
    def _strip_prefixes(name):
        prefixes = ("module.", "model.", "_orig_mod.")
        changed = True

        while changed:
            changed = False
            for prefix in prefixes:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    changed = True

        return name


    @classmethod
    def _normalize_name(cls, name):
        name = cls._strip_prefixes(name)

        replacements = [
            ("patch_embeddings.", "patch_embed."),
            ("weight_zeropoint", "weight_zero_point"),
            ("bias_zeropoint", "bias_zero_point"),
            ("input_zeropoint", "input_zero_point"),
            ("output_zeropoint", "output_zero_point"),

            # Hard coded, don't like me
            ("instance_head.", "ins_embed_head."),
            ("semantic_head.", "sem_seg_head."),
        ]

        for src, dst in replacements:
            name = name.replace(src, dst)

        return name

    @staticmethod
    def _is_norm_like_part(part):
        p = part.lower()

        return (
            # norm
            p == "norm"
            or (p.startswith("norm") and p[4:].isdigit())

            # layernorm
            or p == "layernorm"
            or p.startswith("layernorm_")
            or (p.startswith("layernorm") and p[len("layernorm"):].isdigit())

            # ln
            or p == "ln"
            or (p.startswith("ln") and p[2:].isdigit())

            # batchnorm
            or p == "batchnorm"
            or p.startswith("batchnorm_")
            or (p.startswith("batchnorm") and p[len("batchnorm"):].isdigit())

            # bn
            or p == "bn"
            or (p.startswith("bn") and p[2:].isdigit())
        )

    @classmethod
    def _has_norm_segment(cls, name):
        return any(cls._is_norm_like_part(part) for part in name.split("."))

    @classmethod
    def _candidate_names(cls, name):
        base = cls._normalize_name(name)
        candidates = [base]

        parts = base.split(".")

        for i, part in enumerate(parts):
            if cls._is_norm_like_part(part):
                reduced = parts[:i] + parts[i + 1 :]
                if reduced:
                    candidates.append(".".join(reduced))

        out = []
        seen = set()
        for c in candidates:
            if c not in seen:
                seen.add(c)
                out.append(c)

        return out

    @staticmethod
    def _tensor_shape(value):
        if torch.is_tensor(value):
            return tuple(value.shape)
        return None

    def _build_normalized_index(self, state_dict):
        index = defaultdict(list)
        for key in state_dict.keys():
            for norm in self._candidate_names(key):
                index[norm].append(key)
        return index

    def build_mapping(self):
        fp32_keys = list(self.fp32_state.keys())
        qparam_keys = list(self.qparam_state.keys())

        qparam_exact = set(qparam_keys)
        qparam_norm_index = self._build_normalized_index(self.qparam_state)
        fp32_norm_index = self._build_normalized_index(self.fp32_state)

        matched = {}
        unmatched_fp32 = []
        ambiguous_fp32 = []
        shape_mismatches = []

        for fp32_key in fp32_keys:
            if fp32_key in qparam_exact:
                matched[fp32_key] = fp32_key
                continue

            norm = self._normalize_name(fp32_key)
            candidates = list(dict.fromkeys(qparam_norm_index.get(norm, [])))

            if len(candidates) == 1:
                qkey = candidates[0]
                matched[fp32_key] = qkey

                fp32_shape = self._tensor_shape(self.fp32_state[fp32_key])
                q_shape = self._encoding_shape(self.qparam_state[qkey])

                if fp32_shape and q_shape and fp32_shape[0] != q_shape[0]:
                    shape_mismatches.append((fp32_key, qkey, fp32_shape, q_shape))

            elif len(candidates) > 1:
                ambiguous_fp32.append((fp32_key, norm, candidates))
            else:
                if not self._has_norm_segment(fp32_key):
                    unmatched_fp32.append(fp32_key)

        matched_qparam_keys = set(matched.values())

        unmatched_qparam = [k for k in qparam_keys if k not in matched_qparam_keys]

        reverse_unmatched_qparam = []
        for qkey in unmatched_qparam:
            norm = self._normalize_name(qkey)
            if norm not in fp32_norm_index:
                reverse_unmatched_qparam.append(qkey)

        return {
            "matched": matched,
            "unmatched_fp32": unmatched_fp32,
            "unmatched_qparam": unmatched_qparam,
            "reverse_unmatched_qparam": reverse_unmatched_qparam,
            "ambiguous_fp32": ambiguous_fp32,
            "shape_mismatches": shape_mismatches,
            "fp32_count": len(fp32_keys),
            "qparam_count": len(qparam_keys),
            "matched_count": len(matched),
        }

    def report(self, limit=100):
        result = self.build_mapping()

        print("=" * 80)
        print("Checkpoint Mapping Report")
        print("=" * 80)
        print(f"FP32 keys        : {result['fp32_count']}")
        print(f"QParam keys      : {result['qparam_count']}")
        print(f"Matched          : {result['matched_count']}")
        print(f"Missing in qparam: {len(result['unmatched_fp32'])}")
        print(f"Missing in fp32  : {len(result['reverse_unmatched_qparam'])}")
        print(f"Ambiguous        : {len(result['ambiguous_fp32'])}")
        print(f"Shape mismatch   : {len(result['shape_mismatches'])}")
        print()

        print("=== MATCHED ===")
        for k, v in list(result["matched"].items())[:limit]:
            print(f"{k}  -->  {v}")
        print()

        print("=== MISSING IN QPARAM ===")
        for k in result["unmatched_fp32"][:limit]:
            print(k)
        print()

        print("=== MISSING IN FP32 ===")
        for k in result["reverse_unmatched_qparam"][:limit]:
            print(k)
        print()

        print("=== AMBIGUOUS ===")
        for fp32_key, norm, candidates in result["ambiguous_fp32"][:limit]:
            print(fp32_key)
            print(f"  normalized: {norm}")
            print(f"  candidates: {candidates}")
        print()

        print("=== SHAPE MISMATCH ===")
        for fp32_key, qkey, s1, s2 in result["shape_mismatches"][:limit]:
            print(f"{fp32_key} vs {qkey}")
            print(f"  fp32: {s1}, qparam: {s2}")

        return result


def main():
    parser = argparse.ArgumentParser(
        description="Compare FP32 checkpoint with qparam checkpoint"
    )
    parser.add_argument("fp32_path")
    parser.add_argument("qparam_path")
    parser.add_argument("--limit", type=int, default=100)

    args = parser.parse_args()

    mapper = QuantParamFp32Mapper(args.fp32_path, args.qparam_path)
    mapper.report(limit=args.limit)


if __name__ == "__main__":
    main()