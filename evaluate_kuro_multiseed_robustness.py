import argparse
import gc
import hashlib
import inspect
import io
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def ratio(n, d):
    return n / d if d else float("nan")


def npy(value):
    if isinstance(value, np.ndarray):
        return value
    return np.load(io.BytesIO(value))


def one_band(value, name):
    a = np.asarray(npy(value))
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"{name}: unexpected shape {a.shape}")
    return a


def info_json(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    return json.loads(value)


def to_db(a, valid):
    a = np.asarray(a, dtype=np.float32)
    out = np.zeros_like(a)
    usable = valid & np.isfinite(a) & (a > 0)
    out[usable] = (10.0 * np.log10(a[usable])).astype(np.float32)
    bad = int(np.sum(valid & (~np.isfinite(a) | (a <= 0))))
    return out, bad


def zero_counts():
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def add_counts(dst, src):
    for k in ("tp", "fp", "fn", "tn"):
        dst[k] += int(src[k])


def metrics_from_counts(c):
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    return {
        "iou": ratio(tp, tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def event_key(info):
    return f"{info.get('actid')}|{info.get('flood_date')}"


def load_matrix(repo_root):
    p = repo_root / "experiments" / "fw02b_kuro_multiseed_matrix.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing Stage B matrix: {p}")
    return json.loads(p.read_text())


def build_model(UNet, checkpoint, expected_sha, device):
    actual = sha256_file(checkpoint)
    if actual != expected_sha:
        raise RuntimeError(f"Checkpoint SHA mismatch for {checkpoint.name}: {actual} != {expected_sha}")
    kwargs = {"in_channels": 2, "out_channels": 1, "base": 16}
    if "bilinear" in inspect.signature(UNet).parameters:
        kwargs["bilinear"] = True
    model = UNet(**kwargs)
    try:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        saved = torch.load(checkpoint, map_location="cpu")
    state = saved["model_state"] if isinstance(saved, dict) and "model_state" in saved else saved
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    del saved, state
    return model, actual


def serialize_state(processed_samples, overall, events, bad_vv, bad_vh, started_at):
    return {
        "processed_samples": processed_samples,
        "overall": overall,
        "events": {m: {t: dict(v) for t, v in tv.items()} for m, tv in events.items()},
        "bad_valid_vv_pixels": bad_vv,
        "bad_valid_vh_pixels": bad_vh,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def restore_state(state, model_names):
    overall = {m: {"primary_flood_only": zero_counts(), "secondary_total_water": zero_counts()} for m in model_names}
    events = {m: {"primary_flood_only": defaultdict(zero_counts), "secondary_total_water": defaultdict(zero_counts)} for m in model_names}
    if state is None:
        return 0, overall, events, 0, 0, datetime.now(timezone.utc).isoformat()
    for m in model_names:
        for t in ("primary_flood_only", "secondary_total_water"):
            overall[m][t] = {k: int(v) for k, v in state["overall"][m][t].items()}
            for ev, c in state["events"][m][t].items():
                events[m][t][ev] = {k: int(v) for k, v in c.items()}
    return (
        int(state["processed_samples"]), overall, events,
        int(state.get("bad_valid_vv_pixels", 0)), int(state.get("bad_valid_vh_pixels", 0)),
        state.get("started_at") or datetime.now(timezone.utc).isoformat(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--repo-root", default="/content/sar-edge-flood-ai-fw02b")
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--save-every-batches", type=int, default=20)
    args = ap.parse_args()

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "webdataset", "huggingface_hub"], check=True)
    import webdataset as wds
    from huggingface_hub import HfApi, hf_hub_url

    project_root = Path(args.project_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    matrix = load_matrix(repo_root)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Stage B.")
    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.models.unet import UNet

    ds = matrix["dataset"]
    evcfg = matrix["evaluation"]
    repo_id = ds["repository"]
    repo_files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    shards = sorted(f for f in repo_files if re.fullmatch(r"test_GRD/shard-\d+\.tar", f))
    if len(shards) != int(ds["expected_shards"]):
        raise RuntimeError(f"Expected {ds['expected_shards']} test_GRD shards, found {len(shards)}")
    if not 0 <= args.shard_index < len(shards):
        raise IndexError(f"shard-index must be 0..{len(shards)-1}")

    stage_a_parent = project_root / matrix["stage_a_parent_output"]
    stage_b_parent = project_root / matrix["stage_b_parent_output"]
    model_specs = sorted(matrix["models"], key=lambda x: int(x["order"]))

    models = {}
    ckpt_audit = []
    for spec in model_specs:
        cp = stage_a_parent / spec["run_name"] / "best_validation_checkpoint.pt"
        if not cp.exists():
            raise FileNotFoundError(cp)
        model, actual = build_model(UNet, cp, spec["checkpoint_sha256"], device)
        models[spec["run_name"]] = model
        ckpt_audit.append({
            "run_name": spec["run_name"], "seed": spec["seed"], "beta": spec["beta"],
            "checkpoint": str(cp), "sha256": actual,
        })

    selected = shards[args.shard_index]
    shard_name = Path(selected).stem
    print("STATUS: READY TO START")
    print(json.dumps({
        "experiment_id": matrix["experiment_id"],
        "device": torch.cuda.get_device_name(0),
        "models": len(models),
        "shards": len(shards),
        "selected_shard": selected,
        "threshold": evcfg["threshold"],
        "checkpoint_audit": ckpt_audit,
    }, indent=2))
    if args.preflight_only:
        return

    out = stage_b_parent / shard_name
    if (out / "COMPLETE.json").exists():
        raise RuntimeError(f"{shard_name} already completed; refusing to repeat it")
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment_id": matrix["experiment_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_shard": selected,
        "shard_index": args.shard_index,
        "dataset_repository": repo_id,
        "dataset_split": ds["split"],
        "input_fields": ds["input_fields"],
        "channel_order": ds["channel_order"],
        "radiometric_transform": ds["radiometric_transform"],
        "valid_pixels": ds["valid_pixels"],
        "ignored_label_values": ds["ignored_label_values"],
        "threshold": evcfg["threshold"],
        "precision": evcfg["precision"],
        "fine_tuning": False,
        "threshold_tuning": False,
        "model_selection_on_kuro": False,
        "device": torch.cuda.get_device_name(0),
        "checkpoints": ckpt_audit,
    }
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        comparable = dict(manifest); comparable.pop("created_at")
        old_cmp = dict(old); old_cmp.pop("created_at", None)
        if old_cmp != comparable:
            raise RuntimeError("Existing shard manifest differs; refusing to mix incompatible evidence")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))

    progress_path = out / "progress.json"
    prior = json.loads(progress_path.read_text()) if progress_path.exists() else None
    model_names = list(models.keys())
    processed, overall, events, bad_vv_total, bad_vh_total, started_at = restore_state(prior, model_names)
    print("Resume processed samples:", processed)

    batch_size = int(evcfg["batch_size_cuda"])
    threshold = float(evcfg["threshold"])
    url = hf_hub_url(repo_id, selected, repo_type="dataset")
    stream = "pipe:curl -L --fail --silent --show-error --retry 5 --retry-delay 5 " + repr(url)
    dataset = wds.WebDataset(stream, shardshuffle=False)
    iterator = iter(dataset)

    seen = 0
    batch = []
    batches_since_save = 0
    wall_start = time.time()

    def process_batch(items):
        nonlocal bad_vv_total, bad_vh_total
        x = torch.from_numpy(np.stack([z["x"] for z in items])).to(device)
        for mname, model in models.items():
            with torch.inference_mode():
                pred = (torch.sigmoid(model(x))[:, 0] >= threshold).cpu().numpy().astype(bool)
            for z, yhat in zip(items, pred):
                valid = z["valid"]
                mask = z["mask"]
                ev = z["event_id"]
                for target, ref in (
                    ("primary_flood_only", valid & (mask == 2)),
                    ("secondary_total_water", valid & ((mask == 1) | (mask == 2))),
                ):
                    yp = yhat & valid
                    c = {
                        "tp": int(np.sum(yp & ref)),
                        "fp": int(np.sum(yp & ~ref & valid)),
                        "fn": int(np.sum(~yp & ref)),
                        "tn": int(np.sum(~yp & ~ref & valid)),
                    }
                    add_counts(overall[mname][target], c)
                    add_counts(events[mname][target][ev], c)
        bad_vv_total += sum(z["bad_vv"] for z in items)
        bad_vh_total += sum(z["bad_vh"] for z in items)
        del x

    try:
        for sample in iterator:
            if seen < processed:
                seen += 1
                continue
            seen += 1
            key = str(sample.get("__key__"))
            required = ["flood_vv.npy", "flood_vh.npy", "mask.npy", "valid_mask.npy", "info.json"]
            missing = [f for f in required if f not in sample]
            if missing:
                raise KeyError(f"{shard_name}:{key}: missing {missing}")
            vv = one_band(sample["flood_vv.npy"], "VV").astype(np.float32)
            vh = one_band(sample["flood_vh.npy"], "VH").astype(np.float32)
            mask = one_band(sample["mask.npy"], "mask")
            vm = one_band(sample["valid_mask.npy"], "valid_mask")
            if not (vv.shape == vh.shape == mask.shape == vm.shape == (224, 224)):
                raise ValueError(f"{shard_name}:{key}: incompatible shapes")
            provided_valid = vm == 1
            valid = provided_valid & np.isin(mask, [0, 1, 2])
            vv_db, bad_vv = to_db(vv, valid)
            vh_db, bad_vh = to_db(vh, valid)
            info = info_json(sample["info.json"])
            batch.append({
                "x": np.stack([vv_db, vh_db]).astype(np.float32),
                "mask": mask, "valid": valid, "event_id": event_key(info),
                "bad_vv": bad_vv, "bad_vh": bad_vh,
            })
            if len(batch) >= batch_size:
                process_batch(batch)
                processed += len(batch)
                batch.clear()
                batches_since_save += 1
                if batches_since_save >= args.save_every_batches:
                    progress_path.write_text(json.dumps(
                        serialize_state(processed, overall, events, bad_vv_total, bad_vh_total, started_at), indent=2
                    ))
                    batches_since_save = 0
                    print(f"processed={processed:,} elapsed={(time.time()-wall_start)/60:.1f} min", flush=True)

        if batch:
            process_batch(batch)
            processed += len(batch)
            batch.clear()
        progress_path.write_text(json.dumps(
            serialize_state(processed, overall, events, bad_vv_total, bad_vh_total, started_at), indent=2
        ))
    finally:
        if hasattr(iterator, "close"):
            iterator.close()
        del iterator, dataset
        gc.collect()

    overall_rows = []
    event_rows = []
    for spec in model_specs:
        m = spec["run_name"]
        for target in ("primary_flood_only", "secondary_total_water"):
            c = overall[m][target]
            overall_rows.append({
                "run_name": m, "seed": spec["seed"], "beta": spec["beta"], "target": target,
                **c, **metrics_from_counts(c),
            })
            for ev, ec in events[m][target].items():
                event_rows.append({
                    "run_name": m, "seed": spec["seed"], "beta": spec["beta"], "target": target,
                    "event_id": ev, **ec, **metrics_from_counts(ec),
                })

    overall_df = pd.DataFrame(overall_rows)
    event_df = pd.DataFrame(event_rows)
    overall_df.to_csv(out / "overall_metrics_by_model.csv", index=False)
    event_df.to_csv(out / "event_metrics_by_model.csv", index=False)

    macro = (event_df.groupby(["run_name", "seed", "beta", "target"], as_index=False)
             .agg(events=("event_id", "nunique"), event_macro_iou=("iou", "mean"),
                  event_macro_precision=("precision", "mean"), event_macro_recall=("recall", "mean"),
                  event_macro_f1=("f1", "mean")))
    macro.to_csv(out / "event_macro_metrics_by_model.csv", index=False)

    complete = {
        "completed": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "shard_index": args.shard_index,
        "shard_file": selected,
        "samples": processed,
        "models": len(models),
        "bad_valid_vv_pixels": bad_vv_total,
        "bad_valid_vh_pixels": bad_vh_total,
        "elapsed_minutes_this_session": (time.time() - wall_start) / 60,
        "files": {},
    }
    for name in ("run_manifest.json", "progress.json", "overall_metrics_by_model.csv", "event_metrics_by_model.csv", "event_macro_metrics_by_model.csv"):
        complete["files"][name] = sha256_file(out / name)
    (out / "COMPLETE.json").write_text(json.dumps(complete, indent=2))
    print("SHARD COMPLETE")
    print(json.dumps(complete, indent=2))


if __name__ == "__main__":
    main()
