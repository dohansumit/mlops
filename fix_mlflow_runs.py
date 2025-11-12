#!/usr/bin/env python3
import os, shutil, yaml, pathlib, sys

BASE = pathlib.Path("data/mlruns/0")
BACKUP_DIR = pathlib.Path("data/mlruns/0_meta_backup")
os.makedirs(BACKUP_DIR, exist_ok=True)

if not BASE.exists():
    print("Base path", BASE.resolve(), "does not exist. Run from project root.")
    sys.exit(1)

print("Backing up meta.yaml files to", BACKUP_DIR)
count = 0
for run_dir in sorted([p for p in BASE.iterdir() if p.is_dir()]):
    meta = run_dir / "meta.yaml"
    if not meta.exists():
        print(" - skipping (no meta.yaml):", run_dir)
        continue
    # backup meta.yaml
    shutil.copy2(meta, BACKUP_DIR / (run_dir.name + ".meta.yaml.bak"))
    # load yaml
    try:
        data = yaml.safe_load(meta.read_text()) or {}
    except Exception as e:
        print(" - ERROR reading YAML for", run_dir, ":", e)
        continue

    modified = False

    # Ensure top-level experiment id matches folder name "0"
    desired_exp_id = str(BASE.name)  # "0"
    # common keys that might store experiment id
    for key in ("experiment_id","id","experimentId"):
        if key in data:
            if str(data[key]) != desired_exp_id:
                print(f" - {run_dir.name}: updating {key} {data[key]!r} -> {desired_exp_id!r}")
                data[key] = desired_exp_id
                modified = True
            break
    else:
        # no key present -> add experiment_id
        print(f" - {run_dir.name}: adding experiment_id = {desired_exp_id}")
        data["experiment_id"] = desired_exp_id
        modified = True

    # Ensure info block exists
    info = data.get("info") or {}
    if "artifact_uri" not in info or info.get("artifact_uri") in (None, ""):
        # set artifact_uri to file://<abs path>/artifacts
        art_path = (run_dir / "artifacts").resolve()
        info["artifact_uri"] = "file://" + str(art_path)
        print(f"   set info.artifact_uri -> {info['artifact_uri']}")
        modified = True
    # write back info
    data["info"] = info

    if modified:
        # write yaml safely
        meta.write_text(yaml.safe_dump(data, sort_keys=False))
        count += 1

    # ensure artifacts and inputs folders exist to avoid None joins
    for sub in ("artifacts", "inputs"):
        p = run_dir / sub
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            print(f"   created {p}")

print(f"Finished. Modified {count} meta.yaml files (backups in {BACKUP_DIR}).")
