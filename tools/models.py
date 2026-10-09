"""Manage the models bnk can serve. Every model is a file in configs/ (its settings and where to download it from).

    ./models.sh list                                 the configured models, and which are downloaded
    ./models.sh download iq3_s [--dir DIR]           download a model and check every file's SHA-256
    ./models.sh add REPO [--include PATTERN] [--name NAME] [--alias ALIAS]
                                                     write configs/NAME.json for a GGUF repo on Hugging Face

Where models go: the Hugging Face cache (~/.cache/huggingface/hub, or $HF_HOME / $HF_HUB_CACHE) unless a models
folder is set. `download --dir DIR` sets one: the model goes to DIR/<name>/ and DIR is saved to bnk.env as
BNK_MODELS, so run.sh and later downloads use it too. A model is looked for in BNK_MODELS first, then in the cache.

A config's "source" says what to download:

    "aliases": ["iq3_s"],
    "source": {"repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "revision": "ed59f9...",
               "files": ["IQ3_S/...-00001-of-00002.gguf", "IQ3_S/...-00002-of-00002.gguf"]}

`files` lists every shard, the first one first; `revision` pins the commit (omit it to follow the repo's main).
A model downloaded some other way can be given by path instead: "source": {"path": "/models/x-00001-of-00002.gguf"}.

Gated repos need a Hugging Face account that accepted the model's terms and a token: `.venv/bin/hf auth login`.
run.sh also uses `resolve CHOICE` (prints the config name and the first shard's path) and `menu` (one line per model
for its picker).
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CONFIGS = HERE / "configs"
ENV_FILE = HERE / "bnk.env"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "bnk"
VERIFIED = CACHE / "verified.json"   # files already hashed: path -> size, mtime, sha256
SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")

# a new config's defaults: Qwen's recommendation for thinking, the speculation settings tuned on IQ3_S
DEFAULTS = {
    "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
    "speculation": {"draft": 4, "draft_min_p": 0.8},
    "thinking_loop_guard": True,
}


def env_models() -> str:
    """The models folder: BNK_MODELS from the environment, else from bnk.env (empty: the Hugging Face cache)."""
    if os.environ.get("BNK_MODELS"):
        return os.environ["BNK_MODELS"]
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("BNK_MODELS="):
                return line.split("=", 1)[1].strip().strip('"')
    return ""


def save_env(key: str, value: str) -> None:
    lines = [l for l in (ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []) if not l.startswith(key + "=")]
    ENV_FILE.write_text("\n".join(lines + [f'{key}="{value}"']) + "\n")


def configs() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(CONFIGS.glob("*.json"))}


def find(choice: str) -> tuple[str, dict]:
    """A config by its name or one of its aliases (case-insensitive)."""
    for name, cfg in configs().items():
        if choice.lower() in [name.lower()] + [a.lower() for a in cfg.get("aliases", [])]:
            return name, cfg
    raise SystemExit(f"no model '{choice}' in configs/ (./models.sh list shows them; ./models.sh add REPO adds one)")


def locate(name: str, cfg: dict) -> Path | None:
    """The first shard of a downloaded model (every shard present), or None."""
    src = cfg.get("source", {})
    if "path" in src:
        p = Path(src["path"]).expanduser()
        return p if p.exists() else None
    files = src.get("files", [])
    if not files:
        return None
    if env_models():
        d = Path(env_models()).expanduser() / name
        if all((d / f).exists() for f in files):
            return d / files[0]
    from huggingface_hub import try_to_load_from_cache
    paths = [try_to_load_from_cache(src["repo"], f, revision=src.get("revision")) for f in files]
    if all(isinstance(p, str) for p in paths):
        return Path(paths[0])
    return None


def remote_files(repo: str, revision: str | None) -> tuple[str, dict[str, dict]]:
    """The commit a revision resolves to, and each file's size and SHA-256 there."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    return info.sha, {s.rfilename: {"size": s.lfs.size if s.lfs else s.size, "sha256": s.lfs.sha256 if s.lfs else None}
                      for s in info.siblings}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    done, size = 0, path.stat().st_size
    with open(path, "rb") as f:
        while chunk := f.read(64 << 20):
            h.update(chunk)
            done += len(chunk)
            print(f"\r    checking {path.name}: {100 * done / size:5.1f}%", end="", flush=True)
    print()
    return h.hexdigest()


def verified(path: Path, want: str) -> bool:
    """Whether the file has this SHA-256: hashed once, then remembered while its size and mtime stay the same."""
    try:
        seen = json.loads(VERIFIED.read_text())
    except (OSError, ValueError):
        seen = {}
    real, st = str(path.resolve()), path.stat()
    if seen.get(real) == {"size": st.st_size, "mtime": st.st_mtime, "sha256": want}:
        return True
    if sha256(path) != want:
        return False
    seen[real] = {"size": st.st_size, "mtime": st.st_mtime, "sha256": want}
    CACHE.mkdir(parents=True, exist_ok=True)
    VERIFIED.write_text(json.dumps(seen, indent=1))
    return True


def dump(cfg: dict) -> str:
    """A config in configs/' style: one key per line, short objects and lists inline, the file list one per line."""
    def inline(v):
        if isinstance(v, dict):
            return "{ " + ", ".join(f"{json.dumps(k)}: {json.dumps(x)}" for k, x in v.items()) + " }"
        return json.dumps(v)
    out = []
    for k, v in cfg.items():
        if k == "source":
            body = [f'    {json.dumps(sk)}: ' + ("[\n" + ",\n".join(f"      {json.dumps(f)}" for f in sv) + "\n    ]"
                                                if sk == "files" else json.dumps(sv)) for sk, sv in v.items()]
            out.append(f'  "source": {{\n' + ",\n".join(body) + "\n  }")
        else:
            out.append(f"  {json.dumps(k)}: {inline(v)}")
    return "{\n" + ",\n".join(out) + "\n}\n"


def gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


# ---------------------------------------------------------------------------------------------------------- commands
def cmd_list(_a) -> int:
    where = env_models() or "the Hugging Face cache"
    print(f"models folder: {where}\n")
    for name, cfg in configs().items():
        p = locate(name, cfg)
        aliases = ", ".join(cfg.get("aliases", []))
        src = cfg.get("source", {})
        print(f"  {name}" + (f"  ({aliases})" if aliases else ""))
        print(f"      {'downloaded: ' + str(p) if p else 'not downloaded'}")
        if "repo" in src:
            print(f"      https://huggingface.co/{src['repo']}")
    print("\n./models.sh download NAME   ./models.sh add REPO   ./run.sh NAME")
    return 0


def cmd_menu(_a) -> int:
    for name, cfg in configs().items():
        print(f"{name}\t{'ready' if locate(name, cfg) else 'not downloaded'}")
    return 0


def cmd_resolve(a) -> int:
    name, cfg = find(a.choice)
    p = locate(name, cfg)
    print(f"{name}\t{p or ''}")
    return 0 if p else 2


def cmd_download(a) -> int:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    name, cfg = find(a.model)
    src = cfg.get("source", {})
    if "repo" not in src:
        print(f"{name}: its config has no \"source\" repo to download from")
        return 1
    if a.dir:
        d = str(Path(a.dir).expanduser().resolve())
        if d != env_models():
            save_env("BNK_MODELS", d)
            print(f"models folder set to {d} (saved to bnk.env)")
    root = Path(env_models()).expanduser() / name if env_models() else None
    print(f"{name}: https://huggingface.co/{src['repo']} -> {root or 'the Hugging Face cache'}")
    try:
        commit, info = remote_files(src["repo"], src.get("revision"))
    except GatedRepoError:
        print(f"  this model is gated. Open https://huggingface.co/{src['repo']}, accept its terms (logged in),\n"
              f"  then run .venv/bin/hf auth login and try again.")
        return 1
    except RepositoryNotFoundError:
        print(f"  {src['repo']} is not reachable (renamed, private, or a token is needed: .venv/bin/hf auth login)")
        return 1
    missing = [f for f in src["files"] if f not in info]
    if missing:
        print(f"  not in the repo at {commit[:12]}: {', '.join(missing)}")
        return 1
    need = sum(info[f]["size"] or 0 for f in src["files"])
    disk = root or Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    disk.mkdir(parents=True, exist_ok=True)
    have = locate(name, cfg)
    free = shutil.disk_usage(disk).free
    if not have and need > free:
        print(f"  not enough disk space: up to {gb(need)} to download, {gb(free)} free on {disk}")
        return 1
    for f in src["files"]:
        want = info[f]
        print(f"  {f} ({gb(want['size'] or 0)})")
        try:
            out = Path(hf_hub_download(src["repo"], f, revision=commit, local_dir=str(root) if root else None))
        except GatedRepoError:
            print(f"  gated: accept the terms at https://huggingface.co/{src['repo']}, then .venv/bin/hf auth login")
            return 1
        if out.stat().st_size != want["size"]:
            print(f"  wrong size ({out.stat().st_size} bytes, expected {want['size']}): delete {out} and retry")
            return 1
        if not a.no_verify and want["sha256"] and not verified(out, want["sha256"]):
            print(f"  SHA-256 MISMATCH: the file is damaged. Delete {out.resolve()} and run this again")
            return 1
    print(f"done: ./run.sh {cfg.get('aliases', [name])[0]}")
    return 0


def cmd_add(a) -> int:
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    try:
        commit, info = remote_files(a.repo, a.revision)
    except (GatedRepoError, RepositoryNotFoundError) as e:
        print(f"cannot read {a.repo}: {type(e).__name__} (gated or private: .venv/bin/hf auth login)")
        return 1
    ggufs = sorted(f for f in info if f.endswith(".gguf") and (not a.include or fnmatch.fnmatch(f, a.include)))
    # group the files into models: a split model's shards share everything before -0000N-of-0000M
    groups: dict[str, list[str]] = {}
    for f in ggufs:
        groups.setdefault(SHARD.sub("", f).removesuffix(".gguf"), []).append(f)
    groups = {k: v for k, v in groups.items() if "mmproj" not in k.lower() and "mtp" not in Path(k).name.lower()}
    if len(groups) != 1:
        print(f"{a.repo} has {len(groups)} GGUF models{' matching ' + a.include if a.include else ''}"
              + (": pick one with --include, for example" if groups else "."))
        for k, v in groups.items():
            print(f"  --include '{k}*'   {gb(sum(info[f]['size'] or 0 for f in v))}, {len(v)} file(s)")
        return 1
    stem, files = next(iter(groups.items()))
    name = a.name or Path(stem).name
    path = CONFIGS / f"{name}.json"
    if path.exists() and not a.force:
        print(f"{path.relative_to(HERE)} already exists (--force replaces it)")
        return 1
    cfg = {"notes": f"{a.repo} ({Path(stem).name}). Sampling and speculation are bnk's defaults; not tuned yet.",
           "aliases": a.alias,
           "source": {"repo": a.repo, "revision": commit, "files": files},
           **DEFAULTS,
           "server_args": ["--model-id", name, "--max-tokens", "32768", "--slots", "2"]}
    if not a.alias:
        del cfg["aliases"]
    path.write_text(dump(cfg))
    print(f"wrote {path.relative_to(HERE)}: {len(files)} file(s), {gb(sum(info[f]['size'] or 0 for f in files))}")
    print(f"next: ./models.sh download {name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the configured models and where they are").set_defaults(fn=cmd_list)
    p = sub.add_parser("download", help="download a model (resumes; checks SHA-256)")
    p.add_argument("model", help="config name or alias")
    p.add_argument("--dir", help="models folder (saved to bnk.env); default: BNK_MODELS, else the Hugging Face cache")
    p.add_argument("--no-verify", action="store_true", help="skip the SHA-256 check (sizes are still checked)")
    p.set_defaults(fn=cmd_download)
    p = sub.add_parser("add", help="write configs/NAME.json for a GGUF repo on Hugging Face")
    p.add_argument("repo", help="for example ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF")
    p.add_argument("--include", help="glob picking one model in the repo, for example 'IQ3_S/*'")
    p.add_argument("--name", help="config name (default: the GGUF's name)")
    p.add_argument("--alias", action="append", default=[], help="short name for run.sh (repeatable)")
    p.add_argument("--revision", help="pin this branch or commit (default: main, pinned to its current commit)")
    p.add_argument("--force", action="store_true", help="replace an existing config")
    p.set_defaults(fn=cmd_add)
    p = sub.add_parser("resolve", help=argparse.SUPPRESS)
    p.add_argument("choice")
    p.set_defaults(fn=cmd_resolve)
    sub.add_parser("menu", help=argparse.SUPPRESS).set_defaults(fn=cmd_menu)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
