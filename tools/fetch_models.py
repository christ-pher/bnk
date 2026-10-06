"""Download a model preset from Hugging Face into the folder run.sh expects, and check every file's SHA-256.

    .venv/bin/python tools/fetch_models.py iq3_s [orca ...] --models DIR [--no-verify] [--list]

Presets (pinned to the revision bnk was tested with):
  iq3_s        ISTA-DASLab's GSQ-RCO IQ3_S (the default model)            -> DIR/IQ3_S/
  orca         orcarouter's Uncensored IQ4_XS (gated: accept its terms)    -> DIR/orca-iq4_xs/
  abliterated  SC117's GSQ-RCO-abliterated IQ3_S                           -> DIR/gsq-rco-abliterated/IQ3_S/

Files already there with the right size and hash are kept, so it can be re-run after an interruption. A gated repo
needs a Hugging Face account that accepted the model's terms on its page and a token on this machine:
`.venv/bin/hf auth login`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

PRESETS = {
    "iq3_s": {"repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
              "revision": "ed59f92082b1e93c0e96d60a8b11aab089b52f09",
              "files": ["IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
                        "IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"],
              "dir": ""},                       # the repo's own IQ3_S/ folder lands at DIR/IQ3_S
    "orca": {"repo": "orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF",
             "revision": "0434906af7b5202b676d43f108cf4f73d25691ef",
             "files": ["Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf",
                       "Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00002-of-00003.gguf",
                       "Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00003-of-00003.gguf"],
             "dir": "orca-iq4_xs"},
    "abliterated": {"repo": "SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF",
                    "revision": "eec4e10a2c29b440a2ae85591fa11569ce752963",
                    "files": ["IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf",
                              "IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf"],
                    "dir": "gsq-rco-abliterated"},
}


def token() -> str | None:
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    p = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
    return p.read_text().strip() if p.exists() else None


def file_info(repo: str, revision: str) -> dict[str, dict]:
    """Each file's size and SHA-256 at the revision (Hugging Face's API)."""
    req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true")
    if token():
        req.add_header("Authorization", f"Bearer {token()}")
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    return {s["rfilename"]: {"size": (s.get("lfs") or {}).get("size", s.get("size")),
                             "sha256": (s.get("lfs") or {}).get("sha256")} for s in d.get("siblings", [])}


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("presets", nargs="*", help=", ".join(PRESETS))
    ap.add_argument("--models", required=False, default=os.environ.get("BNK_MODELS", ""), help="models directory")
    ap.add_argument("--no-verify", action="store_true", help="skip the SHA-256 check (sizes are still checked)")
    ap.add_argument("--list", action="store_true", help="show the presets and their sizes")
    a = ap.parse_args()
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        print("huggingface_hub is missing: .venv/bin/pip install -r requirements.txt")
        return 1
    if a.list or not a.presets:
        for name, p in PRESETS.items():
            try:
                info = file_info(p["repo"], p["revision"])
                gb = sum(info[f]["size"] or 0 for f in p["files"] if f in info) / 1e9
                print(f"  {name:12s} {gb:6.1f} GB  https://huggingface.co/{p['repo']}")
            except Exception:
                print(f"  {name:12s}    ?    GB  https://huggingface.co/{p['repo']} (gated or offline: log in to see)")
        return 0
    if not a.models:
        print("where should the models go? pass --models DIR (setup.sh saves it to bnk.env as BNK_MODELS)")
        return 1
    root = Path(a.models).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    for name in a.presets:
        if name not in PRESETS:
            print(f"unknown preset {name!r} ({', '.join(PRESETS)})")
            return 1
        p = PRESETS[name]
        dest = root / p["dir"] if p["dir"] else root
        print(f"{name}: https://huggingface.co/{p['repo']} -> {dest}")
        try:
            info = file_info(p["repo"], p["revision"])
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                print(f"  this model is gated. Open https://huggingface.co/{p['repo']}, accept its terms (logged in),\n"
                      f"  create a read token at https://huggingface.co/settings/tokens, then run:\n"
                      f"      .venv/bin/hf auth login\n  and run this again.")
                return 1
            raise
        need = 0
        for f in p["files"]:
            out = dest / f
            want = info.get(f, {}).get("size") or 0
            if not out.exists() or out.stat().st_size != want:
                need += want
        free = shutil.disk_usage(root).free
        if need > free:
            print(f"  not enough disk space: {need / 1e9:.0f} GB to download, {free / 1e9:.0f} GB free on {root}")
            return 1
        for f in p["files"]:
            out = dest / f
            want = info.get(f, {})
            if out.exists() and out.stat().st_size == want.get("size") and (a.no_verify or out.with_suffix(out.suffix + ".ok").exists()):
                print(f"  {f}: already there")
                continue
            if not (out.exists() and out.stat().st_size == want.get("size")):
                print(f"  {f}: downloading {(want.get('size') or 0) / 1e9:.1f} GB")
                try:
                    hf_hub_download(p["repo"], f, revision=p["revision"], local_dir=str(dest), token=token())
                except GatedRepoError:
                    print(f"  gated: accept the terms at https://huggingface.co/{p['repo']} and run .venv/bin/hf auth login")
                    return 1
                except RepositoryNotFoundError:
                    print(f"  {p['repo']} is not reachable (renamed, private, or a token is needed)")
                    return 1
            if out.stat().st_size != want.get("size"):
                print(f"  {f}: wrong size ({out.stat().st_size} bytes, expected {want.get('size')}); delete it and retry")
                return 1
            if not a.no_verify and want.get("sha256"):
                if sha256(out) != want["sha256"]:
                    print(f"  {f}: SHA-256 MISMATCH - the file is damaged; delete it and run this again")
                    return 1
                out.with_suffix(out.suffix + ".ok").write_text(want["sha256"] + "\n")
            print(f"  {f}: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
