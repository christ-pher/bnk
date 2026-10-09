# Running bnk as a service: changing the model

The server runs as a systemd **user** service, one model per instance: `bnk@<model>`. Only one model fits the GPU,
so changing models means stopping the running instance and starting another. No `sudo` is needed.

## Model names

The part after `@` is the same name `run.sh` takes:

| Instance                   | Model                                    |
| -------------------------- | ---------------------------------------- |
| `bnk@iq3_s`                | Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S         |
| `bnk@orca`                 | Qwen3.8-Flash-Next-Uncensored-IQ4_XS     |
| `bnk@cyber-frost`          | CYBER-FROST-3.8-PS-Q5_K_M                |
| `bnk@abliterated`          | Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S |
| `bnk@swift`                | Swift-1.5-GSQ-RCO-IQ3_XXS                |
| `bnk@swift-abliterated`    | Swift-1.5-GSQ-RCO-abliterated-IQ3_XXS    |

## Change the model

```bash
systemctl --user list-units 'bnk@*'          # which one is running now
systemctl --user stop bnk@iq3_s               # stop it (frees the GPU and ~100 GB of RAM)
systemctl --user start bnk@orca               # start the new one
journalctl --user -u bnk@orca -f              # watch it load; ready when it prints "listening on http://0.0.0.0:8080"
```

Or as one line (stops whatever bnk instance is running, then starts the new one):

```bash
systemctl --user stop 'bnk@*' && systemctl --user start bnk@orca
```

Check it is up:

```bash
curl -s localhost:8080/v1/models        # shows the loaded model's id
```

Loading takes roughly 20-60 seconds; `/health` answers 503 until the engine is ready.

## Start a model at boot

Requires linger, set once: `sudo loginctl enable-linger chris`. Only enable **one** instance:

```bash
systemctl --user disable bnk@iq3_s            # whichever was enabled before
systemctl --user enable bnk@orca
systemctl --user list-unit-files 'bnk@*'      # check: exactly one "enabled"
```

`systemctl --user enable --now bnk@orca` enables and starts it in one step (stop the old one first).

## Everyday commands

```bash
systemctl --user status bnk@orca              # running? memory, PIDs
systemctl --user restart bnk@orca             # restart the same model (e.g. after a git pull)
journalctl --user -u bnk@orca --since "1 hour ago"
journalctl --user -u bnk@orca -f              # live log (what the tmux pane used to show)
```

## Notes

- **Settings per model** live in `configs/<model id>.json` (sampling, speculation, `--slots`, etc.). Edit the file,
  then `systemctl --user restart bnk@<model>`.
- **Extra server options** (e.g. a different port or `--park-gib 12`): run `systemctl --user edit bnk@.service` and
  add an override:
  ```ini
  [Service]
  ExecStart=
  ExecStart=/opt/engines/bnk/run.sh %i --park-gib 12
  ```
  Then `systemctl --user daemon-reload` and restart the instance.
- **Crashes:** if the kernel kills the engine (out of memory), the server restarts the engine by itself; if the
  server process dies, systemd restarts it after 15 s. Conversations parked in RAM are lost either way, so their next
  turn re-reads its prompt.
- **omp needs no change** when you switch models: bnk ignores the model id in requests and answers with whatever is
  loaded. All six ids are already in `~/.omp/agent/models.yml`; pick the matching one in omp's `/model` if you want
  the name in omp to be correct.
- **Running by hand instead** (e.g. to try a `.gguf` path): stop the service first, then `./run.sh <model>` as before.
- The unit file is `~/.config/systemd/user/bnk@.service`; the repo keeps a copy in `deploy/bnk@.service`. On a new
  machine: `mkdir -p ~/.config/systemd/user && cp deploy/bnk@.service ~/.config/systemd/user/`, adjust the paths if
  the repo is not at `/opt/engines/bnk`, then `systemctl --user daemon-reload` (and `loginctl enable-linger $USER`
  so it runs without a login session).
