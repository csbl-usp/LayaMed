# 8. Run on your computer

[← 7. Interface](07-interface.md) · [Guide index](../README.md#-step-by-step-guide) · [9. Limitations →](09-limitations.md)

## 1. Environment

Follow option B in [01-environment.md](01-environment.md).

## 2. Model

Two ways:

- **Train locally** (GPU with 6 GB or more): open `notebooks/layamed_colab.ipynb` in Jupyter or VS Code, with the `layamed` environment, and run cells 1 to 7. It detects that it is not on Colab and uses the repository folder.
- **Bring it from Colab:** cell 9 of the notebook downloads `layamed_model.zip`. Unzip it in the repository folder:

```powershell
Expand-Archive layamed_model.zip -DestinationPath .
```

At the end, the folder must have `checkpoint\model.safetensors`, `checkpoint\rl_agent_config.json` and `data\scope.pt` (and, if created, `data\recalibration.json`). Checkpoint somewhere else: `$env:LAYAMED_CHECKPOINT = "D:\models\my_checkpoint"`.

## 3. Server

```powershell
conda activate layamed
```

```powershell
python server.py
```

Wait for `LayaMed at http://127.0.0.1:8000` (the model takes ~10 s to load) and open that address. `Ctrl+C` stops it.

## 4. Other computers on the same network (optional)

```powershell
python server.py --lan
```

The server prints the address for the other computers (for example `http://192.168.0.10:8000`) and only accepts clients with private IPs (home or lab network).

For another computer to reach it:

1. The Wi-Fi must be set as a **Private** network (Settings → Network & internet → network properties).
2. The Windows firewall must allow the port on the Private profile. In PowerShell **as administrator**:

```powershell
New-NetFirewallRule -DisplayName "LayaMed 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private
```

**Never forward the port on the router** or expose the server to the internet: it has no authentication. Behind a tunnel or reverse proxy (ngrok, cloudflared, VS Code port forwarding) every client looks like `127.0.0.1` to the server, so the private-network check no longer protects it; do not use one outside Colab. If you open the page by the machine's name and get "Invalid host header", add it with `LAYAMED_ALLOWED_HOSTS` (see [06-server.md](06-server.md)). Use `--lan` only on trusted networks. If the model was trained on restricted data, clients see only the probabilities, but the data usage agreement may restrict even that.

To remove the rule later:

```powershell
Remove-NetFirewallRule -DisplayName "LayaMed 8000"
```

## Common problems

| Symptom | Likely cause |
|---|---|
| `Missing data\scope.pt` | run cell 5 of the notebook (the server does not start without the detector) |
| `Checkpoint not found` | `checkpoint\` is not in the repository folder; use `$env:LAYAMED_CHECKPOINT` |
| `OMP: Error #15` | conda numpy/scipy with pip torch (see 01-environment.md) |
| port 8000 busy | `$env:PORT = "8001"` before `python server.py` |
| the page says "needs the local Laya server" | the page was opened from disk; open it through the server's address |

---

[← 7. Interface](07-interface.md) · [Guide index](../README.md#-step-by-step-guide) · [9. Limitations →](09-limitations.md)
