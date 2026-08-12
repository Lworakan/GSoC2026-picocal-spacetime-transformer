# Remote GPU options

The laptop runs one CUDA job at a time (running several powered the machine off), so the
experiment queue is serial. The table below is about lifting that limit.

The column that matters most in practice is **"drivable from the agent session"**: whether
the machine can be reached from a shell, so training can be launched, monitored and scored
without a human clicking through a notebook. Colab and Kaggle cannot; anything with SSH or
a CLI can.

| provider | hardware | cost | drivable from a shell | notes |
|---|---|---|---|---|
| **Lab / CERN node** (mentors offered) | whatever the group has, A100 discussed | free | **yes**, via SSH | Best fit. Persistent, no session limits, and the throughput number the mentors asked for is worth more measured on a standard machine than on a laptop. Access was being arranged; worth chasing on Monday. |
| **Modal** | A10G / A100 / H100, serverless | free credits per month (~$30 at last check), then per-second billing | **yes**, `pip install modal` + one browser login, then `modal run` from the CLI | Best cloud option for this project: python-native, no server to manage, jobs are launched from a script. Cache and outputs go to a Modal volume. |
| **RunPod / Vast.ai** | 4090, A100, H100, spot pricing | ~$0.2–0.5/h for a 4090-class card | **yes**, real SSH | Cheapest per hour. Vast is a marketplace so machines vary; RunPod is more predictable. Pay-as-you-go, needs a card. |
| **Lambda Labs** | A100/H100 instances | ~$1+/h, often waitlisted | **yes**, SSH | Simple and reliable when capacity exists. |
| **Lightning AI Studios** | T4/A10G | free monthly GPU hours | **yes**, SSH + VS Code remote | Persistent studio filesystem, so the 354 MB cache is uploaded once. Good middle ground. |
| **Hugging Face Jobs** | T4 / A10G / A100 | needs Pro (~$9/month) | **yes**, `hf jobs run` from the CLI | You already have an HF account, and the HF connector is available in this session. Jobs are one-shot containers — point the checkpoint dir at a persistent volume or a dataset repo. |
| **Google Colab** | T4 free; L4/A100 on Pro | free / ~$10/month | no — cells must be run by hand | Still useful: several sessions in parallel is the point, not per-job speed (a free T4 is slower than the laptop's 4080). Setup is in `docs/colab.md`. |
| **Kaggle Notebooks** | P100 or 2×T4 | free, ~30 GPU-hours/week | no | Often overlooked and more generous than free Colab; 12-hour sessions, and datasets up to 100 GB, so the cache can live as a private dataset. Same manual-clicking limitation. |
| **University cluster** | varies | usually free for students | usually yes, SSH | Worth asking about in parallel with the CERN-side request. |

## What the job actually needs

Modest, which widens the options:

- One GPU with ≥ 4 GB of memory. The model is 623k parameters and uses ~1.2 GB of VRAM.
- ~3 GB of RAM per job.
- 354 MB of input: `.scratch/cache/minbias_94.pkl` and `.scratch/cache/clean-aux_100.pkl`.
  The 1.6 GB of ROOT files are not needed once the cache exists.
- Outputs are small: a prediction CSV of a few MB and 2.4 MB per model checkpoint.
- Any Linux box with CUDA works after `uv sync`; there is nothing site-specific in the code.

## Recommended order

1. **Ask the mentors for the lab node on Monday.** Free, persistent, and it produces the
   throughput number on hardware worth quoting. It is already an open item from the
   2026-08-07 meeting.
2. **Modal**, if something is needed today. Free monthly credits cover several full runs,
   it is drivable from a shell, and there is no machine to babysit.
3. **Kaggle or Colab** if no card is on file and the only goal is running the four remaining
   experiments in parallel overnight.

## Setting one up

For any SSH-reachable machine:

```bash
git clone https://github.com/Lworakan/GSoC2026-picocal-spacetime-transformer.git
cd GSoC2026-picocal-spacetime-transformer
uv sync                      # then replace the CPU-pinned torch with a CUDA build
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
mkdir -p .scratch/cache      # copy the two .pkl files here (scp/rsync)
uv run scripts/train_picocal.py --sample minbias --cleanaux --extra --dens \
    --seeds 0 1 2 3 4 --device cuda
```

Note that `pyproject.toml` pins torch to the CPU index on purpose (the laptop's power-off
history), which is why the CUDA wheel has to be installed explicitly on a remote box — or a
separate `.venv-gpu` used, as on the laptop.

Every run checkpoints each epoch and skips seeds already present in the output CSV, so an
interrupted job is resumed by re-running the identical command.
