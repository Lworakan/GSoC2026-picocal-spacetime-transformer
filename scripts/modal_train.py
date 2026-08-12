import modal

app = modal.App('picocal-train')

image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch', 'numpy', 'pandas', 'scipy', 'pyyaml',
                      'uproot', 'awkward', 'scikit-learn')
         .add_local_dir('scripts', remote_path='/root/scripts'))

vol = modal.Volume.from_name('picocal', create_if_missing=True)

JOBS = [
    ('exdnw2', ['--extra', '--dens', '--window', '2']),
    ('exdnrho', ['--extra', '--dens', '--rho']),
    ('exdntp', ['--extra', '--dens', '--tpull']),
    ('exdntpw2', ['--extra', '--dens', '--tpull', '--window', '2']),
]


@app.function(image=image, gpu='A10G', volumes={'/vol': vol}, timeout=6 * 60 * 60,
              memory=16384, cpu=4.0, max_containers=4)
def train(name: str, flags: list, seeds: list):
    import os
    import pathlib
    import subprocess

    root = pathlib.Path('/root')
    (root / '.scratch' / 'cache').mkdir(parents=True, exist_ok=True)
    for f in pathlib.Path('/vol/cache').glob('*.pkl'):
        dst = root / '.scratch' / 'cache' / f.name
        if not dst.exists():
            os.symlink(f, dst)

    # The cache key is <tag>_<number of input files>.pkl, so the remote side must see the
    # same file count for the key to match. The ROOT files themselves are never opened
    # because the cache is present, so empty placeholders are enough and 1.6 GB of input
    # data stays on the laptop.
    for sub, pat, n in (('minimum_bias', 'f{}.root', 94), ('full', 'matched_{}.root', 100)):
        d = root / 'data' / sub
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / pat.format(i)).touch()

    for d in ('/vol/preds', '/vol/models', f'/vol/ckpt/{name}'):
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)

    cmd = ['python', '-u', '/root/scripts/train_picocal.py', '--repo', '/root',
           '--sample', 'minbias', '--cleanaux', '--device', 'cuda',
           '--out', '/vol/preds', '--models-dir', '/vol/models',
           '--ckpt-dir', f'/vol/ckpt/{name}', '--seeds'] + [str(s) for s in seeds] + flags
    r = subprocess.run(cmd, cwd='/root', text=True)
    vol.commit()
    print(f'[{name}] rc={r.returncode}', flush=True)
    return f'[{name}] rc={r.returncode}'


@app.local_entrypoint()
def main(seeds: str = '0 1', only: str = ''):
    sl = [int(s) for s in seeds.split()]
    jobs = [(n, f) for n, f in JOBS if not only or n in only.split()]
    print(f'launching {len(jobs)} jobs on A10G: ' + ', '.join(n for n, _ in jobs))
    for out in train.starmap([(n, f, sl) for n, f in jobs]):
        print(out)
