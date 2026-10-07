#!/usr/bin/env python3
"""Render the example figures embedded in the Hugging Face dataset card.

Run from the repo root on a machine with the corpora present (the 3D geometry
figure additionally needs raw/meshes/ for the chosen sample):

    python scripts/make_dataset_card_figures.py [--out corpora_release/assets/]

Produces:
    examples_overview.png       one sample per corpus family: patch / parasitic /
                                far-field @ 5.6 GHz / |S11|
    example_frequencies.png     one sample's far-field maps at all 5 frequencies
                                + parasitic-layer surface current
    example_geometry_3d.png     the antenna mesh with each element type colored
                                and labeled (uses raw/meshes/, trimesh)
"""

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO_ROOT = Path(__file__).resolve().parent.parent

FREQS = [2400, 2800, 5200, 5600, 6000]
IDX56 = 3  # 5.6 GHz, matches config['idx_freq']
LINE = '#2b6cb0'

SAMPLES = [
    ('fmnist_train', 'FMNIST'),
    ('fmnist_cifar_train', 'FMNIST + CIFAR'),
    ('random_pixel', 'random pixel'),
    ('classic_patch_with_reflector', 'classic + parasitic'),
]
MESH_SAMPLE = 'fmnist_cifar_train'  # corpus for the 3D figure + frequency strip
# Pinned showcase ids: the numerically-first sample can be visually poor (e.g.
# fmnist_cifar id 2 has only 4 patch pixels above threshold); id 10000 is rich.
SHOWCASE_IDS = {'fmnist_cifar_train': '10000'}

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
                     'xtick.labelsize': 6, 'ytick.labelsize': 6,
                     'axes.edgecolor': '#cccccc', 'axes.linewidth': 0.6})


def first_file(corpus: str) -> str:
    if corpus in SHOWCASE_IDS:
        return str(REPO_ROOT / f'data/corpora/{corpus}/processed/'
                               f'processed_data_{SHOWCASE_IDS[corpus]}.pt')
    fs = glob.glob(str(REPO_ROOT / f'data/corpora/{corpus}/processed/processed_data_*.pt'))
    fs.sort(key=lambda p: int(Path(p).stem.split('_')[-1])
            if Path(p).stem.split('_')[-1].isdigit() else 0)
    return fs[0]


def gain_db(farfield: torch.Tensor) -> np.ndarray:
    return 10 * np.log10(np.maximum(farfield.numpy(), 1e-3))


def fig_overview(out: Path) -> None:
    titles = ['patch layer (16×16)', 'parasitic layer (16×16)',
              'far-field gain @ 5.6 GHz (dB)', '|S11| (dB)']
    fig, axes = plt.subplots(4, 4, figsize=(9.6, 9.0))
    for r, (corpus, label) in enumerate(SAMPLES):
        d = torch.load(first_file(corpus), weights_only=False)
        ap = d['example_paramters']['ant_parameters']
        axes[r, 0].imshow(np.asarray(ap['matrix']), cmap='Greys', vmin=0, vmax=1)
        axes[r, 0].set_ylabel(label, fontsize=8.5)
        ref = ap.get('reflector_matrix')
        if ref is not None:
            axes[r, 1].imshow(np.asarray(ref), cmap='Greys', vmin=0, vmax=1)
        else:
            axes[r, 1].text(0.5, 0.5, 'no parasitic\nlayer', ha='center', va='center',
                            transform=axes[r, 1].transAxes, color='#999999')
        im = axes[r, 2].imshow(gain_db(d['farfeilds'][IDX56]), cmap='viridis')
        fig.colorbar(im, ax=axes[r, 2], fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
        s11 = d['s11']
        f = np.asarray(s11['frequancy']).squeeze()
        db = np.asarray(s11['s11_abs_db']).squeeze()
        f = f / 1e3 if f.max() > 100 else f
        m = (f >= 2.0) & (f <= 6.0)
        axes[r, 3].plot(f[m], db[m], color=LINE, lw=1.6)
        axes[r, 3].set_xlim(2, 6)
        axes[r, 3].grid(color='#eeeeee', lw=0.5)
        if r == 0:
            for c in range(4):
                axes[r, c].set_title(titles[c])
        if r == len(SAMPLES) - 1:
            axes[r, 3].set_xlabel('frequency (GHz)')
    for ax in axes[:, :3].flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('One sample per corpus family', y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out / 'examples_overview.png', dpi=160, facecolor='white')
    plt.close(fig)
    print('  examples_overview.png')


def fig_frequencies(out: Path) -> None:
    d = torch.load(first_file(MESH_SAMPLE), weights_only=False)
    fig, axes = plt.subplots(1, 6, figsize=(13.2, 2.6))
    for i, fr in enumerate(FREQS):
        im = axes[i].imshow(gain_db(d['farfeilds'][i]), cmap='viridis')
        axes[i].set_title(f'{fr / 1000:.1f} GHz')
        axes[i].set_xticks([]); axes[i].set_yticks([])
    fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.05).ax.tick_params(labelsize=6)
    pos = d['pos_surface_current'][IDX56].numpy()
    mag = np.linalg.norm(d['surface_currents'][IDX56].numpy(), axis=1)
    sel = pos[:, 2] > pos[:, 2].max() - 0.5  # top metal layer (parasitic)
    axes[5].scatter(pos[sel, 0], pos[sel, 1], c=np.log10(mag[sel] + 1e-6),
                    s=4, cmap='viridis', linewidths=0)
    axes[5].set_title('surface current, parasitic layer\n@ 5.6 GHz (log|J|)', fontsize=7.5)
    axes[5].set_aspect('equal'); axes[5].set_xticks([]); axes[5].set_yticks([])
    fig.suptitle('Far-field gain maps (dB, 34×34) across the five simulated frequencies '
                 '— one FMNIST+CIFAR sample', y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'example_frequencies.png', dpi=160, facecolor='white',
                bbox_inches='tight')
    plt.close(fig)
    print('  example_frequencies.png')


def fig_geometry_3d(out: Path) -> None:
    import trimesh

    sample_id = Path(first_file(MESH_SAMPLE)).stem.split('_')[-1]
    mesh_dir = REPO_ROOT / f'data/corpora/{MESH_SAMPLE}/raw/meshes/{sample_id}'
    elements = [  # (file, color, alpha, label)
        ('PEC_ground.stl', '#8a919c', 1.00, 'ground plane (PEC)'),
        ('Dielectric.stl', '#3f9b63', 0.30, 'substrate (FR4)'),
        ('Feed.stl', '#d63b3b', 1.00, 'feed column'),
        ('PEC_pixel.stl', '#d98324', 1.00, 'patch layer + feed via (PEC)'),
        ('PEC_Reflector.stl', '#2b6cb0', 1.00, 'parasitic layer (PEC)'),
    ]
    light = np.array([-.4, .35, .85])
    light /= np.linalg.norm(light)

    all_tris, all_rgba, handles = [], [], []
    for fname, color, alpha, label in elements:
        path = mesh_dir / fname
        if not path.exists():  # e.g. no parasitic layer in this sample
            continue
        m = trimesh.load(path, force='mesh')
        # subdivide so matplotlib's average-z face sorting orders the stack correctly
        v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=3.0)
        m = trimesh.Trimesh(vertices=v, faces=f, process=False)
        v = m.vertices.copy()
        if fname in ('PEC_pixel.stl', 'PEC_Reflector.stl'):
            v[:, 2] += 0.05  # avoid z-fighting with the substrate top
        all_tris.append(v[m.faces])
        base = np.array(to_rgb(color))
        inten = (0.55 + 0.45 * np.clip(m.face_normals @ light, 0, 1))[:, None]
        all_rgba.append(np.hstack([base * inten, np.full((len(m.faces), 1), alpha)]))
        handles.append(Patch(facecolor=color, label=label))
    tris = np.concatenate(all_tris)
    rgba = np.concatenate(all_rgba)

    fig = plt.figure(figsize=(8.4, 6.2))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Poly3DCollection(tris, facecolors=rgba, edgecolors='none',
                                         zsort='average'))
    pts = tris.reshape(-1, 3)
    c = pts.mean(0)
    r = (pts.max(0) - pts.min(0)).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r * 0.55, c[2] + r * 0.55)
    ax.set_box_aspect((1, 1, 0.55))
    ax.view_init(elev=28, azim=-62)
    ax.set_axis_off()
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, 0.90),
              frameon=False, fontsize=9)
    ax.set_title('Antenna geometry — FMNIST+CIFAR sample (mesh from raw/meshes/)',
                 fontsize=10, pad=0)
    fig.tight_layout()
    fig.savefig(out / 'example_geometry_3d.png', dpi=160, facecolor='white',
                bbox_inches='tight')
    plt.close(fig)
    print('  example_geometry_3d.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', type=Path, default=REPO_ROOT / 'corpora_release/assets')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f'rendering card figures into {args.out}/')
    fig_overview(args.out)
    fig_frequencies(args.out)
    fig_geometry_3d(args.out)


if __name__ == '__main__':
    main()
