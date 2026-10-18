"""Build the two deterministic figures used by the public research pages.

Run from the repository root with ``python scripts/build_research_figures.py``.
Only the exact four chart paths listed below are written. The source JSON files
remain the authority for the underlying observations.
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt'] = 'localvoice-public-research-v1'
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SHORT_TURN_PATH = ROOT / 'research' / 'short-turn-cases.json'
SCREEN_PATH = ROOT / 'research' / 'candidate-screen-cases.json'
ASSET_DIR = ROOT / 'docs' / 'assets'

BLUE = '#0072B2'
ORANGE = '#D55E00'
INK = '#1F2937'
MUTED = '#586474'
GRID = '#D9DEE5'
GUIDE = '#6B7280'
PAPER = '#FFFFFF'


def read_json(path):
    with path.open('r', encoding='utf-8') as stream:
        return json.load(stream)


def validate_short_turn(doc):
    if doc.get('schema_version') != 1 or doc.get('study') != 'short-turn-streaming':
        raise ValueError('short-turn-cases.json is not schema version 1 short-turn data')
    rows = doc.get('cases')
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError('short-turn chart requires exactly 16 cases')
    if doc.get('complete_budget_ms') != 160 or doc.get('unfinished_minimum_hold_ms') != 2000:
        raise ValueError('short-turn timing guides differ from the frozen protocol')
    seen = set()
    for row in rows:
        if row.get('id') in seen:
            raise ValueError(f'duplicate short-turn case id: {row.get("id")}')
        seen.add(row.get('id'))
        if row.get('authored_role') not in {'complete', 'incomplete_prefix'}:
            raise ValueError(f'unknown short-turn role: {row.get("authored_role")}')
        if (not isinstance(row.get('extra_wall_ms'), (int, float))
                or not math.isfinite(row['extra_wall_ms']) or row['extra_wall_ms'] < 0):
            raise ValueError(f'invalid observed wall time for {row.get("id")}')
    counts = {role: sum(row['authored_role'] == role for row in rows)
              for role in ('complete', 'incomplete_prefix')}
    if counts != {'complete': 8, 'incomplete_prefix': 8}:
        raise ValueError(f'unexpected short-turn role balance: {counts}')
    pairs = {}
    for row in rows:
        pair = row['id'].split('-', 1)[0]
        pairs.setdefault(pair, set()).add(row['authored_role'])
    if len(pairs) != 8 or any(roles != {'complete', 'incomplete_prefix'} for roles in pairs.values()):
        raise ValueError('short-turn data must contain eight complete/incomplete pairs')
    return sorted(rows, key=lambda row: (row['id'].split('-')[0], row['id']))


def validate_screen(doc):
    if doc.get('schema_version') != 1 or doc.get('study') != 'fixed-endpoint-audio-or':
        raise ValueError('candidate-screen-cases.json is not the frozen fixed-endpoint screen')
    if doc.get('decision') != 'rejected':
        raise ValueError('candidate screen must remain labeled rejected')
    rows = doc.get('cases')
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError('candidate screen chart requires exactly 16 cases')
    by_role = {}
    seen = set()
    for row in rows:
        if row.get('id') in seen:
            raise ValueError(f'duplicate candidate-screen case id: {row.get("id")}')
        seen.add(row.get('id'))
        role = row.get('authored_role')
        if role not in {'complete', 'incomplete_prefix'}:
            raise ValueError(f'unknown candidate-screen role: {role}')
        if (type(row.get('namo_label')) is not int or row['namo_label'] not in {0, 1}
                or type(row.get('combined_positive')) is not bool):
            raise ValueError(f'invalid decision label for {row.get("id")}')
        if not isinstance(row.get('audio_score'), (int, float)) or not math.isfinite(row['audio_score']):
            raise ValueError(f'invalid audio score for {row.get("id")}')
        by_role.setdefault(role, []).append(row)
    if any(len(by_role.get(role, [])) != 8 for role in ('complete', 'incomplete_prefix')):
        raise ValueError('candidate screen requires 8 complete and 8 unfinished cases')
    counts = {
        'complete_namo_positive': sum(row['namo_label'] == 1 for row in by_role['complete']),
        'complete_or_positive': sum(row['combined_positive'] for row in by_role['complete']),
        'prefix_namo_held': sum(row['namo_label'] == 0 for row in by_role['incomplete_prefix']),
        'prefix_or_held': sum(not row['combined_positive'] for row in by_role['incomplete_prefix']),
    }
    expected = {
        'complete_namo_positive': 5,
        'complete_or_positive': 7,
        'prefix_namo_held': 8,
        'prefix_or_held': 4,
    }
    if counts != expected:
        raise ValueError(f'candidate screen counts changed from the reviewed result: {counts}')
    return by_role, counts


def style_axis(ax):
    ax.set_facecolor(PAPER)
    ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRID)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.title.set_color(INK)
    ax.yaxis.label.set_color(INK)


def build_short_turn(rows, output_svg, output_png):
    complete = [row for row in rows if row['authored_role'] == 'complete']
    unfinished = [row for row in rows if row['authored_role'] == 'incomplete_prefix']
    complete_by_pair = {row['id'].split('-')[0]: row for row in complete}
    unfinished_by_pair = {row['id'].split('-')[0]: row for row in unfinished}
    pair_ids = sorted(complete_by_pair)
    if pair_ids != sorted(unfinished_by_pair):
        raise ValueError('short-turn input is missing a complete/incomplete pair')

    fig, (ax_all, ax_zoom) = plt.subplots(
        2, 1, figsize=(14.4, 8.0), gridspec_kw={'height_ratios': [2.1, 1.0]},
        constrained_layout=True)
    fig.patch.set_facecolor(PAPER)
    fig.suptitle('Short-turn release delay after the first raw endpoint',
                 fontsize=17, fontweight='bold', color=INK, x=0.04, ha='left')
    fig.text(
        0.04, 0.944,
        'All 16 cases · one synthetic development run · authored task-intent labels, not human/acoustic ground truth · not end-to-end latency',
        fontsize=10, color=MUTED, ha='left')

    x = np.arange(len(pair_ids), dtype=float)
    x_complete = x - 0.13
    x_prefix = x + 0.13
    y_complete = np.array([complete_by_pair[pair]['extra_wall_ms'] for pair in pair_ids])
    y_prefix = np.array([unfinished_by_pair[pair]['extra_wall_ms'] for pair in pair_ids])
    budget = 160
    min_hold = 2000

    style_axis(ax_all)
    ax_all.scatter(x_complete, y_complete, s=64, marker='o', color=BLUE,
                   edgecolors='white', linewidths=0.8, label='Authored complete', zorder=3)
    ax_all.scatter(x_prefix, y_prefix, s=66, marker='^', color=ORANGE,
                   edgecolors='white', linewidths=0.8, label='Authored unfinished prefix', zorder=3)
    ax_all.axhline(budget, color=GUIDE, linestyle='--', linewidth=1.2, zorder=1)
    ax_all.axhline(min_hold, color=ORANGE, linestyle=':', linewidth=1.5, zorder=1)
    ax_all.text(7.58, budget + 38, '160 ms complete-turn budget', color=MUTED,
                fontsize=9, ha='right', va='bottom')
    ax_all.text(7.58, min_hold + 38, '2,000 ms unfinished-hold guide', color=ORANGE,
                fontsize=9, ha='right', va='bottom')
    ax_all.set_xlim(-0.55, 7.65)
    ax_all.set_ylim(0, 2750)
    ax_all.set_xticks(x, [f'{int(pair[-2:]):02d}' for pair in pair_ids])
    ax_all.set_xlabel('Authored pair ID (see case table for exact text)', labelpad=8)
    ax_all.set_ylabel('Additional wall time to capture-gate release (ms)')
    ax_all.legend(loc='upper left', frameon=False, ncol=2, fontsize=9)

    # Magnify the five near-zero complete observations without changing the
    # full-scale panel above or hiding the three slow complete observations.
    zoom_pairs = [pair for pair in pair_ids
                  if complete_by_pair[pair]['extra_wall_ms'] <= 300]
    zoom_x = np.arange(len(zoom_pairs), dtype=float)
    zoom_y = np.array([complete_by_pair[pair]['extra_wall_ms'] for pair in zoom_pairs])
    style_axis(ax_zoom)
    ax_zoom.scatter(zoom_x, zoom_y, s=68, marker='o', color=BLUE,
                    edgecolors='white', linewidths=0.8, zorder=3)
    ax_zoom.axhline(budget, color=GUIDE, linestyle='--', linewidth=1.2)
    for xpos, value in zip(zoom_x, zoom_y):
        ax_zoom.annotate(f'{value:.3f}', (xpos, value), xytext=(0, 7),
                         textcoords='offset points', ha='center', fontsize=8, color=INK)
    ax_zoom.set_xlim(-0.5, max(0.5, len(zoom_pairs) - 0.5))
    ax_zoom.set_ylim(0, 210)
    ax_zoom.set_xticks(zoom_x, [pair[-2:] for pair in zoom_pairs])
    ax_zoom.set_xlabel('Pair ID · complete cases at or below 300 ms only', labelpad=6)
    ax_zoom.set_ylabel('Additional wall time (ms)')
    ax_zoom.set_title('Near-zero detail; the full-scale panel retains every case',
                      loc='left', fontsize=10, pad=7)

    fig.savefig(output_svg, format='svg', metadata={
        'Date': None,
        'Creator': 'Turnwave research figure generator',
        'Title': 'Short-turn release delay after the first raw endpoint',
    })
    fig.savefig(output_png, format='png', dpi=150, metadata={
        'Software': 'Turnwave research figure generator',
    })
    # SVG path tokens use whitespace separators; normalize generated line tails.
    output_svg.write_bytes(('\n'.join(line.rstrip() for line in
                            output_svg.read_text(encoding='utf-8').splitlines()) + '\n').encode('utf-8'))
    plt.close(fig)


def build_completion_screen(by_role, counts, output_svg, output_png):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.4), sharey=True,
                             gridspec_kw={'wspace': 0.18})
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.25, top=0.82, wspace=0.18)
    fig.patch.set_facecolor(PAPER)
    fig.suptitle('The fixed audio OR rule trades unfinished-prefix holds for more positive decisions',
                 fontsize=15, fontweight='bold', color=INK, x=0.04, ha='left', y=0.96)
    fig.text(
        0.04, 0.89,
        'Rejected at frozen first-endpoint snapshots · descriptive counts only (n=8 per group), not population rates',
        fontsize=9.5, color=MUTED, ha='left')

    groups = [
        ('Complete intents positive', 'complete',
         ('Namo', 'Namo OR Turn-1\npublisher example > 0.5'),
         (counts['complete_namo_positive'], counts['complete_or_positive'])),
        ('Unfinished prefixes held', 'incomplete_prefix',
         ('Namo', 'Namo OR Turn-1\npublisher example > 0.5'),
         (counts['prefix_namo_held'], counts['prefix_or_held'])),
    ]
    for idx, (ax, (title, role, labels, values)) in enumerate(zip(axes, groups)):
        style_axis(ax)
        bars = ax.bar([0, 1], values, width=0.58, color=[BLUE, ORANGE], zorder=3)
        ax.set_title(title, loc='left', fontsize=11, pad=12, fontweight='bold')
        ax.set_xticks([0, 1], labels, fontsize=9)
        ax.set_ylim(0, 9.4)
        ax.set_yticks(range(0, 9))
        if idx == 0:
            ax.set_ylabel('Cases meeting the panel outcome (of 8)')
        else:
            ax.tick_params(labelleft=False)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.16,
                    f'{value}/8  ({value / 8:.1%})', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color=INK)
    fig.text(
        0.04, 0.045,
        'OR rule: current ready Namo label = 1 OR final Turn-1 probability > 0.5 (publisher example comparator; not tuned).',
        fontsize=9, color=MUTED, ha='left')
    fig.savefig(output_svg, format='svg', metadata={
        'Date': None,
        'Creator': 'Turnwave research figure generator',
        'Title': 'Rejected fixed-endpoint audio OR screen',
    })
    fig.savefig(output_png, format='png', dpi=150, metadata={
        'Software': 'Turnwave research figure generator',
    })
    output_svg.write_bytes(('\n'.join(line.rstrip() for line in
                            output_svg.read_text(encoding='utf-8').splitlines()) + '\n').encode('utf-8'))
    plt.close(fig)


def main():
    short_doc = read_json(SHORT_TURN_PATH)
    screen_doc = read_json(SCREEN_PATH)
    short_rows = validate_short_turn(short_doc)
    screen_roles, counts = validate_screen(screen_doc)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    build_short_turn(
        short_rows,
        ASSET_DIR / 'short-turn-delay.svg',
        ASSET_DIR / 'short-turn-delay.png',
    )
    build_completion_screen(
        screen_roles,
        counts,
        ASSET_DIR / 'completion-screen.svg',
        ASSET_DIR / 'completion-screen.png',
    )
    print('Built the four reviewed chart assets from the frozen public case JSON.')


if __name__ == '__main__':
    main()
