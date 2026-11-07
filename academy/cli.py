"""ML Learning & Research Academy — CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import db, progress
from .content.loader import load_all_modules, get_module, Module
from .exercises.engine import get_exercise, grade_exercise


def _modules():
    db.init_db()
    return load_all_modules()


def _print_module_brief(m: Module):
    print(f"\n{'='*70}\n{m.id} — {m.title}")
    print(f"Phase {m.phase} | {m.difficulty} | ~{m.estimated_hours}h | "
          f"{len(m.exercises)} exercises")
    print(f"{'='*70}")
    print("\nObjectives:")
    for o in m.objectives:
        print(f"  - {o}")
    print("\nKey concepts:")
    for c in m.concepts:
        print(f"  - {c}")
    if m.formulas:
        print("\nFormulas:")
        for f in m.formulas:
            print(f"  * {f.get('name','')}: {f.get('latex', f.get('formula',''))}")
            if f.get("note"):
                print(f"    {f['note']}")


def _print_module_full(m: Module):
    _print_module_brief(m)
    for name in ["Penjelasan", "Konsep Penting", "Formula", "Contoh Kasus",
                 "Implementasi", "Referensi"]:
        if name in m.sections:
            print(f"\n## {name}\n{m.sections[name]}")
    if m.exercises:
        print("\n## Latihan")
        for ex in m.exercises:
            print(f"\n[{ex['id']}] ({ex['type']}) {ex['question'][:200]}")


def cmd_list(args):
    mods = _modules()
    if args.phase:
        mods = [m for m in mods if m.phase == args.phase]
    current_phase = None
    for m in mods:
        if m.phase != current_phase:
            current_phase = m.phase
            print(f"\n--- PHASE {current_phase} ---")
        mp = progress.get_progress(m.id)
        status = mp.status if mp else "not_started"
        score = progress.module_score(m.id)
        score_s = f" (score {score:.0f}%)" if score else ""
        print(f"  {m.id:<40} {m.difficulty:<10} {status}{score_s}")
    print(f"\n{len(mods)} modules")


def cmd_show(args):
    mods = _modules()
    try:
        m = get_module(args.module_id, mods)
    except KeyError as e:
        print(e); sys.exit(1)
    if args.full:
        _print_module_full(m)
    else:
        _print_module_brief(m)


def cmd_learn(args):
    mods = _modules()
    try:
        m = get_module(args.module_id, mods)
    except KeyError as e:
        print(e); sys.exit(1)
    progress.mark_started(m.id, m.phase)
    _print_module_full(m)
    print(f"\nMarked '{m.id}' as in_progress.")
    print(f"Run exercises:  python -m academy exercise {m.id}")
    print(f"Complete:       python -m academy complete {m.id}")


def cmd_complete(args):
    mods = _modules()
    try:
        m = get_module(args.module_id, mods)
    except KeyError as e:
        print(e); sys.exit(1)
    score = progress.module_score(m.id)
    progress.mark_completed(m.id, m.phase, score)
    print(f"Module '{m.id}' marked completed (exercise score: {score:.0f}%).")


def _print_exercise_prompt(ex):
    print(f"\n--- Exercise {ex['id']} ({ex['type']}) ---")
    print(ex["question"])
    if ex.get("starter_code"):
        print("\nStarter code:\n" + ex["starter_code"])
    if ex.get("options"):
        for i, opt in enumerate(ex["options"]):
            print(f"  {i}) {opt}")
    if ex.get("hints"):
        print("Hints: " + " | ".join(ex["hints"]))
    if ex.get("rubric"):
        print("Rubric (manual grading):")
        for r in ex["rubric"]:
            print(f"  - {r}")


def _read_multiline(prompt="Answer (empty line to finish): ") -> str:
    print(prompt)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "" and lines:
            break
        lines.append(line)
    return "\n".join(lines)


def _run_exercise_interactive(m: Module):
    exs = m.exercises
    if not exs:
        print("No exercises in this module."); return
    total_score = total_max = 0.0
    for ex in exs:
        _print_exercise_prompt(ex)
        etype = ex["type"]
        if etype == "open":
            ans = _read_multiline("Your answer: ")
            res = grade_exercise(ex, ans)
            progress.record_attempt(m.id, ex["id"], etype, ans, 0.0,
                                    ex.get("points", 1), False, res.detail)
            print("Stored for manual grading (rubric above).")
            continue
        if etype == "coding":
            print("\nWrite your solution to a file, then enter its path "
                  "(or paste code and end with an empty line):")
            raw = _read_multiline("> ")
            p = Path(raw.strip().splitlines()[-1]) if raw.strip() else None
            if p and p.exists() and p.suffix == ".py":
                code = p.read_text()
            else:
                code = raw
            res = grade_exercise(ex, code)
        else:
            ans = input("Your answer: ").strip()
            res = grade_exercise(ex, ans)
        progress.record_attempt(m.id, ex["id"], etype,
                                ans if etype != "coding" else "<code>",
                                res.score, res.max_score, res.passed, res.detail)
        mark = "PASS" if res.passed else "FAIL"
        print(f"[{mark}] score {res.score}/{res.max_score}")
        if res.detail:
            print(f"  {res.detail[:500]}")
        total_score += res.score; total_max += res.max_score
    if total_max:
        print(f"\nModule exercise score: {total_score}/{total_max} "
              f"({100*total_score/total_max:.0f}%)")


def cmd_exercise(args):
    mods = _modules()
    try:
        m = get_module(args.module_id, mods)
    except KeyError as e:
        print(e); sys.exit(1)
    if args.exercise_id:
        try:
            ex = get_exercise(m, args.exercise_id)
        except KeyError as e:
            print(e); sys.exit(1)
        _print_exercise_prompt(ex)
        print("\nSubmit with: python -m academy submit "
              f"{m.id} {ex['id']} --file solution.py")
        return
    _run_exercise_interactive(m)


def cmd_submit(args):
    mods = _modules()
    try:
        m = get_module(args.module_id, mods)
        ex = get_exercise(m, args.exercise_id)
    except KeyError as e:
        print(e); sys.exit(1)
    code = Path(args.file).read_text()
    res = grade_exercise(ex, code)
    progress.record_attempt(m.id, ex["id"], ex["type"], "<code>",
                            res.score, res.max_score, res.passed, res.detail)
    print(f"Score: {res.score}/{res.max_score} "
          f"({'PASS' if res.passed else 'FAIL'})")
    if res.detail:
        print(res.detail[:1000])


def cmd_progress(args):
    mods = _modules()
    rows = progress.all_progress()
    by_id = {r.module_id: r for r in rows}
    phases = sorted({m.phase for m in mods})
    for ph in phases:
        pmods = [m for m in mods if m.phase == ph]
        done = sum(1 for m in pmods if by_id.get(m.id) and by_id[m.id].status == "completed")
        print(f"Phase {ph}: {done}/{len(pmods)} completed")
    print()
    for m in mods:
        mp = by_id.get(m.id)
        status = mp.status if mp else "not_started"
        if args.all or status != "not_started":
            score = progress.module_score(m.id)
            print(f"  {m.id:<40} {status:<12} score={score:.0f}%")


def cmd_note(args):
    if args.action == "add":
        text = " ".join(args.text)
        progress.add_note(args.module_id, text)
        print("Note saved.")
    else:
        notes = progress.get_notes(args.module_id)
        for n in notes:
            print(f"[{n.module_id}] {n.text}")


def cmd_dataset(args):
    from .datasets.manager import list_datasets, dataset_info, load_dataset, DATASETS
    if args.action == "list":
        for spec in list_datasets():
            flag = "download" if spec.download else ("synthetic" if spec.generator else "bundled")
            print(f"{spec.name:<28} {spec.stage:<12} {flag:<10} {spec.task}")
        return
    if args.action == "info":
        s = dataset_info(args.name)
        for k, v in s.__dict__.items():
            print(f"{k}: {v}")
        return
    if args.action == "fetch":
        df = load_dataset(args.name)
        out = args.out or str(Path("data/downloaded") / f"{args.name}.csv")
        if isinstance(df, dict):
            print("Image-array dataset; keys:", list(df.keys()))
            return
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved {len(df)} rows x {len(df.columns)} cols -> {out}")
        return
    if args.action == "generate":
        df = load_dataset(args.name, **(json.loads(args.kwargs) if args.kwargs else {}))
        print(df.head())
        print(f"shape: {df.shape}")
        return


def cmd_project(args):
    from .projects.specs import list_projects, get_project
    if args.action == "list":
        for p in list_projects():
            print(f"{p.id:<22} {p.stage:<12} phase{p.phase}  {p.title}")
        return
    if args.action == "show":
        p = get_project(args.project_id)
        print(f"\n{p.id} — {p.title} ({p.stage}, phase {p.phase})")
        print(f"\nBrief: {p.brief}\n")
        print("Objectives:")
        for o in p.objectives: print(f"  - {o}")
        print(f"\nDataset: {p.dataset}")
        print("\nTask breakdown:")
        for t in p.task_breakdown: print(f"  - {t}")
        print("\nStarter code:\n```python\n" + p.starter_code + "```")
        print("\nAcceptance criteria:")
        for a in p.acceptance_criteria: print(f"  - [ ] {a}")
        print(f"\nMetrics: {', '.join(p.metrics)}")
        print(f"Deliverables: {', '.join(p.deliverables)}")
        print(f"\nResearch extension: {p.research_extension}")
        return
    if args.action == "check":
        p = get_project(args.project_id)
        print(f"Manual acceptance checklist for {p.id}:")
        for a in p.acceptance_criteria:
            print(f"  [ ] {a}")
        return


def cmd_experiment(args):
    if args.action == "log":
        progress.log_experiment(
            project_id=args.project_id, name=args.name,
            params=json.loads(args.params or "{}"),
            metrics=json.loads(args.metrics or "{}"),
            artifacts=json.loads(args.artifacts or "[]"),
            notes=args.notes or "",
        )
        print("Experiment logged.")
    else:
        for e in progress.list_experiments(args.project_id):
            print(f"[{e.id}] {e.project_id}/{e.name} "
                  f"metrics={e.metrics} params={e.params} ({e.created_at})")


def cmd_search(args):
    mods = _modules()
    q = args.query.lower()
    for m in mods:
        hay = (m.title + " " + m.id + " " + " ".join(m.concepts) +
               " " + " ".join(m.objectives)).lower()
        if q in hay:
            print(f"{m.id:<40} phase{m.phase}  {m.title}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="academy",
        description="ML Learning & Research Academy CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="List modules")
    p.add_argument("--phase", type=int)
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("show", help="Show module brief")
    p.add_argument("module_id")
    p.add_argument("--full", action="store_true")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("learn", help="Print full module content, mark started")
    p.add_argument("module_id")
    p.set_defaults(fn=cmd_learn)

    p = sub.add_parser("complete", help="Mark module completed")
    p.add_argument("module_id")
    p.set_defaults(fn=cmd_complete)

    p = sub.add_parser("exercise", help="Run/list exercises for a module")
    p.add_argument("module_id")
    p.add_argument("exercise_id", nargs="?")
    p.set_defaults(fn=cmd_exercise)

    p = sub.add_parser("submit", help="Submit a coding solution file")
    p.add_argument("module_id")
    p.add_argument("exercise_id")
    p.add_argument("--file", required=True)
    p.set_defaults(fn=cmd_submit)

    p = sub.add_parser("progress", help="Show learning progress")
    p.add_argument("--all", action="store_true")
    p.set_defaults(fn=cmd_progress)

    p = sub.add_parser("note", help="Manage notes")
    p.add_argument("action", choices=["add", "list"])
    p.add_argument("module_id", nargs="?")
    p.add_argument("text", nargs="*")
    p.set_defaults(fn=cmd_note)

    p = sub.add_parser("dataset", help="Dataset manager")
    p.add_argument("action", choices=["list", "info", "fetch", "generate"])
    p.add_argument("name", nargs="?")
    p.add_argument("--out")
    p.add_argument("--kwargs", default=None)
    p.set_defaults(fn=cmd_dataset)

    p = sub.add_parser("project", help="Project-based learning")
    p.add_argument("action", choices=["list", "show", "check"])
    p.add_argument("project_id", nargs="?")
    p.set_defaults(fn=cmd_project)

    p = sub.add_parser("experiment", help="Log/list experiments")
    p.add_argument("action", choices=["log", "list"])
    p.add_argument("project_id", nargs="?")
    p.add_argument("name", nargs="?")
    p.add_argument("--params", default="{}")
    p.add_argument("--metrics", default="{}")
    p.add_argument("--artifacts", default="[]")
    p.add_argument("--notes", default="")
    p.set_defaults(fn=cmd_experiment)

    p = sub.add_parser("search", help="Search modules")
    p.add_argument("query")
    p.set_defaults(fn=cmd_search)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
