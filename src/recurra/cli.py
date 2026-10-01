"""Command-line interface. Thin wrapper; all logic lives in the library."""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="recurra", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="report available backends")
    sub.add_parser("figures", help="list every figure the library can draw")

    g = sub.add_parser("generate", help="generate a synthetic CFC corpus: the signals "
                       "(one CSV each, under <out>/signals) and their ground truth")
    g.add_argument("--modality", default="pac", choices=["pac", "ppc", "aac", "ppa", "none"])
    g.add_argument("--n", type=int, default=10)
    g.add_argument("--duration", type=float, default=20.0)
    g.add_argument("--fs", type=float, default=500.0)
    g.add_argument("--out", default="outputs")
    g.add_argument("--seed", type=int, default=0)

    args = p.parse_args(argv)

    if args.command == "doctor":
        import recurra

        print(recurra.doctor())
        return 0

    if args.command == "figures":
        from .viz import list_figures

        print(list_figures())
        return 0

    if args.command == "generate":
        import pandas as pd

        from .config import set_config
        from .io.export import export_frame
        from .signals import generate_corpus

        set_config(output_dir=args.out)
        sigs = generate_corpus(
            {"alpha": [i / (args.n - 1) if args.n > 1 else 0.5 for i in range(args.n)]},
            modality=args.modality, duration=args.duration, fs=args.fs, seed=args.seed,
        )
        from .io.export import export_recording

        # The signals themselves, not only their description: before 0.25 the
        # command wrote the ground-truth table and discarded the corpus.
        for k, s in enumerate(sigs):
            name = f"signal_{k:03d}"
            export_recording(s.to_recording(name), name, subdir="signals",
                             include_timeseries=True)
        df = pd.DataFrame([s.truth_row() for s in sigs])
        df.insert(0, "signal", [f"signal_{k:03d}" for k in range(len(sigs))])
        path = export_frame(df, "corpus_truth")
        print(f"generated {len(sigs)} signals -> {args.out}/signals/; "
              f"ground truth -> {path}")
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
