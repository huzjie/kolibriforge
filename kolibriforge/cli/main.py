"""Command-line entrypoint.

Subcommands: doctor / tokenize / route / abstain / train / evaluate / serve.
"""
import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="kolibriforge",
        description="Sovereign MoE reasoning LLM with honest abstention (Kolibri-1 direction)")
    sub = parser.add_subparsers(dest="cmd")

    p_doc = sub.add_parser("doctor", help="check components / backends / benchmarks")
    p_doc.add_argument("--config", default=None)

    p_tok = sub.add_parser("tokenize", help="run the UniBPE compound-aware tokenizer")
    p_tok.add_argument("text")
    p_tok.add_argument("--config", default=None)

    p_rt = sub.add_parser("route", help="route a prompt to a reasoning-depth budget")
    p_rt.add_argument("query")
    p_rt.add_argument("--config", default=None)

    p_ab = sub.add_parser("abstain", help="run Merlin-Arthur abstention gate on a question")
    p_ab.add_argument("question")
    p_ab.add_argument("--config", default=None)

    p_tr = sub.add_parser("train", help="RL-train the honesty policy on synthetic data")
    p_tr.add_argument("--config", default=None)
    p_tr.add_argument("--steps", type=int, default=None)

    p_ev = sub.add_parser("evaluate", help="run the honesty / math / context benchmarks")
    p_ev.add_argument("--config", default=None)

    p_serve = sub.add_parser("serve", help="start the OpenAI-compatible HTTP server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--config", default=None)

    args = parser.parse_args(argv)
    from ..config import load_config
    cfg = load_config(args.config)

    if args.cmd == "doctor":
        from ..doctor import run_doctor
        ok = run_doctor(cfg)
        sys.exit(0 if ok else 1)
    elif args.cmd == "tokenize":
        from ..tokenizer import build_unibpe
        tok = build_unibpe(cfg)
        print(json.dumps(tok.tokenize_with_info(args.text), ensure_ascii=False, indent=2))
    elif args.cmd == "route":
        from ..reason.budget import ReasoningBudget
        rb = ReasoningBudget(cfg)
        print(json.dumps(rb.plan(args.query), ensure_ascii=False, indent=2))
    elif args.cmd == "abstain":
        from ..engine import KolibriEngine
        eng = KolibriEngine(cfg)
        print(json.dumps(eng.abstain(args.question), ensure_ascii=False, indent=2))
    elif args.cmd == "train":
        from ..train.trainer import run_training
        run_training(cfg, steps=args.steps)
    elif args.cmd == "evaluate":
        from ..bench.runner import run_all
        run_all(cfg)
    elif args.cmd == "serve":
        from ..serving.server import serve
        serve(cfg, args.host, args.port)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
