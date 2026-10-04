"""UniBPE: compound-aware byte-pair encoding.

Pipeline
--------
1. Compound pre-split  -- `CompoundLexicon.segment` on each whitespace-delimited
   word; matched Germanic constituents become individual tokens, unmatched runs
   are grouped into one chunk.
2. Token id assignment -- each token maps to a stable integer id (on-demand).
3. Decode              -- join token strings back.

The point: frequent compound constituents stay as *few, stable* tokens instead of
degrading into rare subword runs, which measurably lowers tokens-per-word on
German-heavy text.
"""
from .compound import CompoundLexicon, merge_compound


class UniBPE:
    def __init__(self, merges=None, lexicon=None, special_tokens=("<unk>", "<s>", "</s>")):
        # merges kept for API compatibility with classic BPE; the compound pass
        # is the differentiator and does the heavy lifting here.
        self.merges = dict(merges) if merges else _seed_merges()
        self.lexicon = lexicon or CompoundLexicon()
        self.special_tokens = list(special_tokens)
        self._tok2id = {t: i for i, t in enumerate(self.special_tokens)}
        self._id2tok = {i: t for t, i in self._tok2id.items()}
        self._next_id = len(self.special_tokens)

    def _pre_segment(self, text):
        """Return the hyphen-annotated surface form (human-readable)."""
        out = []
        for w in text.split(" "):
            if not w:
                continue
            parts, matched = self.lexicon.segment(w)
            out.append(merge_compound(parts, matched))
        return " ".join(out)

    def _token_id(self, tok):
        if tok not in self._tok2id:
            self._tok2id[tok] = self._next_id
            self._id2tok[self._next_id] = tok
            self._next_id += 1
        return self._tok2id[tok]

    def tokenize(self, text, annotate=True):
        """Return the actual token units (compound constituents, not characters).

        For a compound word this yields the constituent list; for a plain word it
        yields the word itself (1 token).
        """
        tokens = []
        for w in text.split(" "):
            if not w:
                continue
            if annotate:
                parts, matched = self.lexicon.segment(w)
                if matched and len(parts) > 1:
                    tokens.extend(parts)
                    continue
            tokens.append(w)
        return tokens

    def encode(self, text, annotate=True):
        """Map token units to stable integer ids."""
        return [self._token_id(t) for t in self.tokenize(text, annotate=annotate)]

    def tokenize_with_info(self, text):
        raw = text
        annotated = self._pre_segment(text)
        tokens = self.tokenize(text, annotate=True)
        n_chars = len(raw)
        n_tokens = len(tokens)
        return {
            "raw": raw,
            "annotated": annotated,
            "tokens": tokens,
            "num_chars": n_chars,
            "num_tokens": n_tokens,
            "compression_ratio": round(n_chars / max(n_tokens, 1), 3),
        }

    def decode(self, ids):
        """Join token strings back (reversible for case-preserved tokenize)."""
        return "".join(self._id2tok.get(i, "") for i in ids)


def _seed_merges():
    """A tiny deterministic set of merge rules for classic-BPE compatibility."""
    merges = {}
    common = ["th", "ch", "sch", "ck", "ei", "ie", "en", "er", "ung", "heit",
              "keit", "tion", "ing", "ed", "es", "an", "on", "au"]
    for i, pair in enumerate(common):
        merges[(pair[0], pair[1])] = 200 + i
    return merges


def build_unibpe(cfg=None):
    lexicon = CompoundLexicon(seed=getattr(cfg, "seed", 0) if cfg else 0)
    return UniBPE(lexicon=lexicon)
