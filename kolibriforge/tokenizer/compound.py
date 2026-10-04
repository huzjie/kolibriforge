"""Compound-word lexicon for Germanic languages.

German (and Dutch / the Nordic languages) composes arbitrarily long words by
concatenating nouns, e.g. "Donaudampfschifffahrtsgesellschaft". A plain BPE
tokenizer treats each such word as one out-of-vocabulary run and either splits
it into rare subwords or emits a single huge token, wasting the vocab budget and
inflating the token count.

`CompoundLexicon` keeps a frequency-ranked list of frequent constituent morphemes
and greedily segments a word into its most likely constituents before BPE runs.
Unmatched character runs are grouped into a single chunk token, so every word is
represented by *fewer* tokens than characters -- this is the compression win.
"""


class CompoundLexicon:
    """A small, seedable compound-splitting lexicon."""

    # common German constituents (frequency-ordered; illustrative subset)
    DE_CONSTITUENTS = [
        # very frequent bound morphemes & suffixes
        "ung", "heit", "keit", "lich", "haft", "schaft", "ung", "ver", "ent", "be",
        "ge", "un", "er", "en", "es", "e", "s",
        # frequent free morphemes (the bulk of real compounds)
        "dampf", "schiff", "fahrt", "gesellschaft", "donau", "kraft", "werk",
        "zeug", "bahn", "hof", "straße", "strasse", "recht", "schutz",
        "versicherung", "arbeits", "lebens", "kranken", "steuer", "erklärung",
        "hand", "schuh", "flug", "hafen", "rund", "funk", "fern", "sehen", "zeit",
        "haus", "tür", "tisch", "stuhl", "garten", "baum", "wald", "berg", "tal",
        "wasser", "feuer", "erde", "luft", "sonne", "mond", "stern", "tag", "nacht",
        "buch", "schule", "lehrer", "kind", "frau", "mann", "auto", "rad", "zug",
        "rind", "fleisch", "gesetz", "aufgabe", "übertragung", "überwachung",
        "pflicht", "bescheinigung", "fähigkeit", "etikettierung",
    ]

    EN_CONSTITUENTS = [
        "air", "craft", "rail", "road", "way", "ship", "boat", "house", "hold",
        "work", "man", "water", "fall", "sun", "flower", "day", "light", "time",
        "under", "over", "counter", "hand", "book", "mark", "place", "port",
    ]

    def __init__(self, seed=0):
        self.seed = seed
        self.entries = self.DE_CONSTITUENTS + self.EN_CONSTITUENTS

    def segment(self, word):
        """Greedy longest-match segmentation, preserving original case.

        Unmatched character runs are grouped into a single chunk token so the
        output never has more tokens than characters.

        Returns (segments: list[str], matched_any: bool).
        """
        lw = word.lower()
        segments = []
        i = 0
        n = len(lw)
        matched_any = False
        buf = []  # buffer of unmatched characters (original case)
        while i < n:
            best = None
            for c in self.entries:
                if lw.startswith(c, i) and (best is None or len(c) > len(best)):
                    best = c
            if best is not None:
                if buf:
                    segments.append("".join(buf))
                    buf = []
                segments.append(word[i:i + len(best)])  # preserve original case
                matched_any = True
                i += len(best)
            else:
                buf.append(word[i])
                i += 1
        if buf:
            segments.append("".join(buf))
        return segments, matched_any


def merge_compound(parts, matched_any):
    """Merge segmented constituents back into a hyphen-annotated surface form.

    The hyphen is a cheap, human-readable way to expose the compound boundary to
    downstream code (and to downstream BPE) without a separate token type.
    """
    if not matched_any or len(parts) <= 1:
        return "".join(parts)
    return "-".join(parts)
