"""Regression tests for NCBI Disease Corpus document offsets.

The corpus gives entity offsets relative to one document string.  In this
project that string is constructed as ``title + " " + abstract``.  These
tests use PMID 9050866 as a fixed example so that a later preprocessing
change cannot silently break entity alignment.
"""

from pathlib import Path
import unittest


PMID = "9050866"
RAW_DATA = (
    Path(__file__).resolve().parent
    / "Dataset"
    / "RawData"
    / "NCBIdevelopset_validation.txt"
)
EXPECTED_MENTIONS = (
    (4, 25, "ataxia-telangiectasia"),
    (152, 173, "ataxia-telangiectasia"),
    (686, 707, "ataxia-telangiectasia"),
)


def load_pubtator_record(path: Path, pmid: str):
    """Return the title, abstract, and annotations for one PubTator record."""
    title = None
    abstract = None
    annotations = []

    with path.open(encoding="utf-8") as source:
        for raw_line in source:
            line = raw_line.rstrip("\r\n")
            if line.startswith(f"{pmid}|t|"):
                title = line.split("|", 2)[2]
            elif line.startswith(f"{pmid}|a|"):
                abstract = line.split("|", 2)[2]
            elif line.startswith(f"{pmid}\t"):
                fields = line.split("\t")
                annotations.append((int(fields[1]), int(fields[2]), fields[3]))

    if title is None or abstract is None:
        raise ValueError(f"PMID {pmid} was not found in {path}")

    return title, abstract, tuple(annotations)


def join_title_and_abstract(title: str, abstract: str) -> str:
    """Build the canonical document text used by the corpus offsets."""
    return f"{title} {abstract}"


class TestTitleAbstractJoining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.title, cls.abstract, cls.annotations = load_pubtator_record(
            RAW_DATA, PMID
        )
        cls.text = join_title_and_abstract(cls.title, cls.abstract)

    def test_fixture_annotations_are_the_expected_example(self):
        self.assertEqual(self.annotations, EXPECTED_MENTIONS)

    def test_abstract_starts_after_one_separator_character(self):
        # The second mention is at abstract-local [19, 40).  Its global start
        # is therefore len(title) + 1 + 19 == 152.
        local_start = self.abstract.index("ataxia-telangiectasia")
        global_start = len(self.title) + 1 + local_start

        self.assertEqual(local_start, 19)
        self.assertEqual(global_start, 152)
        self.assertEqual(self.text[len(self.title)], " ")
        self.assertEqual(self.text[len(self.title) + 1 :], self.abstract)

    def test_all_annotated_spans_match_the_joined_text(self):
        for start, end, mention in EXPECTED_MENTIONS:
            with self.subTest(start=start, end=end):
                self.assertEqual(self.text[start:end], mention)

    def test_joining_without_a_separator_breaks_abstract_offsets(self):
        incorrectly_joined = self.title + self.abstract
        start, end, mention = EXPECTED_MENTIONS[1]
        self.assertNotEqual(incorrectly_joined[start:end], mention)


if __name__ == "__main__":
    unittest.main(verbosity=2)
