from app.sign import WordBuilder


def feed_run(wb: WordBuilder, letter, conf, start, end, step=0.2):
    """Feed `letter` every `step` seconds from start to end (inclusive); return committed chars."""
    out = []
    n = round((end - start) / step)
    for i in range(n + 1):
        c = wb.feed(letter, conf, round(start + i * step, 3))
        if c:
            out.append(c)
    return out


def test_hold_commits_once():
    wb = WordBuilder()
    assert feed_run(wb, "A", 0.9, 0.0, 3.0) == ["A"]
    assert wb.text == "A"


def test_short_hold_not_committed():
    wb = WordBuilder()
    assert feed_run(wb, "A", 0.9, 0.0, 0.6) == []
    assert wb.text == ""


def test_commit_exactly_at_hold_time():
    wb = WordBuilder()
    assert wb.feed("A", 0.9, 0.0) is None
    assert wb.feed("A", 0.9, 0.79) is None
    assert wb.feed("A", 0.9, 0.8) == "A"


def test_repeated_letter_needs_release():
    wb = WordBuilder()
    feed_run(wb, "L", 0.9, 0.0, 1.0)
    # Same letter keeps being held: no second L.
    feed_run(wb, "L", 0.9, 1.2, 3.0)
    assert wb.text == "L"
    # Brief hand drop (< space_s), then hold again: second L.
    wb.feed(None, 0.0, 3.2)
    feed_run(wb, "L", 0.9, 3.4, 4.4)
    assert wb.text == "LL"


def test_repeated_letter_via_other_shape():
    wb = WordBuilder()
    feed_run(wb, "L", 0.9, 0.0, 1.0)
    feed_run(wb, "V", 0.9, 1.2, 2.2)
    feed_run(wb, "L", 0.9, 2.4, 3.4)
    assert wb.text == "LVL"


def test_low_confidence_ignored():
    wb = WordBuilder()
    assert feed_run(wb, "A", 0.59, 0.0, 3.0) == []
    assert wb.text == ""


def test_low_confidence_interrupts_hold():
    wb = WordBuilder()
    wb.feed("A", 0.9, 0.0)
    wb.feed("A", 0.3, 0.5)  # treated as no hand: resets the hold
    assert wb.feed("A", 0.9, 0.9) is None
    assert wb.feed("A", 0.9, 1.5) is None
    assert wb.feed("A", 0.9, 1.8) == "A"


def test_flickering_letters_not_committed():
    wb = WordBuilder()
    for i in range(20):
        wb.feed("AB"[i % 2], 0.9, i * 0.2)
    assert wb.text == ""


def test_space_after_pause():
    wb = WordBuilder()
    feed_run(wb, "A", 0.9, 0.0, 1.0)
    assert wb.feed(None, 0.0, 1.5) is None  # 0.5 s since last seen (1.0)
    assert wb.feed(None, 0.0, 2.4) is None
    assert wb.feed(None, 0.0, 2.5) == " "
    assert wb.text == "A "


def test_no_double_space():
    wb = WordBuilder()
    feed_run(wb, "A", 0.9, 0.0, 1.0)
    spaces = [wb.feed(None, 0.0, t) for t in (2.6, 3.0, 5.0, 10.0)]
    assert spaces == [" ", None, None, None]
    assert wb.text == "A "


def test_no_leading_space():
    wb = WordBuilder()
    for t in (0.0, 2.0, 5.0, 9.0):
        assert wb.feed(None, 0.0, t) is None
    assert wb.text == ""


def test_word_then_space_then_word():
    wb = WordBuilder()
    feed_run(wb, "H", 0.9, 0.0, 1.0)
    feed_run(wb, "I", 0.9, 1.2, 2.2)
    for t in (2.4, 3.0, 3.8):
        wb.feed(None, 0.0, t)
    feed_run(wb, "Y", 0.9, 4.0, 5.0)
    assert wb.text == "HI Y"


def test_custom_thresholds():
    wb = WordBuilder(hold_s=0.2, space_s=0.5, min_confidence=0.9)
    assert feed_run(wb, "A", 0.8, 0.0, 1.0) == []
    assert feed_run(wb, "A", 0.95, 1.2, 1.6) == ["A"]
