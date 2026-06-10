from app.core.services.integrity import compute_integrity_summary, build_integrity_report


def test_summary_empty():
    summary = compute_integrity_summary({})
    assert summary.duplicates_count == 0
    assert summary.leakage.train_val == 0
    assert summary.leakage.train_test == 0
    assert summary.leakage.val_test == 0


def test_summary_clean_dataset():
    hashes = {
        "train/cat/a.jpg": "h1",
        "train/dog/b.jpg": "h2",
        "val/cat/c.jpg": "h3",
        "test/cat/d.jpg": "h4",
    }
    summary = compute_integrity_summary(hashes)
    assert summary.duplicates_count == 0
    assert summary.leakage.train_test == 0


def test_summary_duplicates_inside_split():
    hashes = {
        "train/cat/a.jpg": "h1",
        "train/cat/a_copy.jpg": "h1",
        "train/dog/b.jpg": "h1",
    }
    summary = compute_integrity_summary(hashes)
    # три файла с одним хешем в train -> две лишние копии
    assert summary.duplicates_count == 2


def test_summary_train_test_leakage():
    hashes = {
        "train/cat/a.jpg": "h1",
        "test/cat/b.jpg": "h1",
        "val/dog/c.jpg": "h2",
    }
    summary = compute_integrity_summary(hashes)
    assert summary.leakage.train_test == 1
    assert summary.leakage.train_val == 0
    assert summary.leakage.val_test == 0


def test_report_details():
    hashes = {
        "train/cat/a.jpg": "h1",
        "train/cat/a_copy.jpg": "h1",
        "test/cat/b.jpg": "h1",
        "val/dog/c.jpg": "h2",
        "test/dog/d.jpg": "h2",
    }
    report = build_integrity_report("ds1", "v1", hashes)

    assert report.dataset_id == "ds1"
    assert report.version_id == "v1"
    assert report.summary.duplicates_count == 1
    assert report.summary.leakage.train_test == 1
    assert report.summary.leakage.val_test == 1

    assert len(report.duplicates) == 1
    dup = report.duplicates[0]
    assert dup.split == "train"
    assert dup.files == ["train/cat/a.jpg", "train/cat/a_copy.jpg"]

    train_test = report.leakage["train_test"]
    assert len(train_test) == 1
    assert train_test[0].files["train"] == ["train/cat/a.jpg", "train/cat/a_copy.jpg"]
    assert train_test[0].files["test"] == ["test/cat/b.jpg"]
