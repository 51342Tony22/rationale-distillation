from fyp_distill.summarize import bootstrap_difference


def test_paired_bootstrap_direction():
    better = {"a": True, "b": True, "c": False, "d": True}
    worse = {"a": False, "b": True, "c": False, "d": False}
    result = bootstrap_difference(better, worse, samples=1000, seed=7)
    assert result["difference"] == 0.5
    assert result["n"] == 4

