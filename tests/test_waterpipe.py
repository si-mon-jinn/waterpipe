"""Tests for waterpipe."""


def test_imports():
    """Test that main modules can be imported."""
    from waterpipe import config
    from waterpipe import generation
    from waterpipe import detection
    from waterpipe import stats
    from waterpipe.attacks import ATTACKS
    from waterpipe.metrics import METRICS
    
    assert len(ATTACKS) > 0
    assert len(METRICS) > 0


def test_attack_registry():
    """Test that all attacks can be instantiated."""
    from waterpipe.attacks import get_attack, ATTACKS
    
    for name in ATTACKS:
        attack = get_attack(name)
        assert attack is not None


def test_metric_registry():
    """Test that all metrics can be instantiated."""
    from waterpipe.metrics import get_metric, METRICS
    
    for name in METRICS:
        # Skip bertscore as it has heavy dependencies
        if name == "bertscore":
            continue
        metric = get_metric(name)
        assert metric is not None
