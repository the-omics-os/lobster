def test_jinja2_importable():
    """Jinja2 must be available as a core dependency."""
    import jinja2

    assert hasattr(jinja2, "Environment")
