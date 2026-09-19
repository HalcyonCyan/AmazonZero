"""Value perspective shared by search, matches and model callers."""

def perspective_value(value: float, source_player: int, target_player: int) -> float:
    """Convert a zero-sum value between player perspectives.

    This is the single place where value signs are changed.  The staged action
    model does *not* change player after selecting a piece or destination, so
    blindly negating on every search edge would be incorrect.
    """
    if source_player not in (1, 2) or target_player not in (1, 2):
        raise ValueError("Players must be 1 or 2")
    return float(value if source_player == target_player else -value)


