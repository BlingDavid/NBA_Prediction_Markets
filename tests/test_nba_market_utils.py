"""
Tests for nba_market_utils.normalize_nba_abbrev.

This function is the join-key bridge between ESPN game-state data and
Kalshi market data. A missed variant silently drops a whole game from
the labeled training set (discovered via 2026-04-17 GSW-PHX, which
shipped as game_key '2026-04-17_GS_PHX' in game_states but
'2026-04-17_GSW_PHX' everywhere else, so the join never fired).

Lock down the full set of known variants here so regressions are caught
before they corrupt a capture window.
"""

import pytest

from nba_market_utils import normalize_nba_abbrev


class TestNormalizeNbaAbbrev:
    @pytest.mark.parametrize("variant,expected", [
        # Canonical codes pass through unchanged.
        ("GSW", "GSW"),
        ("NYK", "NYK"),
        ("NOP", "NOP"),
        ("SAS", "SAS"),
        ("UTA", "UTA"),
        ("WAS", "WAS"),
        ("PHX", "PHX"),
        ("BKN", "BKN"),
        # ESPN-style short forms → Kalshi canonical.
        ("GS", "GSW"),
        ("NY", "NYK"),
        ("NO", "NOP"),
        ("SA", "SAS"),
        ("WSH", "WAS"),      # ESPN uses WSH for Washington.
        ("UTAH", "UTA"),     # ESPN sometimes uses UTAH.
        ("PHO", "PHX"),      # Some feeds use PHO.
        ("BRK", "BKN"),      # Legacy/alt Brooklyn code.
    ])
    def test_known_variants(self, variant, expected):
        assert normalize_nba_abbrev(variant) == expected

    def test_case_insensitive(self):
        assert normalize_nba_abbrev("gs") == "GSW"
        assert normalize_nba_abbrev("wsh") == "WAS"

    def test_passthrough_for_unknown(self):
        assert normalize_nba_abbrev("XYZ") == "XYZ"

    def test_none_and_empty(self):
        assert normalize_nba_abbrev(None) is None
        assert normalize_nba_abbrev("") == ""
