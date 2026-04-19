"""
Tests for the ENABLE_ODDS_API gating flag.

Context: the live-capture loop calls fetch_odds_api_lines every 15s, and
when the 500-req/mo free quota is exhausted each call prints an HTTP 401
warning. Gate the whole function behind ENABLE_ODDS_API (default off) so
captures stay clean without ripping out the #8 divergence signal code.
"""

from unittest.mock import patch, MagicMock


class TestEnableOddsApiFlag:
    def test_returns_empty_when_flag_is_false(self, monkeypatch):
        """When ENABLE_ODDS_API=False, the function must short-circuit
        without any HTTP call — even if a valid-looking key is set."""
        import live_data

        monkeypatch.setattr(live_data, "ENABLE_ODDS_API", False)
        monkeypatch.setattr(live_data, "ODDS_API_KEY", "real-looking-key-abc123")

        with patch("live_data.requests.get") as mock_get:
            result = live_data.fetch_odds_api_lines()

        assert result == []
        mock_get.assert_not_called()

    def test_calls_api_when_flag_is_true(self, monkeypatch):
        """Sanity check: flag=True + valid key → HTTP call is made."""
        import live_data

        monkeypatch.setattr(live_data, "ENABLE_ODDS_API", True)
        monkeypatch.setattr(live_data, "ODDS_API_KEY", "real-looking-key-abc123")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        with patch("live_data.requests.get", return_value=mock_resp) as mock_get:
            live_data.fetch_odds_api_lines()

        mock_get.assert_called_once()

    def test_returns_empty_when_key_missing_even_if_flag_true(self, monkeypatch):
        """Missing/placeholder key still short-circuits regardless of flag."""
        import live_data

        monkeypatch.setattr(live_data, "ENABLE_ODDS_API", True)
        monkeypatch.setattr(live_data, "ODDS_API_KEY", "")

        with patch("live_data.requests.get") as mock_get:
            result = live_data.fetch_odds_api_lines()

        assert result == []
        mock_get.assert_not_called()
