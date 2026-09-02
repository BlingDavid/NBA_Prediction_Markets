"""Smoke tests for the local dashboard (webapp/app.py)."""
import webapp.app as appmod
from webapp.app import app


def test_index_route_returns_200_with_title():
    resp = app.test_client().get("/")
    assert resp.status_code == 200
    assert b"Today's Picks" in resp.data


def test_load_bet_card_sorts_by_edge_desc(tmp_path, monkeypatch):
    csv_path = tmp_path / "bet_card.csv"
    csv_path.write_text(
        "home_team,away_team,market,side,selection,model_prob,implied_prob,edge,bet_size\n"
        "OKC,LAL,total,over,O 209.7,0.68,0.52,0.16,500\n"
        "MIA,BOS,moneyline,home,MIA,0.55,0.52,0.03,120\n"
    )
    monkeypatch.setattr(appmod, "BET_CARD", csv_path)
    rows, updated = appmod.load_bet_card()
    assert [r["edge_pct"] for r in rows] == ["+16.0%", "+3.0%"]  # sorted big-edge first
    assert rows[0]["pick"] == "O 209.7"
    assert rows[0]["cls"] == "big" and rows[1]["cls"] == "mid"
    assert updated is not None


def test_load_bet_card_handles_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(appmod, "BET_CARD", tmp_path / "nope.csv")
    rows, updated = appmod.load_bet_card()
    assert rows == [] and updated is None
