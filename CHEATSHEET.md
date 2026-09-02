# 🏀 NBA Model — Morning Routine

## Every game day: 3 steps

**1. Get the picks.** Open the **Terminal** app, paste this line, hit Enter:
```
~/Documents/Claude/Projects/NBA_Live_Model/run.sh pregame
```
Wait for it to finish (a minute or two).

**2. Open the bet card.** In Finder, open:
```
Documents ▸ Claude ▸ Projects ▸ NBA_Live_Model ▸ outputs ▸ bet_card.csv
```
(double-click → opens in Excel/Numbers)

**3. Read it like a line.** Each row = one flagged edge:

| Column | What it means |
|---|---|
| `home_team` / `away_team` | the matchup |
| `market` / `side` | market type (moneyline/spread/total) + which side |
| `selection` | **what you'd actually bet** (e.g. `O 209.7`, or a team) |
| `implied_prob` | what the market is pricing (implied %) |
| `model_prob` | the model's fair % |
| `edge` | model % − implied % → **the gap you'd be betting** |
| `bet_size` | suggested $ (fractional Kelly) |

**Decision rule:** only act when `edge` is clearly bigger than **~3–4 points** (fees + vig eat small edges), and it passes your own smell test.

---

## Before you click "buy" — every time
- ✅ **Injury/news check** — is a star out? Is the line stale for a real reason?
- ✅ **Bigger edge = better** — Kalshi fee is ~1–2¢/contract near 50¢; a 2-point edge is mostly fees.
- ✅ **Paper first** — log the pick + the price, don't fund real money until you've seen it beat closing lines over weeks.

## Do / Don't
- ✅ **DO** use the **pre-game** picks (`./run.sh pregame`) as a screen / second opinion.
- ❌ **DON'T** trade the **live in-game** scripts — tested, they lose to fees. Not proven.
- ❌ **DON'T** treat the bet card as gospel — it finds *candidate* edges, not sure things.

## If something looks off
- "Connected?" check: `~/Documents/Claude/Projects/NBA_Live_Model/run.sh scan`
- Deep-dive one game: `run.sh ev <TICKER>` (ticker is in the bet card)
- Broken / weird output → grab a technical person; it's usually a data or API-key issue, not a signal.

*Screen for edges. Verify with your own eyes. Size for the fees. Paper-trade before real money.*
