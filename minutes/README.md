# Shared test-set player minutes

This lookup supports the current xG, xT, VAEP and Atomic-VAEP notebooks, whose
test match IDs are identical: 231 of the 771 women's matches. xA is excluded.
No notebook predictions, training settings or player filters are changed.

## Generate

From the project root, using the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m minutes.player_minutes
```

The script reads the current xG split through `load_xg_test_and_rest()` and the
local StatsBomb event and lineup JSON files. Existing xG caches are required;
the script does not train models or download data.

Outputs are stored in `minutes/`, alongside this script. Existing xG input
caches remain in `vaep_data/`.

- `women_test_player_minutes.csv`: one row per appearing player, with
  `player_id`, `player_name`, `test_games_played`, `test_milliseconds` and
  `test_minutes`. Join on `player_id`, not player name.
- `women_test_player_match_minutes.csv`: one row per player and test match,
  including team, starter status and that appearance's duration.
- `women_test_minutes_metadata.json`: exact test match IDs, split settings,
  time conventions and recorded durations for every match.

## Read from a notebook

The loader checks the exact test match IDs, preventing accidental reuse with
xA or a changed split:

```python
from minutes.player_minutes import load_test_player_minutes

player_minutes = load_test_player_minutes(bundle.women_test_games)
```

In xT, pass `WOMEN_TEST_GAMES` instead of `bundle.women_test_games`.
For direct inspection, the totals are also an ordinary UTF-8 CSV:

```python
player_minutes = pd.read_csv("minutes/women_test_player_minutes.csv")
```

## Time convention

- Use Starting XI and substitution events to open appearances; close them at
  substitution, dismissal, permanent Player Off, or the recorded final whistle.
- Include stoppage time and extra time. Exclude breaks between periods and
  penalty shootouts. Use period-relative timestamps rather than subtracting
  cumulative event `minute` values; first-half stoppage must not be lost.
- Temporary treatment absences do not end an appearance and are not deducted.
  This is appearance duration, not ball-in-play time or strictly on-pitch time.
- Use lineup files for player identity and appearance coverage checks, not for
  summing position intervals. Some source position records overlap or wrongly
  reintroduce dismissed players after a tactical change.
- Count every appearance in the selected matches, even without shots or other
  metric-specific actions. Exclude unused substitutes; keep goalkeepers.
- Sum integer milliseconds before converting to minutes. A recorded match
  shorter than 90 minutes is not padded. Match `3881507`, for example, has about
  56.654 recorded minutes including stoppage; its duration is retained as-is.

The CSVs contain denominators only. No per-90 values or rankings are calculated.
If a notebook's test split changes, verify the shared-match assumption again and
regenerate the lookup before using it.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s minutes -v
```
