# Test-set player minutes

The shared lookup supports xG, xT, VAEP and Atomic-VAEP, whose current test
match IDs are identical: 231 of the 771 women's matches (30%, seed 42).
A separate xA lookup covers its current 154 test matches (20%, seed 42).
Both lookups use the same appearance-time calculation, but their totals must
not be mixed. No notebook predictions, training settings or player filters are changed.

## Generate

From the project root, using the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m minutes.player_minutes
```

This default command regenerates only the shared lookup. To generate only xA:

```powershell
.\.venv\Scripts\python.exe -m minutes.player_minutes --split xa
```

The script reads the selected current split through `load_xg_test_and_rest()`
or `load_xa_test_and_rest()` and the local StatsBomb event and lineup JSON files.
Existing caches for the selected pipeline are required; the script does not
train models or download data. Generating either lookup leaves the other untouched.

Outputs are stored directly in `minutes/`, alongside this script. Input caches
remain in `vaep_data/`; no xA subdirectory is created.

- `women_test_player_minutes.csv`: shared per-player totals, with `player_id`,
  `player_name`, `test_games_played`, `test_milliseconds` and `test_minutes`.
- `women_test_player_match_minutes.csv`: shared per-player, per-match details.
- `women_test_minutes_metadata.json`: shared test match IDs and time conventions.
- `xa_women_test_player_minutes.csv`: per-player totals for the xA test matches.
- `xa_women_test_player_match_minutes.csv`: xA per-player, per-match details.
- `xa_women_test_minutes_metadata.json`: xA test match IDs and time conventions.

The xA files use the same schemas as the shared files. Join totals on `player_id`,
not player name. The xA lookup follows the current 20% code configuration, not
the historical 77-match (10%) outputs still saved in the xA notebook.
Restart its kernel and rerun the notebook before pairing its predictions with these minutes.

## Read from a notebook

The loader checks the exact test match IDs, preventing accidental reuse across
splits. The shared lookup remains the default:

```python
from minutes.player_minutes import load_test_player_minutes

player_minutes = load_test_player_minutes(bundle.women_test_games)
```

In xT, pass `WOMEN_TEST_GAMES` instead of `bundle.women_test_games`.
In xA, explicitly select its separate lookup:

```python
player_minutes = load_test_player_minutes(bundle.women_test_games, split="xa")
```

For direct inspection, the totals are also ordinary UTF-8 CSVs.

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
If a notebook's test split changes, verify its exact match IDs again and regenerate
the corresponding lookup before using it.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s minutes -v
```
