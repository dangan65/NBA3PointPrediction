# Daily Player-Game CSV Schema

This project should use one canonical historical dataset:

- `data/player_game_3pt_features.csv`

Each row is one completed player-game from the 2025-26 season. The row must contain only information that was knowable before tipoff for feature columns, plus the final result in `player_made_3pt` as the training target.

## Row Grain

- One row per `player_id + game_id`
- Season scope: 2025-26 only
- Update cadence: once daily after the final NBA game of the day is complete and stats pages have updated
- Upsert key: `game_id, player_id`

## Required Columns

These align directly with the trainer in `/Users/danielgannon/ML3ptProject/train_3pt_model.py`.

### Identifiers and context

- `season` (`2025-26`)
- `as_of_date` (date the row was generated)
- `game_date`
- `game_id`
- `player_id`
- `player_name`
- `team`
- `opponent`

### Target

- `player_made_3pt`

### Core rolling shooter features

- `last_5_games_3pm_avg`
- `last_5_games_3pa_avg`
- `last_3_games_3pm`
- `season_avg_3pm`
- `rolling_3_game_trend_slope`
- `team_3pa_share`
- `minutes_per_game`
- `usage_rate`
- `rest_days`

### Opponent team defense

- `opp_3pa_allowed`
- `opp_3pm_allowed`
- `opp_def_rating`
- `opp_pace`

### Game context

- `home_game`
- `spread`
- `game_total`

### Optional model features already supported

- `back_to_back`
- `injury_usage_bump`
- `position`
- `opp_position_3pt_rank`

## Recommended Additional Columns

These are not used by the current trainer yet, but they are worth storing now because they improve future feature engineering and make daily refreshes easier.

### Player shooting quality

- `season_3pa`
- `season_3pm`
- `season_3pt_pct`
- `catch_shoot_3pa`
- `catch_shoot_3pm`
- `pull_up_3pa`
- `pull_up_3pm`
- `touch_time`
- `dribbles_before_shot`
- `wide_open_3pa`
- `wide_open_3pt_pct`
- `open_3pa`
- `open_3pt_pct`
- `tight_3pa`
- `tight_3pt_pct`
- `very_tight_3pa`
- `very_tight_3pt_pct`

### Opponent/defender proxies

- `opp_team_wide_open_3pa_allowed`
- `opp_team_above_break_3pa_allowed`
- `opp_team_corner_3pa_allowed`
- `expected_primary_defender`
- `primary_defender_id`
- `primary_defender_pos`
- `primary_defender_3pt_pct_allowed`
- `primary_defender_def_fg_pct_diff`

### Data quality / pipeline control

- `stats_finalized_at`
- `source_boxscore`
- `source_tracking`
- `row_version`

## Canonical CSV Order

Use this order to keep the file stable:

1. `season`
2. `as_of_date`
3. `game_date`
4. `game_id`
5. `player_id`
6. `player_name`
7. `team`
8. `opponent`
9. `home_game`
10. `position`
11. `rest_days`
12. `back_to_back`
13. `minutes_per_game`
14. `usage_rate`
15. `team_3pa_share`
16. `last_5_games_3pm_avg`
17. `last_5_games_3pa_avg`
18. `last_3_games_3pm`
19. `season_avg_3pm`
20. `rolling_3_game_trend_slope`
21. `injury_usage_bump`
22. `opp_3pa_allowed`
23. `opp_3pm_allowed`
24. `opp_def_rating`
25. `opp_pace`
26. `opp_position_3pt_rank`
27. `spread`
28. `game_total`
29. `season_3pa`
30. `season_3pm`
31. `season_3pt_pct`
32. `catch_shoot_3pa`
33. `catch_shoot_3pm`
34. `pull_up_3pa`
35. `pull_up_3pm`
36. `touch_time`
37. `dribbles_before_shot`
38. `wide_open_3pa`
39. `wide_open_3pt_pct`
40. `open_3pa`
41. `open_3pt_pct`
42. `tight_3pa`
43. `tight_3pt_pct`
44. `very_tight_3pa`
45. `very_tight_3pt_pct`
46. `opp_team_wide_open_3pa_allowed`
47. `opp_team_above_break_3pa_allowed`
48. `opp_team_corner_3pa_allowed`
49. `expected_primary_defender`
50. `primary_defender_id`
51. `primary_defender_pos`
52. `primary_defender_3pt_pct_allowed`
53. `primary_defender_def_fg_pct_diff`
54. `player_made_3pt`
55. `stats_finalized_at`
56. `source_boxscore`
57. `source_tracking`
58. `row_version`

## Daily Refresh Rule

Run the refresh job once each morning after all games from the prior NBA day are final.

### Historical training file

- Pull all completed games from the previous day
- For each player who appeared, compute features using only games strictly before `game_date`
- Append or update rows in `data/player_game_3pt_features.csv`
- If a row already exists for the same `game_id, player_id`, replace it

### Upcoming scoring file

Also generate:

- `data/upcoming_props_features.csv`

This file should contain one row per player prop for today's games. It uses the same feature columns except:

- omit `player_made_3pt`
- include `decimal_odds` or `american_odds`

For upcoming rows, features should use the latest completed games through the prior day only.

## Source Mapping

- Box score and game logs: NBA API player game logs / box score endpoints
- Team defense and pace: NBA API team dashboard endpoints
- Tracking splits (catch-and-shoot, closest defender): NBA API player tracking endpoints
- Odds (`spread`, `game_total`, prop odds): sportsbook or odds API feed

## Implementation Note

Do not compute rolling features from the same row's final result. For example, `last_5_games_3pm_avg` for a game on February 10 must be based only on games before February 10.
