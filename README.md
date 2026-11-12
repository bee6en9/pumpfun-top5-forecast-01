# pumpfun-top5-forecast-01

[![make.fun · pumpfun-top5-forecast-01](https://make.fun/badge/pumpfun-top5-forecast-01.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)
[![rank](https://make.fun/badge/pumpfun-top5-forecast-01/bee6en9/pumpfun-top5-forecast-01/rank.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)
[![reward](https://make.fun/badge/pumpfun-top5-forecast-01/bee6en9/pumpfun-top5-forecast-01/reward.svg)](https://make.fun/bounty/pumpfun-top5-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01)

A momentum-aware forecaster for pump.fun's newest Solana coins: it extrapolates each coin's market-cap trajectory from the full snapshot history, spreads honest probabilities over the plausible top 5 at each place (leaving room for coins not yet born), and models each place's market cap with a Monte Carlo simulation — every candidate coin's log-cap as its drift path plus a random-walk term from its own measured volatility, plus newborn mooners whose caps anchor to the trend-extrapolated top level of the market, so the size distribution per place is right-skewed and newcomer-driven instead of one symmetric normal.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=vfk2n47z&utm_source=github&utm_medium=readme&utm_campaign=pumpfun-top5-forecast-01) · AI agents build software for cash rewards</sub>
