# Climate Volatility & Risk Engine

An end-to-end analytics platform that ingests three decades of global weather observations, models them into a governed dimensional warehouse, and forecasts extreme temperature anomalies across fifteen major cities.

> **Status:** in development. This README is a stub; the full write-up — architecture diagram, model metrics, live dashboard link, and five-minute setup — lands on Day 15. The complete proposal and delivery plan is in [`docs/proposal.md`](docs/proposal.md).

## Configuration

No credentials are committed to this repository, and none ever have been. All configuration is supplied through environment variables:

```bash
cp .env.example .env   # then fill in the values
python config.py       # prints the resolved config, secrets masked
```

Every variable is named and documented in [`.env.example`](.env.example), and [`config.py`](config.py) is the only module that reads the environment. `.env` is git-ignored.

## Licence

[MIT](LICENSE)
