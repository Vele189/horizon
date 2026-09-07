-- Roughly 0.5-3% of scored city-days.
--
-- A normal distribution puts 1.24% past |Z| > 2.5. Real temperature residuals
-- have fatter tails, so somewhat more is expected; an order of magnitude more
-- would mean sigma is being underestimated, and near zero would mean it is
-- being estimated from the very days it is scoring.
--
-- Denominator is scored days only. Including unscorable ones would dilute the
-- rate with days that could never have been flagged.
with rate as (

    select
        count(*) filter (where is_anomaly)::numeric
            / nullif(count(*) filter (where z_temperature_2m_mean is not null), 0)
            as anomaly_rate
    from {{ ref('fact_weather_anomalies') }}

)

select anomaly_rate from rate
where anomaly_rate is null or anomaly_rate < 0.005 or anomaly_rate > 0.03
