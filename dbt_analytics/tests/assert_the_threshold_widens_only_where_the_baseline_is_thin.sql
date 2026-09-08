-- The correction must be a correction, not a change of definition.
--
-- DBT-14 replaced a fixed |Z| > 2.5 with a t-based bar that depends on how many
-- observations sigma was estimated from. The danger in that is not the thin
-- cities it was written for, which are few rows and loudly wrong; it is the
-- complete ones, which are every row that matters and would move silently. A
-- sign error, a degree of freedom counted the wrong way, or a critical value
-- read from the wrong column would still produce a plausible threshold near
-- 2.5 and shift a few hundred flags nobody would notice.
--
-- So the bound is asserted from both sides. A complete baseline of ~459
-- observations must land within one per cent of the nominal threshold, and no
-- baseline of any size may ever land *below* it: the correction only ever
-- widens, because a sigma estimated from a finite sample is never more
-- trustworthy than one known exactly.
with judged as (

    select
        city_id,
        date_key,
        baseline_observations,
        baseline_degrees_of_freedom,
        anomaly_z_critical
    from {{ ref('fact_weather_anomalies') }}
    where is_anomaly is not null

)

select *
from judged
where anomaly_z_critical < {{ var('anomaly_z_threshold', 2.5) }}
   or (baseline_observations >= 400
       and anomaly_z_critical > {{ var('anomaly_z_threshold', 2.5) }} * 1.01)
