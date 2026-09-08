-- The seed is generated for one threshold, and the build has to notice a drift.
--
-- `t_critical.csv` holds Student-t quantiles at the tail probability of
-- |Z| > 2.5. Change `anomaly_z_threshold` without regenerating it and the mart
-- silently judges days against critical values for a tail nobody asked for:
-- every column still populates, every range test still passes, and the flag
-- means something no document describes.
--
-- ML-09's sweep is unaffected -- it re-flags in Python against scipy directly --
-- so this guards the warehouse rather than the experiment.
select distinct z_threshold
from {{ ref('t_critical') }}
where z_threshold <> {{ var('anomaly_z_threshold', 2.5) }}
