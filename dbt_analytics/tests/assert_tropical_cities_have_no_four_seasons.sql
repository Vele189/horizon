-- Lagos and Singapore must never be given a thermal season.
--
-- The failure this guards against is a refactor that reaches for the
-- four-season macro directly instead of season_for(), which would silently
-- hand every city a winter. Nothing with a non-four-season regime may carry a
-- four-season label.
select
    seasons.city_id,
    seasons.season_model,
    seasons.month,
    seasons.season
from {{ ref('dim_city_season') }} as seasons
where seasons.season_model <> 'four_season'
  and seasons.season in ('winter', 'spring', 'summer', 'autumn')
