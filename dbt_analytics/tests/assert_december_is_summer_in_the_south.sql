-- The checklist's assertion, and the one that catches a global month lookup.
--
-- A season mapping written as `case month when 12 then 'winter'` passes review
-- and is wrong for a third of this city set. It does not merely mislabel the
-- five southern cities: it puts their hottest month in the same cohort as
-- Moscow's coldest, so any seasonal aggregate averages summer and winter
-- together and reports something close to the annual mean with a season's name
-- on it.
select
    seasons.city_id,
    cities.hemisphere,
    cities.season_model,
    seasons.month,
    seasons.season
from {{ ref('dim_city_season') }} as seasons
join {{ ref('dim_cities') }} as cities
  on cities.city_id = seasons.city_id
where cities.season_model = 'four_season'
  and (
        (seasons.month = 12 and cities.hemisphere = 'north' and seasons.season <> 'winter')
     or (seasons.month = 12 and cities.hemisphere = 'south' and seasons.season <> 'summer')
     or (seasons.month = 7  and cities.hemisphere = 'north' and seasons.season <> 'summer')
     or (seasons.month = 7  and cities.hemisphere = 'south' and seasons.season <> 'winter')
  )
