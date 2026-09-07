-- Which season a given month is, for a given city.
--
-- The join that dim_date cannot make on its own: season depends on the city's
-- hemisphere and on its season regime, and one row of dim_date has a different
-- answer for Sydney than for London. Fifteen cities by twelve months is 180
-- rows, so this is a lookup rather than a bridge over the whole spine.
--
-- Tropical cities are not forced into four seasons. Lagos gets wet and dry;
-- Singapore gets `year_round`, which says there is no thermal cycle rather
-- than inventing one.
with months as (

    select generate_series(1, 12) as month

),

city_months as (

    select
        cities.city_id,
        cities.hemisphere,
        cities.season_model,
        months.month
    from {{ ref('dim_cities') }} as cities
    cross join months

)

select
    city_id,
    month,
    hemisphere,
    season_model,
    {{ season_for('month', 'hemisphere', 'season_model') }} as season
from city_months
