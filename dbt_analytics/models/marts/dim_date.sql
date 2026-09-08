-- The date spine, covering every day any observation could fall on.
--
-- Generated rather than derived from the facts: a spine built from what has
-- landed would have a hole wherever ingestion does, and a join against it
-- would hide the hole rather than reveal it. The range is the backfill's own,
-- from the configured start to the archive edge, with a year of headroom at
-- the end so a spine rebuilt less often than the data still covers it.
--
-- Seasons are NOT here as a single column, because a season is not a property
-- of a date: December is summer in Sydney and winter in London. Both
-- four-season answers are carried side by side, and dim_city_season resolves
-- the right one per city including the two tropical regimes.
{#
    The end defaults to today rather than to a pinned date: the archive edge
    moves, and a spine that stopped at a hardcoded day would silently start
    dropping the newest observations from every join. `dim_date_end` overrides
    it for a reproducible build.
#}
{%- set spine_end = var('dim_date_end', '') -%}

with spine as (

    select generate_series(
        date '{{ var("dim_date_start", "1995-01-01") }}',
        {% if spine_end %}date '{{ spine_end }}'{% else %}current_date{% endif %}
            + interval '1 year',
        interval '1 day'
    )::date as date_day

),

attributes as (

    select
        date_day,
        extract(year    from date_day)::int  as year,
        extract(quarter from date_day)::int  as quarter,
        extract(month   from date_day)::int  as month,
        extract(day     from date_day)::int  as day_of_month,
        to_char(date_day, 'Month')           as month_name,
        to_char(date_day, 'Day')             as day_name,
        extract(isodow  from date_day)::int  as iso_day_of_week,
        extract(week    from date_day)::int  as iso_week,
        extract(isoyear from date_day)::int  as iso_year,
        extract(doy     from date_day)::int  as day_of_year,
        extract(isodow  from date_day)::int in (6, 7) as is_weekend,

        -- The leap-year trap, made explicit rather than left to a join.
        --
        -- `day_of_year` is 1..366, so 1 March is day 60 in a common year and
        -- day 61 in a leap year. Grouping a thirty-year climatology by
        -- day_of_year therefore mixes 1 March with 29 February and shifts
        -- every day after February by one in three years out of four: a
        -- systematic error that looks like a seasonal signal.
        --
        -- month_day is the join key that does not have that problem: it is
        -- stable across every year, and 29 February simply has a quarter of
        -- the sample size, which is true and worth knowing rather than hidden.
        to_char(date_day, 'MM-DD')           as month_day,
        (extract(month from date_day) = 2
         and extract(day from date_day) = 29) as is_leap_day,
        (extract(year from date_day)::int % 4 = 0
         and (extract(year from date_day)::int % 100 <> 0
              or extract(year from date_day)::int % 400 = 0)) as is_leap_year,

        -- The climatology axis: day-of-year as it would be in a leap year, so
        -- every month_day has exactly one number and 29 February gets its own
        -- (60) rather than sharing. This is what a circular +/-7 day window is
        -- measured on. Raw day_of_year cannot serve, because 31 December is
        -- 365 in a common year and 366 in a leap one, so a window around it
        -- would draw on different days depending on the year.
        {{ climatology_day_of('date_day') }} as climatology_day,

        -- Day-of-year as if every year were common, for anything that needs a
        -- contiguous numeric axis rather than a key. 29 February shares day 59
        -- with 28 February; nothing after it shifts.
        case
            when extract(doy from date_day)::int <= 59
                then extract(doy from date_day)::int
            when (extract(month from date_day) = 2
                  and extract(day from date_day) = 29) then 59
            when (extract(year from date_day)::int % 4 = 0
                  and (extract(year from date_day)::int % 100 <> 0
                       or extract(year from date_day)::int % 400 = 0))
                then extract(doy from date_day)::int - 1
            else extract(doy from date_day)::int
        end as day_of_year_common,

        -- Both answers, side by side. Neither is "the" season.
        {{ meteorological_season('extract(month from date_day)::int', "'north'") }}
            as season_northern,
        {{ meteorological_season('extract(month from date_day)::int', "'south'") }}
            as season_southern

    from spine

)

select * from attributes
