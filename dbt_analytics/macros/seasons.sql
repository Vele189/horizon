{#
    Season resolution. The logic lives here once because it is wrong in three
    different ways if written inline, and each way is quiet.

    A season is not a property of a date. It is a property of a date *and a
    place*, and this project's fifteen cities span three regimes:

    `four_season`
        Meteorological seasons, which are whole months and not the astronomical
        solstice-to-solstice ones: DJF, MAM, JJA, SON in the north. The
        southern set is the same months shifted by six, so December is summer
        in Sydney and winter in London — the same row of dim_date, two answers.
    `wet_dry`
        Lagos. A tropical monsoon climate has no thermal season worth the name;
        what varies is rainfall. Forcing "winter" onto a Lagos December
        describes nothing, and would pull it into a northern-winter cohort in
        any seasonal aggregate.
    `seasonless`
        Singapore. Af, within 1.4° of the equator: no thermal cycle and no dry
        season either. The honest label is that there is no season, and saying
        so is more useful than inventing four.
#}

{% macro meteorological_season(month_column, hemisphere_column) -%}
    {#
        Whole-month meteorological seasons, flipped for the south.

        Written as one expression rather than two branches so the northern and
        southern cases cannot drift: the southern label is the northern one
        six months away, and that is stated once.
    #}
    case
        when {{ hemisphere_column }} = 'north' then
            case
                when {{ month_column }} in (12, 1, 2)  then 'winter'
                when {{ month_column }} in (3, 4, 5)   then 'spring'
                when {{ month_column }} in (6, 7, 8)   then 'summer'
                else 'autumn'
            end
        else
            case
                when {{ month_column }} in (12, 1, 2)  then 'summer'
                when {{ month_column }} in (3, 4, 5)   then 'autumn'
                when {{ month_column }} in (6, 7, 8)   then 'winter'
                else 'spring'
            end
    end
{%- endmacro %}


{% macro wet_dry_season(month_column, hemisphere_column) -%}
    {#
        West African monsoon timing for the northern tropics: rains from April
        through October, dry from November through March. The southern tropics
        run the opposite half of the year.

        Coarse, and labelled as coarse. A per-city rainfall climatology would
        be better and is not available until the marts are built; what matters
        here is that Lagos is not told it has a winter.
    #}
    case
        when {{ hemisphere_column }} = 'north' then
            case when {{ month_column }} between 4 and 10 then 'wet' else 'dry' end
        else
            case when {{ month_column }} between 10 and 12
                   or {{ month_column }} between 1 and 4 then 'wet' else 'dry' end
    end
{%- endmacro %}


{% macro season_for(month_column, hemisphere_column, season_model_column) -%}
    {#
        The regime-aware entry point. Everything downstream should use this
        rather than reaching for the four-season macro directly, because
        reaching directly is exactly how Lagos and Singapore acquire a winter.
    #}
    case {{ season_model_column }}
        when 'four_season' then
            {{ meteorological_season(month_column, hemisphere_column) }}
        when 'wet_dry' then
            {{ wet_dry_season(month_column, hemisphere_column) }}
        when 'seasonless' then 'year_round'
    end
{%- endmacro %}
