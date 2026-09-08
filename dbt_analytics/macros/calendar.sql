{#
    Calendar expressions shared between the date dimension and anything that
    needs them before the dimension exists.

    `int_climatology_contributions` sits in the intermediate layer and must not
    read a mart, so it cannot join dim_date for these. Defining them once here
    is what keeps the two derivations from drifting, and a drift would be
    quiet, because both would still produce a number between 1 and 366.
#}

{% macro climatology_day_of(date_column) -%}
    {#
        Day-of-year as it would be in a leap year, 1..366, so every calendar
        day has exactly one number and 29 February gets its own rather than
        sharing. Raw day_of_year cannot serve: 31 December is 365 in a common
        year and 366 in a leap one, so a window around it would draw on
        different days depending on the year.
    #}
    extract(doy from make_date(2024,
        extract(month from {{ date_column }})::int,
        extract(day   from {{ date_column }})::int))::int
{%- endmacro %}


{% macro utc_date_of(timestamp_column) -%}
    {#
        The UTC calendar day of a timestamptz. A projection of a day that is
        already a UTC day upstream, not a timezone conversion.
    #}
    ({{ timestamp_column }} at time zone 'UTC')::date
{%- endmacro %}
