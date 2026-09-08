{#
    UTC to a city's wall-clock time.

    One way, deliberately. Postgres returns a *naive* timestamp from
    `AT TIME ZONE`, which is correct: a wall-clock reading has no offset, and
    at a daylight-saving fall-back two different instants produce the same
    reading. Converting back is therefore ambiguous. Measured here, ten of
    274,920 hourly rows do not survive a round trip, one per DST-observing city
    per autumn transition.

    So UTC is the stored truth and local time is a presentation of it. Anything
    that needs to be joined, ordered or compared must use the timestamptz;
    anything a human reads may use this.
#}
{% macro to_local_time(column, timezone_column) -%}
    ({{ column }} at time zone {{ timezone_column }})
{%- endmacro %}


{% macro local_date(column, timezone_column) -%}
    {#
        The city's own calendar day, which is what "the hottest day in Sydney"
        means. Not the same as the UTC day for any city more than a few hours
        from Greenwich.
    #}
    (({{ column }} at time zone {{ timezone_column }})::date)
{%- endmacro %}
