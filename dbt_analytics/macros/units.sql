{#
    Unit conversions, as macros rather than inline SQL.

    ING-01 asks the API for metric units explicitly and asserts them on every
    response, so almost nothing needs converting here — silver's job is to
    assert, not to convert. These exist for the two places where the source's
    own units are internally inconsistent or where an assertion is naturally
    expressed in a different unit from the stored one.

    Keeping them as macros means the factor appears once. A `/ 3.6` typed into
    six schema entries is six chances to type `* 3.6` instead, and the result
    would be a range test that passes on data it should reject.
#}

{% macro cm_to_mm(column) -%}
    {#
        Open-Meteo reports `snowfall_sum` in centimetres while every other
        depth in the same row — precipitation_sum, rain_sum — is millimetres.
        That is the source's inconsistency, not ours, and it is the single most
        likely place for a downstream model to add two numbers that are not in
        the same unit. Silver publishes the millimetre form alongside so no
        model has to remember.
    #}
    ({{ column }} * 10.0)
{%- endmacro %}


{% macro kmh_to_ms(column) -%}
    {#
        Wind is stored as km/h, which is what ING-01 requests and what the
        bronze columns are documented as. Physical bounds for wind are
        conventionally quoted in m/s, so the range assertions convert rather
        than restating the limit in a second unit where it could drift from the
        first.
    #}
    ({{ column }} / 3.6)
{%- endmacro %}
