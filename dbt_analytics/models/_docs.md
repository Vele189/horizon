{# Shared column descriptions. Defined once and referenced with doc(), because
   city_id appears in nine models and a description that drifts between them is
   worse than none — a reviewer cannot tell which one to believe. #}

{% docs col_city_id %}
Slug from `config/cities.yml`, and the join key to every dimension and fact.
Not a foreign key in bronze — that layer must land even if the city config
changes — but constrained by a `relationships` test from gold onwards.
{% enddocs %}

{% docs col_observation_time %}
The instant the observation covers, as `timestamptz` in UTC. ING-01 requests
`timezone=UTC` and asserts the returned offset is zero on every response, so
this is UTC by construction rather than by conversion.
{% enddocs %}

{% docs col_date_key %}
The UTC calendar day, and the foreign key to `dim_date`. A projection of a day
that is already a UTC day upstream, not a timezone conversion.
{% enddocs %}

{% docs col_ingested_at %}
When this pipeline landed the row. One timestamp per extraction run, so rows
from one run tie rather than ordering by how long the `COPY` took to reach
them — which is what makes silver's deduplication deterministic.
{% enddocs %}

{% docs col_batch_id %}
Groups every row written by one extraction run, so a bad run can be deleted
wholesale with a single `delete ... where batch_id = ...`.
{% enddocs %}

{% docs col_api_latitude %}
Latitude of the ERA5 grid cell that actually answered, which is not the
coordinate that was asked for: London's 51.5074 resolves to 51.4938. Recorded
per row so provenance survives an edit to `cities.yml`.
{% enddocs %}

{% docs col_api_longitude %}
Longitude of the grid cell that answered. See `api_latitude`.
{% enddocs %}

{% docs col_api_elevation_m %}
Elevation of the grid cell that answered, in metres — not the city's own
elevation, which `dim_cities` holds. London's 11 m resolves to a 16 m cell.
{% enddocs %}

{% docs col_temperature_2m_mean %}
Daily mean 2 m air temperature, °C. The primary measure: the climatology,
Z-scores and anomaly flags are all built on it.
{% enddocs %}

{% docs col_temperature_2m_max %}
Daily maximum 2 m air temperature, °C.
{% enddocs %}

{% docs col_temperature_2m_min %}
Daily minimum 2 m air temperature, °C.
{% enddocs %}

{% docs col_apparent_temperature %}
Apparent ("feels like") temperature, °C — air temperature adjusted for
humidity, wind and radiation. Diverges most from the dry-bulb reading in the
humid tropics, where it runs several degrees higher.
{% enddocs %}

{% docs col_dew_point %}
Dew point, °C. Cannot exceed the air temperature except by the source's 0.1 °C
rounding, which a singular test allows for and nothing more.
{% enddocs %}

{% docs col_relative_humidity %}
Relative humidity, per cent. Bounded 0–100 by an `accepted_range` test.
{% enddocs %}

{% docs col_surface_pressure %}
Pressure at the grid cell's own elevation, hPa — **not** reduced to sea level.
Johannesburg at 1753 m reads 822 hPa while its sea-level pressure is 998, which
is why this column carries a looser lower bound than `pressure_msl`.
{% enddocs %}

{% docs col_pressure_msl %}
Pressure reduced to mean sea level, hPa. The comparable-across-cities measure,
and what pressure tendency is computed on.
{% enddocs %}

{% docs col_wind_speed %}
Wind speed at 10 m, km/h. Stored in km/h as ING-01 requests it; bounded in m/s
through the `kmh_to_ms` macro, because physical limits for wind are quoted in
m/s and a 0–120 bound read as km/h would fail on an ordinary winter storm.
{% enddocs %}

{% docs col_wind_gusts %}
Maximum 10 m wind gust, km/h. See `wind_speed_10m` for the unit note.
{% enddocs %}

{% docs col_wind_direction %}
Dominant 10 m wind direction, degrees clockwise from north. 0 and 360 are both
present and both mean north.
{% enddocs %}

{% docs col_precipitation %}
Total precipitation, mm — rain plus the water equivalent of snow.
{% enddocs %}

{% docs col_cloud_cover %}
Cloud cover, per cent.
{% enddocs %}

{% docs col_weather_code %}
WMO 4677 present-weather code. A category, not a magnitude: 51 is drizzle and
95 a thunderstorm, and nothing sensible comes of averaging them.
{% enddocs %}

{% docs col_season_model %}
Which season regime the city follows: `four_season`, `wet_dry`, or
`seasonless`. Set in `cities.yml` and read by `dim_city_season`, so a tropical
city is never handed a winter.
{% enddocs %}

{% docs col_hemisphere %}
`north` or `south`, derived from the sign of the latitude and never configured.
The season mapping reads it, so for the five southern cities a wrong value
inverts summer and winter rather than mislabelling them.
{% enddocs %}
