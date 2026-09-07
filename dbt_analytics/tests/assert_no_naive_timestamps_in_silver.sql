-- Every timestamp in silver must carry a zone.
--
-- A `timestamp without time zone` is a wall-clock reading with no instant
-- attached: it compares and sorts against other naive values without
-- complaint, and means something different for each of fifteen cities. One
-- column slipping to naive is how a "UTC" pipeline stops being one, silently.
--
-- Checked against the catalogue rather than a column list, so a model added
-- tomorrow is covered without anyone remembering to add it here.
select
    table_schema,
    table_name,
    column_name,
    data_type
from information_schema.columns
where table_schema = 'silver_staging'
  and data_type in ('timestamp without time zone', 'time without time zone')
