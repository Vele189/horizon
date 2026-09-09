-- ============================================================================
-- Teleconnection indices: a vintage table
-- ============================================================================
-- Idempotent by construction: every statement is IF NOT EXISTS or a COMMENT.
-- Applied by
--
--     python ingestion/teleconnections.py --write
--
-- **Grain: one row per index per nominal period per vintage.** Not one row per
-- period. "The ONI for January 2024" is not a number, it is a series of
-- numbers with dates: NOAA restates history when the ENSO base period shifts,
-- which it does every five years, and a model scoring a day in 2024 must read
-- the value as it stood on that day rather than as it stands now.
--
-- Append-on-change, not append-always. A re-run that finds the same number
-- writes nothing; one that finds a different number writes a new row beside
-- the old and never over it. That keeps the table proportional to the number
-- of actual revisions rather than to the number of times the job has run,
-- while still making a revision impossible to lose.
--
-- **Two dates, and neither is the nominal period.**
--
--   publication_date  the earliest date this value could be READ. End of the
--                     last covered month plus the index's stated lag. This is
--                     what any join to a forecast date must use.
--   vintage_at        when THIS pipeline first saw this number. What makes a
--                     revision visible.
--
-- The nominal period is a label, not a timestamp, and joining on it is the
-- leak this whole table exists to prevent. `covers_start` and `covers_end` are
-- stored per row rather than derived at read time, because the derivation for
-- a centred index is not obvious -- the ONI labelled January covers December
-- through February -- and a query that got it wrong would look correct.
--
-- **`publication_is_estimated` is honest about what we cannot know.**
-- Publication dates for periods that predate this pipeline are computed from
-- the registry's lag rule, because NOAA does not distribute historical
-- vintages. From the first ingest forward, the first vintage of a period is an
-- observation of its availability rather than an estimate. The column says
-- which kind of date a row carries, so an analysis can restrict itself to the
-- observed ones without having to know when this table was created.
-- ============================================================================

create schema if not exists bronze_raw;

create table if not exists bronze_raw.teleconnection_indices (
    -- Grain -------------------------------------------------------------
    index_id                text        not null,
    nominal_period          date        not null,
    vintage_at              timestamptz not null,

    -- What the value summarises ------------------------------------------
    covers_start            date        not null,
    covers_end              date        not null,

    -- When it may be read -------------------------------------------------
    publication_date        date        not null,
    publication_is_estimated boolean    not null,

    value                   double precision not null,

    -- Provenance ----------------------------------------------------------
    source_url              text        not null,
    source_created_at       date,
    ingested_at             timestamptz not null default now(),
    batch_id                uuid        not null,

    primary key (index_id, nominal_period, vintage_at)
);

-- A value may never be readable before the last month it summarises has ended.
-- This is the leak, stated as a constraint: an off-by-one in the centred-window
-- arithmetic, or a lag accidentally set negative, produces rows the database
-- refuses rather than a model that scores suspiciously well.
alter table bronze_raw.teleconnection_indices
    drop constraint if exists teleconnection_publication_follows_coverage;

alter table bronze_raw.teleconnection_indices
    add constraint teleconnection_publication_follows_coverage
    check (
        covers_start <= covers_end
        and publication_date > covers_end
    );

-- The nominal period must sit inside the span it labels. Catches a parser that
-- mislabels a centred season by putting the first month of the window in the
-- period column instead of the middle one.
alter table bronze_raw.teleconnection_indices
    drop constraint if exists teleconnection_nominal_is_inside_its_span;

alter table bronze_raw.teleconnection_indices
    add constraint teleconnection_nominal_is_inside_its_span
    check (nominal_period between covers_start and covers_end);

-- The read pattern is "every index, as it stood on date D", which scans by
-- publication date and then by vintage.
create index if not exists teleconnection_publication_idx
    on bronze_raw.teleconnection_indices (publication_date, index_id, vintage_at);

comment on table bronze_raw.teleconnection_indices is
  'One row per index per nominal period per vintage. Large-scale climate indices from NOAA, each carrying the date it became readable and the date this pipeline first saw it. Append-on-change: a revision adds a row beside the old value, never over it.';
comment on column bronze_raw.teleconnection_indices.nominal_period is
  'The LABEL the source gives the value, as the first of the month. For a centred index this is the middle month of the window, not the first. Never join a forecast date to this column.';
comment on column bronze_raw.teleconnection_indices.covers_end is
  'Last month the value summarises. For the ONI this is one month AFTER the nominal period, because the index is centred: the value labelled January covers December through February.';
comment on column bronze_raw.teleconnection_indices.publication_date is
  'Earliest date this value could be read: end of covers_end plus the index lag from config/teleconnections.yml. This is the column a join to a forecast date must use.';
comment on column bronze_raw.teleconnection_indices.vintage_at is
  'When this pipeline first saw this number. A second row for the same period with a later vintage_at is a revision by the source.';
comment on column bronze_raw.teleconnection_indices.publication_is_estimated is
  'True where the publication date comes from the lag rule rather than from having watched the value appear. NOAA distributes no historical vintages, so every period predating this table is estimated.';
