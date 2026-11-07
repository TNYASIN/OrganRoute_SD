# San Diego organ-transport flight dataset: data dictionary

**Source:** U.S. Bureau of Transportation Statistics, Reporting Carrier On-Time Performance (transtats.bts.gov).
**Built:** 2026-11-07 with build_san_dataset.py
**Date range:** 2025-01-01 to 2025-10-31
**Rows:** 332,917 total; 32,612 arriving at SAN; plus arrivals into hubs LAS, SFO, PHX, DEN, SJC
**Duplicates removed:** 0 | **Rows with missing delay but not cancelled/diverted (data_issue=1):** 0

## Files
| File | What it is |
|---|---|
| flights_into_SAN.csv | One row per scheduled flight arriving at SAN (and at the hub airports) |
| san_routes.csv | One row per airline + origin + destination: volume, typical timetable, delay and cancellation rates |
| organ_windows.csv | Organ viability windows (hours) and whether each usually travels on commercial flights |
| san_hospitals.csv | San Diego transplant centers and organs; drive times from SAN to be filled from Google Maps |

## flights_into_SAN.csv columns
| Column | Meaning |
|---|---|
| flight_date | Scheduled flight date (YYYY-MM-DD, origin local date) |
| month, day_of_week | Derived from flight_date |
| carrier | Reporting airline code (WN=Southwest, AA=American, DL=Delta, UA=United, AS=Alaska, B6=JetBlue, NK=Spirit, F9=Frontier, HA=Hawaiian, OO=SkyWest...) |
| flight_number | Airline flight number |
| origin, dest | Airport codes |
| crs_dep_time / crs_arr_time | Scheduled departure / arrival, **local time** at each airport, 4-digit text "HHMM" |
| dep_hour / arr_hour | Hour of scheduled departure / arrival (0-23) |
| crs_elapsed_min | Scheduled gate-to-gate minutes |
| distance_mi | Route distance in miles |
| arr_delay_min | Minutes late at arrival (negative = early). Blank if cancelled or diverted |
| cancelled / diverted | 1 = yes |
| cancellation_code | A = airline, B = weather, C = National Air System, D = security |
| late_15 / late_60 / late_120 / late_240 | 1 if arrived more than N minutes late OR was cancelled/diverted |
| *_delay_min (5 columns) | Minutes of delay by cause; only reported when arrival delay is 15+ min |
| data_issue | 1 = delay missing without a cancellation/diversion (exclude from analysis) |

## Busiest routes into SAN in this data
| Airline | From | Flights/day | % late >60 min or cancelled | % cancelled |
|---|---|---|---|---|
| WN | SMF | 10.42 | 6.9% | 0.5% |
| WN | SJC | 10.37 | 5.9% | 1.0% |
| WN | LAS | 10.13 | 9.6% | 0.9% |
| AA | DFW | 8.8 | 16.7% | 3.5% |
| WN | PHX | 8.77 | 8.4% | 0.8% |
| WN | OAK | 8.37 | 7.0% | 0.9% |
| AS | SEA | 7.59 | 6.9% | 2.1% |
| UA | SFO | 6.9 | 5.3% | 0.4% |
| WN | DEN | 6.55 | 9.9% | 0.7% |
| OO | LAX | 6.48 | 3.9% | 0.0% |

## Rows per input file
| File | US flights | Into SAN |
|---|---|---|
| On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_1.zip | 539,747 | 7,189 |
| On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_10.zip | 605,844 | 8,689 |
| On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_4.zip | 583,950 | 7,959 |
| On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_7.zip | 631,428 | 8,775 |

## Limitations
- Commercial flights only; charter flights (how hearts, lungs and most livers travel) are not in BTS.
- Direct legs into SAN only, unless hub arrivals were added. A connecting organ's first-leg delay is not captured otherwise.
- BTS measures aircraft delays, not organ-handling failures (organs not loaded, left on board, unclaimed at cargo).
- Carriers below the BTS reporting threshold are not included; some regional flights report under their partner airline.
- Times are local to each airport.
- Drive times in san_hospitals.csv are the upper end of Google Maps’ typical range from SAN, checked 2026-11-07 for 2 am and 5 pm weekday departures
