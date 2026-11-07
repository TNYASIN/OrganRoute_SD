# EDA summary: flights into San Diego (SAN)

Loaded 332,917 flights, 27 columns.

## Cleaning steps
1. Dropped 0 rows with data_issue = 1.
2. Dropped 0 duplicate flights.
3. Dropped 0 rows with impossible flight time, distance or hour.
4. Added sched_arr_local (local arrival date-time); 6,041 flights land the next calendar day.
5. Added flags: is_direct_to_SAN, outcome, dep_time_block, route_low_sample.
   Long delays were kept on purpose: extreme delays are the events that threaten an organ, not noise.
Rows: 332,917 -> 332,917

## Flights into SAN: 32,612 flights, 70 origin airports, 11 airlines
- Late >60 min, cancelled or diverted: 8.3%
- Late >120 min, cancelled or diverted: 3.7%
- Cancelled: 1.03% | Diverted: 0.47%
- Arrival delay (minutes): median -3, 90th pct 44, 95th 76, 99th 172, max 998

**By departure hour:** lowest risk at 9:00 (4.3%), highest at 23:00 (16.0%). Delays build up through the day.
**By month:** Jan 7.6%, Apr 7.2%, Jul 10.6%, Oct 7.6%
**By airline (500+ flights):** B6 5.0% (n=542), DL 6.9% (n=3,189), UA 7.3% (n=3,476), WN 7.9% (n=11,877), AS 8.0% (n=3,669), OO 9.0% (n=4,968), AA 11.1% (n=3,236), F9 13.1% (n=854)
**Riskiest regular routes into SAN:** F9 ORD 35.5%, AS EWR 23.2%, F9 AUS 20.6%, AA DFW 16.7%, OO JAC 16.7%
**Causes of long delays (share of minutes):** Late incoming aircraft 44%, Airline 29%, Air traffic system 18%, Weather 9%, Security 0%
**Cancellation reasons:** weather 190, airline 111, air traffic system 36
**Tail risk:** >30 min: 15.6%, >60 min: 8.3%, >120 min: 3.7%, >180 min: 2.4%, >240 min: 1.9%, >360 min: 1.6%

## Data quality notes
- Times are local to each airport; arrivals after midnight are handled in sched_arr_local.
- route_low_sample = 1 marks routes with <20 days of data; treat their rates as unreliable.
- BTS records aircraft delays only, not organ-handling failures, so these rates are a lower bound on risk.
