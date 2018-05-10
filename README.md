# peakload
Notebooks, data, and code for neural network prediction of coincidental peaks in ERCOT power demand.

Data:

ERCOT historical power generation

-Last 7 days worth of hourly loads
---Last 4 months of loads on the same day of week and hour of day (e.g. all Mondays at 1 PM)
-These loads are binned into a normalized histogram (7 bins)
-Variance of these loads
-Mean of these loads
-Max of these loads
-Min of these loads
---Last 2 months of loads on the same hour of the day (e.g. all 2 PM loads)
-These loads are binned into a normalized histogram (7 bins)
-Variance of these loads
-Mean of these loads
-Max of these loads
-Min of these loads
-Season
-Week number of year
-Month
-Weekday
-Hour of day
-Maximum load observed so far through the billing cycle
-Average load observed so far through the billing cycle
-Variance of loads observed so far through the billing cycle

Weather data from the following cities:

"Witchita Falls": (33.9137, -98.4934), 
"Mineral Wells": (32.8085, -98.1128), 
"Paris": (33.6609, -95.5555), 
"Dallas": (32.7767, -96.7970), 
"Fort Worth": (32.7555, -97.3308), 
"Tyler": (32.3513, -95.3011),  
"Waco": (31.5493, -97.1467), 
"San Angelo": (31.4638, -100.4370), 
"Abilene": (32.4487, -99.7331), 
"Midland": (31.9973, -102.0779), 
"Wink": (31.7512, -103.1599), 
"Junction": (30.4894, -99.7720), 
"Austin": (30.2672, -97.7431), 
"Houston": (29.7604, -95.3698), 
"Galveston": (29.3013, -94.7977), 
"Victoria": (28.8053, -97.0036), 
"Corpus Christi": (27.8006, -97.39640), 
"Laredo": (27.5306, -99.4803),
"San Antonio": (29.4241, -98.4936)

\item Engineered Features
\begin{itemize}
\item Maximum temperature observed thus far through the billing schedule
\item Apparent maximum temperature observed thus far
\item Minimum temperature observed thus far
\item Apparent minimum temperature thus far
\item Precipitation accumulation thus far
\item Average high over the last two weeks
\item Average apparent high over the last two weeks
\item Average low over the last two weeks
\item Average apparent low over the last two weeks
\item Average humidity over last two weeks
\item Average cloud cover over last two weeks
\end{itemize
