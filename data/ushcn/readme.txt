U.S. HISTORICAL CLIMATOLOGY NETWORK MONTHLY (USHCN) Version 2.5.0
(Last Updated: 06 August, 2014)

1. INTRODUCTION

    1.1 OVERVIEW

        In October 2012, a revision to the USHCN version 2 datasets was 
	released as version 2.5. The version 2.5 processing steps are essentially 
	the same as in version 2.0, but the increase in version number change reflects 
	modifications that have been made to the underlying database as well as 
	coding changes to the 	pairwise homogenization algorithm (PHA) that improve 
	its overall efficiency. These modifications are listed in more detail in 
	Table 1 of the USHCN version 2 website: 
	
	   http://www.ncdc.noaa.gov/oa/climate/research/ushcn
	
	and in the techical documents linked to therein.  
		
	The composition of the network remains unchanged at 1218 stations, but the 
	data formats and reprocessing frequency have changed. USHCN version 2.5 is now 
	produced using the same processing system used for GHCN-Monthly version 3.
	This reprocessing consists of a construction process that assembles the USHCN 
	version 2.5 monthly data in a specific source priority order (one that favors 
	monthly data calculated directly from the latest version of GHCN-Daily), quality 
	controls the data, identifies inhomogeneities and performs adjustments where 
	possible. 


    1.2 INTERNET ACCESS

        WEBPAGE: http://www.ncdc.noaa.gov/oa/climate/research/ushcn

        FTP: ftp://ftp.ncdc.noaa.gov/pub/data/ushcn/v2.5

    1.3 DOWNLOADING AND INSTALLING

        WINDOWS (example):

	USHCN v2.5 files are compressed into gzip format by element. 
	
	Data files are all of the format ushcn.element.latest.dataset.tar.gz where
	element=tmax, tmin, tavg, or prcp; and, 
	dataset=raw, tob, or FLs.52i. 
	
	The data will untar/uncompress into a directory called ushcn.version.date where 
	version=2.5.0 and date=the year, month and day (yyyymmdd) that the data were last 
	reprocessed and updated. 
	
	Element definitions are as follows: 
	
	  tmax = monthly mean maximum temperature
	  tmin = monthly mean minimum temperature
	  tavg = average monthly temperature (tmax+tmin)/2
 	  prcp = total monthly precipitation
	
	See section 2.2 for a definition of the dataset types
	
	For more information on gzip, please see the following web site:

        www.gzip.org

        and for potential software that can compress/decompress gzip files, please see:

        www.gzip.org/#faq4

        LINUX (example):

        wget ftp://ftp.ncdc.noaa.gov/pub/data/ushcn/v2.5/ushcn.tmax.latest.FLs.52i.tar.gz
        tar -zxvf ushcn.tmax.latest.FLs.52i.tar.gz

        (alternatively, if "tar" does not support decompression, a user can try:

        wget ftp://ftp.ncdc.noaa.gov/pub/data/ushcn/v2.5/ushcn.tmax.latest.FLs.52i.tar.gz
        gzip -d ushcn.tmax.latest.FLs.52i.tar.gz
        tar -xvf ushcn.tmax.latest.FLs.52i.tar.gz)

        Note: the data are placed in their own separate directory, that is named
              according to the following specification:

              ushcn.v2.x.y.YYYYMMDD where 

              x = integer to be incremented with major modifications to the processing system
	          and/or database 
              y = integer to be incremented with minor modifications to the processing system
	          and/or database
              YYYY = year specific dataset was processed and produced
              MM   = month specific dataset was processed and produced
              DD   = day specific dataset was processed and produced
 
              Three files (per element) should be present in the directory along with the station
	      list (ushcn-v2.5-stations.txt).

2. STATION LIST AND DATA FORMATS

	FORMAT OF "ushcn-v2.5-stations.txt"

	----------------------------------------
	Variable             Columns    Type
	----------------------------------------
	COUNTRY CODE             1-2    Character
	NETWORK CODE               3    Character
	ID PLACEHOLDERS ("00")   4-5    Character
	COOP ID                 6-11    Character
	LATITUDE               13-20    Real
	LONGITUDE              22-30    Real
	ELEVATION              33-37    Real
	STATE                  39-40    Character
	NAME                   42-71    Character
	COMPONENT 1 (COOP ID)  73-78    Character
	COMPONENT 2 (COOP ID)  80-85    Character
	COMPONENT 3 (COOP ID)  87-92    Character
	UTC OFFSET             94-95    Integer
	-----------------------------------------
	These variables have the following definitions:
	
	COOP ID     is the six-digit U.S. Cooperative Observer Network station 
	            identification code.  
		    Note that the first two digits in the Coop Id correspond 
		    to the state. 
	
	LATITUDE    is latitude of the station (in decimal degrees).
	
	LONGITUDE   is the longitude of the station (in decimal degrees).
	
	ELEVATION   is the elevation of the station (in meters, missing = -999.9).
	
	STATE       is the U.S. postal code for the state.
	
	NAME        is the name of the station location.
	
	COMPONENT 1 is the Coop Id for the first station (in chronologic order) whose 
            records were joined with those of the HCN site to form a longer time
	    series.  "------" indicates "not applicable".
	
	COMPONENT 2 is the Coop Id for the second station (if applicable) whose records 
            were joined with those of the HCN site to form a longer time series.
	    
	COMPONENT 3 is the Coop Id for the third station (if applicable) whose records 
            were joined with those of the HCN site to form a longer time series.
	    
	UTC OFFSET  is the time difference between Coordinated Universal Time (UTC) and 
            local standard time at the station (i.e., the number of hours that 
	    must be added to local standard time to match UTC).  


    2.2  DATA 

         Monthly data in the USHCN v2.5 data use a "3 flag" format
         similar to that used within the Global Historical Climatology 
         Network-Daily (GHCN-Daily) and GHCN-Monthly (version 3) datasets.

    2.2.1 DATA FORMAT

          Variable          Columns      Type
          --------          -------      ----

          ID                 1-11        Integer
          YEAR              13-16        Integer
          VALUE1            17-22        Integer
          DMFLAG1           23-23        Character
          QCFLAG1           24-24        Character
          DSFLAG1           25-25        Character
            .                 .             .
            .                 .             .
            .                 .             .
          VALUE12          116-121       Integer
          DMFLAG12         122-122       Character
          QCFLAG12         123-123       Character
          DSFLAG12         124-124       Character

          Variable Definitions:

          ID: 11 character alfanumeric identifier:
	      characters 1-2=Country Code ('US' for all USHCN stations)
	      character 3=Network code ('H' for Historical Climatology Network)
              digits 4-11='00'+6-digit Cooperative Observer Identification Number 

          YEAR: 4 digit year of the station record.
 
          VALUE: monthly value (MISSING=-9999).  Temperature values are in
                 hundredths of a degree Celsius, but are expressed as whole
                 integers (e.g. divide by 100.0 to get whole degrees Celsius).
		 Precipitation values are in tenths of millimeters, but are
		 also expressed as whole integers (e.g. divide by 10.0 to 
		 get millimeters).  

          DMFLAG: data measurement flag, nine possible values:

                  blank = no measurement information applicable
                  a-i = number of days missing in calculation of monthly mean
                        temperature 
		  E = The value is estimated using values from surrounding
		      stations because a monthly value could not be computed 
		      from daily data; or,
                      the pairwise homogenization algorithm removed the value 
		      because of too many apparent inhomogeneities occuring 
		      close together in time.
		      .
 		
          QCFLAG: quality control flag, seven possibilities within
                  quality controlled unadjusted (qcu) dataset, and 2 
                  possibilities within the quality controlled adjusted (qca) 
                  dataset.

                  Quality Controlled Unadjusted (QCU) QC Flags:
         
                  BLANK = no failure of quality control check or could not be
                          evaluated.

                  D = monthly value is part of an annual series of values that
                      are exactly the same (e.g. duplicated) within another
                      year in the station's record.

                  I = checks for internal consistency between TMAX and TMIN. 
                      Flag is set when TMIN > TMAX for a given month. 

                  L = monthly value is isolated in time within the station
                      record, and this is defined by having no immediate non-
                      missing values 18 months on either side of the value.

                  M = Manually flagged as erroneous.

                  O = monthly value that is >= 5 bi-weight standard deviations
                      from the bi-weight mean.  Bi-weight statistics are
                      calculated from a series of all non-missing values in 
                      the station's record for that particular month.

                  S = monthly value has failed spatial consistency check.
                      Any value found to be between 2.5 and 5.0 bi-weight
                      standard deviations from the bi-weight mean, is more
                      closely scrutinized by exmaining the 5 closest neighbors
                      (not to exceed 500.0 km) and determine their associated
                      distribution of respective z-scores.  At least one of 
                      the neighbor stations must have a z score with the same
                      sign as the target and its z-score must be greater than
                      or equal to the z-score listed in column B (below),
                      where column B is expressed as a function of the target
                      z-score ranges (column A). 

                                  ---------------------------- 
                                       A       |        B
                                  ----------------------------
                                    4.0 - 5.0  |       1.9
                                  ----------------------------
                                    3.0 - 4.0  |       1.8
                                  ----------------------------
                                   2.75 - 3.0  |       1.7
                                  ----------------------------
                                   2.50 - 2.75 |       1.6
                                     


                  W = monthly value is duplicated from the previous month,
                      based upon regional and spatial criteria and is only 
                      applied from the year 2000 to the present.                   

                  Quality Controlled Adjusted (QCA) QC Flags:

                  A = alternative method of adjustment used.
 
                  M = values with a non-blank quality control flag in the "qcu"
                      dataset are set to missing the adjusted dataset and given
                      an "M" quality control flag.


          DSFLAG: data source flag for monthly value:

                  Blank = Value was computed from daily data available in GHCN-Daily
		  
	          Not Blank = Daily data are not available so the monthly value was 
		              obtained from the USHCN version 1 dataset.  The possible 
			      Version 1 DSFLAGS are as follows:
	       
                              1 =  NCDC Tape Deck 3220, Summary of the Month Element Digital File
	                      2 =  Means Book - Smithsonian Institute, C.A. Schott (1876, 1881 thru 1931)
                              3 =  Manuscript - Original Records, National Climatic Data Center 
                              4 =  Climatological Data (CD), monthly NCDC publication 
                              5 =  Climate Record Book, as described in History of Climatological Record
		                   Books, U.S. Department of Commerce, Weather Bureau, USGPO (1960)
                              6 =  Bulletin W - Summary of the Climatological Data for the United States (by
                                   section), F.H. Bigelow, U.S. Weather Bureau (1912); and, Bulletin W -
                                   Summary of the Climatological Data for the United States, 2nd Ed.
                              7 =  Local Climatological Data (LCD), monthly NCDC publication
                              8 =  State Climatologists, various sources
                              B =  Professor Raymond Bradley - Refer to Climatic Fluctuations of the Western
                                   United States During the Period of Instrumental Records, Bradley, et. al.,
                                   Contribution No. 42, Dept. of Geography and Geology, University of
                                   Massachusetts (1982)
                              D =  Dr. Henry Diaz, a compilation of data from Bulletin W, LCD, and NCDC Tape
                                   Deck 3220 (1983)
                              G =  Professor John Griffiths - primarily from Climatological Data
		  

3. HOW TO CITE

	Users are requested to cite the version number (version 2.0 for version 2
	or version 2.5.x.YYYYMMDD for version 2.5.0 and above) as well as 
	
	Menne, M.J., C.N. Williams, and R.S. Vose, 2009:  The United States Historical 
	Climatology Network monthly temperature data?Version 2.  Bulletin of the American 
	Meteorological Society, 90, 993-1007.


4. CONTACT

    4.1 QUESTIONS AND FEEDBACK

        NCDC.USHCN@noaa.gov

    4.2 OBTAINING ARCHIVED VERSIONS

        At this time, the National Climatic Data Center does not maintain an
        online ftp archive of daily processed USHCN versions.  The latest
        version always overwrites the previous version and thus represents the
        latest data, quality control, etc.  
