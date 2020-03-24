/*
Course: IST 722 Data Warehouse
Assignment: Group 5 - Final Project - Implementing Dimensional Models
Author: Ryan Timbrook
NetID: RTIMBROO
Date: 2/28/2020



Source Dependencies:
	 - DimCategories
	 - DimFeatures
	 - DimAmenitites
	 - DimHostReviewScores
	 - DimLocations
	 - FactListingEstOcc
	 
*/


-- DECLARE CONFIG VARIABLES --

-- SCHEMA CONFIGURATIONS -- UNCOMMENT ONLY ONE SCHEMA
--
:setvar use_schema dev_brook       -- development schema

-- IN SQL Server Manager Studio - Under 'Query' select SQLCMD Mode - to utilize command line variables for configurations
-- DATABASE CONFIGURATIONS -- UNCOMMENT ONLY ONE DATABASE
--                           
--:setvar use_database mar653_airbnb_dw         -- my local development DW database
:setvar stage_db mar653_airbnb_seattle
:setvar dw_db mar653_airbnb_dw


print(N'$(use_schema)')




-- STEP 2: DROP TABLES IF THEY EXIST --
-- DimPropertyTypes
-- DimRoomTypes
-- DimNeighbourhoods
-- DimOccupancyFeatures
-- DimAmenitites
-- FactHostReviewScores
-- DimLocations
-- FactListingEstOcc
---------------------------------------

/* Drop table fudgeinc.FactListingEstOcc */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'$(use_schema).FactListingEstOcc') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE $(use_schema).FactListingEstOcc 
;


/* Drop table fudgeinc.DimListingHostReviewScores */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'$(use_schema).FactHostReviewScores') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE $(use_schema).FactHostReviewScores 
;

/* Drop table fudgeinc.DimSubFeatures */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'$(use_schema).DimSubFeatures') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE $(use_schema).DimSubFeatures 
;

/* Drop table fudgeinc.DimPropertyTypes */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'$(use_schema).DimPropertyTypes') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE $(use_schema).DimPropertyTypes 
;

/* Drop table fudgeinc.DimOccupancyFeatures */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'$(use_schema).DimOccupancyFeatures') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE $(use_schema).DimOccupancyFeatures 
;


/* Drop table fudgeinc.DimLocations */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'$(use_schema).DimLocations') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE $(use_schema).DimLocations 
;

/* Drop table fudgeinc.DimNeighbourhoods */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'$(use_schema).DimNeighbourhoodScores') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE $(use_schema).DimNeighbourhoodScores 
;

/* Drop table fudgeinc.DimListingsHostsMap */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'$(use_schema).DimListingHostMap') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE $(use_schema).DimListingHostMap 
;

/* Drop table fudgeinc.DimPolicies */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'$(use_schema).DimPolicies') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE $(use_schema).DimPolicies 
;

/* Drop table fudgeinc.DimPolicies */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'$(use_schema).DimAmenities') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE $(use_schema).DimAmenities 
;

/* Drop table fudgeinc.DimPriceFees */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'$(use_schema).DimPriceFees') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE $(use_schema).DimPriceFees 
;

/* Drop table fudgeinc.DimReviewScores */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'$(use_schema).DimReviewScores') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE $(use_schema).DimReviewScores 
;

-- STEP 4: CREAT SCHEMAS  --
----------------------------
IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = N'$(use_schema)')
	EXEC('CREATE SCHEMA $(use_schema)');
GO


-- STEP 5: CREAT TABLES  --
---------------------------
-- DimPriceFees
-- DimPropertyTypes
-- DimSubFeatures
-- DimPolicies
-- DimNeighbourhoods
-- DimOccupancyFeatures
-- DimAmenitites
-- DimListingAmenties
-- DimListingsHostsMap
-- FactListingHostReviewScores
-- DimLocations
-- FactListingEstOcc
---------------------------

/* Create table fudgeinc.DimPriceFees */
CREATE TABLE $(use_schema).DimPriceFees (
   [PriceFeesKey]  int IDENTITY  NOT NULL
,  [ListingID]  int   NOT NULL
,  [Price]  float   NOT NULL
,  [WeeklyPrice]  float  NOT NULL
,  [MonthlyPrice]  float NOT NULL
,  [SecurityDeposit]  float NOT NULL
,  [CleaningFee]  float NOT NULL
,  [GuestsIncluded]  float NOT NULL
,  [ExtraPeopleFee]  float NOT NULL
,  [HasWeeklyPrice]  nchar(1) NOT NULL
,  [HasMonthlyPrice]  nchar(1) NOT NULL
,  [HasSecurityDeposit]  nchar(1) NOT NULL
,  [HasCleaningFee]  nchar(1) NOT NULL
,  [HasExtraPeopleFee]  nchar(1) NOT NULL
,  [CalendarAvailableDays]  smallint NOT NULL
,  [RowIsCurrent]  bit   DEFAULT 1 NOT NULL
,  [RowStartDate]  datetime  DEFAULT '12/31/1899' NOT NULL
,  [RowEndDate]  datetime  DEFAULT '12/31/9999' NOT NULL
,  [RowChangeReason]  nvarchar(200)   NULL
, CONSTRAINT [PK_$(use_schema).DimPriceFees] PRIMARY KEY CLUSTERED 
( [PriceFeesKey] )
) ON [PRIMARY]
;

SET IDENTITY_INSERT $(use_schema).DimPriceFees ON
;
INSERT INTO $(use_schema).DimPriceFees (PriceFeesKey, ListingID, Price, WeeklyPrice, MonthlyPrice, SecurityDeposit, CleaningFee, GuestsIncluded, ExtraPeopleFee,HasWeeklyPrice,HasMonthlyPrice,HasSecurityDeposit,HasCleaningFee,HasExtraPeopleFee,CalendarAvailableDays, RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, -1, -1, -1, -1, -1, -1, -1, -1, 'N', 'N', 'N', 'N','N',-1, -1, '12/31/1899', '12/31/9999', 'N/A')
;
SET IDENTITY_INSERT $(use_schema).DimPriceFees OFF
;



/* Create table fudgeinc.DimAmenities */
CREATE TABLE $(use_schema).DimAmenities (
   [AmenitiesKey]  int IDENTITY  NOT NULL
,  [ListingID]  int   NOT NULL
,  [AmenityName]  varchar(50)   NOT NULL
,  [RowIsCurrent]  bit   DEFAULT 1 NOT NULL
,  [RowStartDate]  datetime  DEFAULT '12/31/1899' NOT NULL
,  [RowEndDate]  datetime  DEFAULT '12/31/9999' NOT NULL
,  [RowChangeReason]  nvarchar(200)   NULL
, CONSTRAINT [PK_$(use_schema).DimAmenities] PRIMARY KEY CLUSTERED 
( [AmenitiesKey] )
) ON [PRIMARY]
;

SET IDENTITY_INSERT $(use_schema).DimAmenities ON
;
INSERT INTO $(use_schema).DimAmenities (AmenitiesKey, ListingID, AmenityName, RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, -1, 'None',-1, '12/31/1899', '12/31/9999', 'N/A')
;
SET IDENTITY_INSERT $(use_schema).DimAmenities OFF
;

/* Create table fudgeinc.DimPolicies */
CREATE TABLE $(use_schema).DimPolicies (
   [ListingPolicyKey]  int IDENTITY  NOT NULL
,  [ListingID]  int   NOT NULL
,  [Minimum_Nights]  smallint   NOT NULL
,  [Maximum_Nights] int NOT NULL
,  [Cancellation_Policy] varchar(50) NOT NULL
,  [Instant_Bookable] nchar(1) NOT NULL
,  [RowIsCurrent]  bit   DEFAULT 1 NOT NULL
,  [RowStartDate]  datetime  DEFAULT '12/31/1899' NOT NULL
,  [RowEndDate]  datetime  DEFAULT '12/31/9999' NOT NULL
,  [RowChangeReason]  nvarchar(200)   NULL
, CONSTRAINT [PK_$(use_schema).DimPolicies] PRIMARY KEY CLUSTERED 
( [ListingPolicyKey] )
) ON [PRIMARY]
;

SET IDENTITY_INSERT $(use_schema).DimPolicies ON
;
INSERT INTO $(use_schema).DimPolicies (ListingPolicyKey, ListingID, Minimum_Nights, Maximum_Nights, Cancellation_Policy, Instant_Bookable, RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, -1, -1,-1,'None','f',-1, '12/31/1899', '12/31/9999', 'N/A')
;
SET IDENTITY_INSERT $(use_schema).DimPolicies OFF
;

/* Create table fudgeinc.DimListingHostMap */
CREATE TABLE $(use_schema).DimListingHostMap (
   [ListingHostMapKey]  int IDENTITY  NOT NULL
,  [ListingID]  int   NOT NULL
,  [HostID] int NOT NULL
,	 [HostName] varchar(100) NOT NULL
,  [ListingName]  varchar(100)   NOT NULL
,  [RowIsCurrent]  bit   DEFAULT 1 NOT NULL
,  [RowStartDate]  datetime  DEFAULT '12/31/1899' NOT NULL
,  [RowEndDate]  datetime  DEFAULT '12/31/9999' NOT NULL
,  [RowChangeReason]  nvarchar(200)   NULL
, CONSTRAINT [PK_$(use_schema).DimListingHostMap] PRIMARY KEY CLUSTERED 
( [ListingHostMapKey] )
) ON [PRIMARY]
;

SET IDENTITY_INSERT $(use_schema).DimListingHostMap ON
;
INSERT INTO $(use_schema).DimListingHostMap (ListingHostMapKey, ListingID, HostID,HostName ,ListingName, RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, -1, -1,'None' ,'None',-1, '12/31/1899', '12/31/9999', 'N/A')
;
SET IDENTITY_INSERT $(use_schema).DimListingHostMap OFF
;

/* Create table fudgeinc.DimPropertyTypes */
CREATE TABLE $(use_schema).DimPropertyTypes (
   [PropertyTypeKey]  int IDENTITY  NOT NULL
,  [ListingID]  int   NOT NULL
,  [PropertyTypeName]  varchar(100)   NOT NULL
,  [RowIsCurrent]  bit   DEFAULT 1 NOT NULL
,  [RowStartDate]  datetime  DEFAULT '12/31/1899' NOT NULL
,  [RowEndDate]  datetime  DEFAULT '12/31/9999' NOT NULL
,  [RowChangeReason]  nvarchar(200)   NULL
, CONSTRAINT [PK_$(use_schema).DimPropertyTypes] PRIMARY KEY CLUSTERED 
( [PropertyTypeKey] )
) ON [PRIMARY]
;

SET IDENTITY_INSERT $(use_schema).DimPropertyTypes ON
;
INSERT INTO $(use_schema).DimPropertyTypes (PropertyTypeKey, ListingID, PropertyTypeName, RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, -1, 'None', -1, '12/31/1899', '12/31/9999', 'N/A')
;
SET IDENTITY_INSERT $(use_schema).DimPropertyTypes OFF
;


/* Create table fudgeinc.DimSubFeatures */
CREATE TABLE $(use_schema).DimSubFeatures (
   [SubFeatureKey]  int IDENTITY  NOT NULL
,  [ListingID]  int   NOT NULL
,  [SubFeatureName]  varchar(100)   NOT NULL
,  [RowIsCurrent]  bit   DEFAULT 1 NOT NULL
,  [RowStartDate]  datetime  DEFAULT '12/31/1899' NOT NULL
,  [RowEndDate]  datetime  DEFAULT '12/31/9999' NOT NULL
,  [RowChangeReason]  nvarchar(200)   NULL
, CONSTRAINT [PK_$(use_schema).DimSubFeatures] PRIMARY KEY CLUSTERED 
( [SubFeatureKey] )
) ON [PRIMARY]
;

SET IDENTITY_INSERT $(use_schema).DimSubFeatures ON
;
INSERT INTO $(use_schema).DimSubFeatures (SubFeatureKey, ListingID, SubFeatureName, RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, -1, 'None', -1, '12/31/1899', '12/31/9999', 'N/A')
;
SET IDENTITY_INSERT $(use_schema).DimSubFeatures OFF
;

/* Create table fudgeinc.DimNeighbourhoodScores */
CREATE TABLE $(use_schema).DimNeighbourhoodScores (
   [NeighbourhoodKey]  int IDENTITY  NOT NULL
--,  [NeighbourhoodID]  int   IDENTITY NOT NULL
,  [NeighbourhoodName]  varchar(100)   NOT NULL
,  [Rank] smallint NOT NULL
,	 [Walk_Score] smallint NOT NULL
,	 [Transit_score] smallint NOT NULL
,	 [Bike_score] smallint NOT NULL
,	 [Population] int NOT NULL
,  [RowIsCurrent]  bit   DEFAULT 1 NOT NULL
,  [RowStartDate]  datetime  DEFAULT '12/31/1899' NOT NULL
,  [RowEndDate]  datetime  DEFAULT '12/31/9999' NOT NULL
,  [RowChangeReason]  nvarchar(200)   NULL
, CONSTRAINT [PK_$(use_schema).DimNeighbourhoodScores] PRIMARY KEY CLUSTERED 
( [NeighbourhoodKey] )
) ON [PRIMARY]
;

SET IDENTITY_INSERT $(use_schema).DimNeighbourhoodScores ON
;
INSERT INTO $(use_schema).DimNeighbourhoodScores (NeighbourhoodKey, NeighbourhoodName, Rank, Walk_Score, Transit_score, Bike_score, Population, RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, 'None', -1,-1,-1,-1,-1, -1, '12/31/1899', '12/31/9999', 'N/A')
;
SET IDENTITY_INSERT $(use_schema).DimNeighbourhoodScores OFF
;


/* Create table fudgeinc.DimFeatures --  from occupany_features
 [ListingID]
	  ,[Accommodates]
	  ,[Bathrooms]
	  ,[Bedrooms]
	  ,[Beds]
	  ,[Bedroom_share] --calculated value bedrooms/accommades
	  ,[Bathroom_share] --calculated value bathrooms/accommodates

*/
CREATE TABLE $(use_schema).DimOccupancyFeatures (
   [OccupancyFeatureKey]  int IDENTITY  NOT NULL
,  [ListingID]  int   NOT NULL
,  [Accommodates]  smallint NOT NULL
,  [Bathrooms]  float NOT NULL
,  [Bedrooms]  float NOT NULL
,  [Beds]  float NOT NULL
,  [Bedroom_share]  float NOT NULL
,  [Bathroom_share]  float NOT NULL
,  [RowIsCurrent]  bit   DEFAULT 1 NOT NULL
,  [RowStartDate]  datetime  DEFAULT '12/31/1899' NOT NULL
,  [RowEndDate]  datetime  DEFAULT '12/31/9999' NOT NULL
,  [RowChangeReason]  nvarchar(200)   NULL
, CONSTRAINT [PK_$(use_schema).DimOccupancyFeatures] PRIMARY KEY CLUSTERED 
( [OccupancyFeatureKey] )
) ON [PRIMARY]
;

SET IDENTITY_INSERT $(use_schema).DimOccupancyFeatures ON
;
INSERT INTO $(use_schema).DimOccupancyFeatures (OccupancyFeatureKey, ListingID, Accommodates,Bathrooms,Bedrooms,Beds,Bedroom_share,Bathroom_share, RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, -1,-1,-1,-1,-1,-1,-1,-1, '12/31/1899', '12/31/9999', 'N/A')
;
SET IDENTITY_INSERT $(use_schema).DimOccupancyFeatures OFF
;


/* Create table fudgeinc.DimLocations */
CREATE TABLE $(use_schema).DimLocations (
   [LocationKey]  int IDENTITY  NOT NULL
,  [ListingID]  int   NOT NULL
,  [Neighbourhood]  varchar(100)   NOT NULL
,  [Neighbourhood_Group]  varchar(100)   NOT NULL
,	 [ZipCode] varchar(50) NOT NULL
,	 [Latitude] float NOT NULL
,  [Longitude] float NOT NULL
,  [RowIsCurrent]  bit   DEFAULT 1 NOT NULL
,  [RowStartDate]  datetime  DEFAULT '12/31/1899' NOT NULL
,  [RowEndDate]  datetime  DEFAULT '12/31/9999' NOT NULL
,  [RowChangeReason]  nvarchar(200)   NULL
, CONSTRAINT [PK_$(use_schema).DimLocations] PRIMARY KEY CLUSTERED 
( [LocationKey] )
) ON [PRIMARY]
;

SET IDENTITY_INSERT $(use_schema).DimLocations ON
;
INSERT INTO $(use_schema).DimLocations (LocationKey, ListingID, Neighbourhood, Neighbourhood_Group, ZipCode, Latitude, Longitude, RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, -1, 'None', 'None','None',-1, -1,1, '12/31/1899', '12/31/9999', 'N/A')
;
SET IDENTITY_INSERT $(use_schema).DimLocations OFF
;


/* Create table fudgeinc.FactListingEstOcc */

CREATE TABLE $(use_schema).FactListingEstOcc (
   --[ListingID] int NOT NULL
  [LocationKey]  int   NOT NULL
,  [OccupancyFeatureKey]  int   NOT NULL
,  [SubFeatureKey]  int   NOT NULL
,  [PropertyTypeKey]  int   NOT NULL
,  [ListingHostMapKey]  int   NOT NULL
,  [ListingPolicyKey]  int   NOT NULL
,  [AmenitiesKey]  int   NOT NULL
,  [PriceFeesKey]  int   NOT NULL

--# Calculated Facts --
,  [EstLifetimeOccupancyDays]  int   NOT NULL -- as days
,	 [EstLifetimeOccDailyRate]  float   NOT NULL
,  [EstLifetimeOccYearlyRate]  float   NOT NULL
,  [EstLifetimeIncome] float   NOT NULL
,  [EstLifetimeYearlyIncome] float NOT NULL
,  [EstLifetimePotentialIncome]  float   NOT NULL
,  [EstLifetimePotentialYearlyIncome]  float   NOT NULL
,  [EstLifetimePercentOfPotentialIncome]  float   NOT NULL
,  [LifetimeEarner] smallint NOT NULL
, CONSTRAINT [PK_$(use_schema).FactListingEstOcc] PRIMARY KEY NONCLUSTERED 
( [LocationKey], [OccupancyFeatureKey], [SubFeatureKey], [PropertyTypeKey], [ListingHostMapKey], [ListingPolicyKey], [AmenitiesKey], [PriceFeesKey] )
) ON [PRIMARY]
;


/* Create table fudgeinc.FactHostReviewScores */

CREATE TABLE $(use_schema).FactHostReviewScores (
    --[ListingID] int NOT NULL
[LocationKey]  int   NOT NULL
,  [OccupancyFeatureKey]  int   NOT NULL
,  [SubFeatureKey]  int   NOT NULL
,  [PropertyTypeKey]  int   NOT NULL
,  [ListingHostMapKey]  int   NOT NULL
,  [ListingPolicyKey]  int   NOT NULL
,  [AmenitiesKey]  int   NOT NULL
,  [PriceFeesKey]  int   NOT NULL
--# Review Score Facts
--,  [FirstReviewDateKey]  int   NOT NULL
--,	 [LastReviewDateKey]  int   NOT NULL
,  [NumberOfReviews]  int   NOT NULL 
,  [ReviewsPerMonth] float NOT NULL
,  [MinimumNights] int NOT NULL
,	 [ReviewScoresRating]  float   NOT NULL
,  [ReviewScoresAccuracy]  float   NOT NULL
,  [ReviewScoresCleanliness] float   NOT NULL
,  [ReviewScoresCheckin] float NOT NULL
,  [ReviewScoresCommunication]  float   NOT NULL
,  [ReviewScoresLocation]  float   NOT NULL
,  [ReviewScoresValue]  float   NOT NULL
,  [AvgPolarityPositiveScore] float NOT NULL
,  [AvgPolarityNegativeScore] float NOT NULL
,  [AvgPolarityCompoundScore] float NOT NULL
,  [ReviewDaysRange] float NOT NULL
,  [ReviewYearsRange] float NOT NULL
,  [ReviewsPerYearRate] float NOT NULL
,  [ReviewsFrequency] float NOT NULL
, CONSTRAINT [PK_$(use_schema).FactHostReviewScores] PRIMARY KEY CLUSTERED 
( [LocationKey], [OccupancyFeatureKey], [SubFeatureKey], [PropertyTypeKey], [ListingHostMapKey], [ListingPolicyKey], [AmenitiesKey], [PriceFeesKey] )
) ON [PRIMARY]
;


-- STEP 6: ADD TABLE CONSTRAINTS  --
------------------------------------
--- FactListingEstOcc
--- 
ALTER TABLE $(use_schema).FactListingEstOcc ADD CONSTRAINT
   FK_$(use_schema)_FactListingEstOcc_LocationKey FOREIGN KEY
   (
   LocationKey
   ) REFERENCES $(use_schema).DimLocations
   ( LocationKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
ALTER TABLE $(use_schema).FactListingEstOcc ADD CONSTRAINT
   FK_$(use_schema)_FactListingEstOcc_OccupancyFeatureKey FOREIGN KEY
   (
   OccupancyFeatureKey
   ) REFERENCES $(use_schema).DimOccupancyFeatures
   ( OccupancyFeatureKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
ALTER TABLE $(use_schema).FactListingEstOcc ADD CONSTRAINT
   FK_$(use_schema)_FactListingEstOcc_SubFeatureKey FOREIGN KEY
   (
   SubFeatureKey
   ) REFERENCES $(use_schema).DimSubFeatures
   ( SubFeatureKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
ALTER TABLE $(use_schema).FactListingEstOcc ADD CONSTRAINT
   FK_$(use_schema)_FactListingEstOcc_PropertyTypeKey FOREIGN KEY
   (
   PropertyTypeKey
   ) REFERENCES $(use_schema).DimPropertyTypes
   ( PropertyTypeKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
ALTER TABLE $(use_schema).FactListingEstOcc ADD CONSTRAINT
   FK_$(use_schema)_FactListingEstOcc_ListingHostMapKey FOREIGN KEY
   (
   ListingHostMapKey
   ) REFERENCES $(use_schema).DimListingHostMap
   ( ListingHostMapKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
ALTER TABLE $(use_schema).FactListingEstOcc ADD CONSTRAINT
   FK_$(use_schema)_FactListingEstOcc_ListingPolicyKey FOREIGN KEY
   (
   ListingPolicyKey
   ) REFERENCES $(use_schema).DimPolicies
   ( ListingPolicyKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
ALTER TABLE $(use_schema).FactListingEstOcc ADD CONSTRAINT
   FK_$(use_schema)_FactListingEstOcc_AmenitiesKey FOREIGN KEY
   (
   AmenitiesKey
   ) REFERENCES $(use_schema).DimAmenities
   ( AmenitiesKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
ALTER TABLE $(use_schema).FactListingEstOcc ADD CONSTRAINT
   FK_$(use_schema)_FactListingEstOcc_PriceFeesKey FOREIGN KEY
   (
   PriceFeesKey
   ) REFERENCES $(use_schema).DimPriceFees
   ( PriceFeesKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;


--- FactHostReviewScores
--- 
ALTER TABLE $(use_schema).FactHostReviewScores ADD CONSTRAINT
   FK_$(use_schema)_FactHostReviewScores_LocationKey FOREIGN KEY
   (
   LocationKey
   ) REFERENCES $(use_schema).DimLocations
   ( LocationKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
ALTER TABLE $(use_schema).FactHostReviewScores ADD CONSTRAINT
   FK_$(use_schema)_FactHostReviewScores_OccupancyFeatureKey FOREIGN KEY
   (
   OccupancyFeatureKey
   ) REFERENCES $(use_schema).DimOccupancyFeatures
   ( OccupancyFeatureKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
ALTER TABLE $(use_schema).FactHostReviewScores ADD CONSTRAINT
   FK_$(use_schema)_FactHostReviewScores_SubFeatureKey FOREIGN KEY
   (
   SubFeatureKey
   ) REFERENCES $(use_schema).DimSubFeatures
   ( SubFeatureKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
ALTER TABLE $(use_schema).FactHostReviewScores ADD CONSTRAINT
   FK_$(use_schema)_FactHostReviewScores_PropertyTypeKey FOREIGN KEY
   (
   PropertyTypeKey
   ) REFERENCES $(use_schema).DimPropertyTypes
   ( PropertyTypeKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
ALTER TABLE $(use_schema).FactHostReviewScores ADD CONSTRAINT
   FK_$(use_schema)_FactHostReviewScores_ListingHostMapKey FOREIGN KEY
   (
   ListingHostMapKey
   ) REFERENCES $(use_schema).DimListingHostMap
   ( ListingHostMapKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
ALTER TABLE $(use_schema).FactHostReviewScores ADD CONSTRAINT
   FK_$(use_schema)_FactHostReviewScores_ListingPolicyKey FOREIGN KEY
   (
   ListingPolicyKey
   ) REFERENCES $(use_schema).DimPolicies
   ( ListingPolicyKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
ALTER TABLE $(use_schema).FactHostReviewScores ADD CONSTRAINT
   FK_$(use_schema)_FactHostReviewScores_AmenitiesKey FOREIGN KEY
   (
   AmenitiesKey
   ) REFERENCES $(use_schema).DimAmenities
   ( AmenitiesKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
ALTER TABLE $(use_schema).FactHostReviewScores ADD CONSTRAINT
   FK_$(use_schema)_FactHostReviewScores_PriceFeesKey FOREIGN KEY
   (
   PriceFeesKey
   ) REFERENCES $(use_schema).DimPriceFees
   ( PriceFeesKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
/*
ALTER TABLE $(use_schema).FactHostReviewScores ADD CONSTRAINT
   FK_$(use_schema)_FactFfPlanTypeProfits_FirstReviewDateKey FOREIGN KEY
   (
   FirstReviewDateKey
   ) REFERENCES $(use_schema).DimDate
   ( DateKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
 
ALTER TABLE $(use_schema).FactHostReviewScores ADD CONSTRAINT
   FK_$(use_schema)_FactFfPlanTypeProfits_LastReviewDateKey FOREIGN KEY
   (
   LastReviewDateKey
   ) REFERENCES $(use_schema).DimDate
   ( DateKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
*/

-- STEP 7: CREATE VIEWS  --


/*--------------- LOAD DIMENSION Tables ----------------------------------------- */
-----------------------------------------------------------------------------------


/* Load Dimension table dev_brook.DimPriceFees */

----------------------------------------------------------------------------------
/* 


*/
INSERT INTO $(dw_db).$(use_schema).[DimPriceFees]
	(
	  [ListingID]
	  ,[Price]
	  ,[WeeklyPrice]
	  ,[MonthlyPrice]
	  ,[SecurityDeposit]
	  ,[CleaningFee]
	  ,[GuestsIncluded]
	  ,[ExtraPeopleFee]
	  ,[HasWeeklyPrice]
	  ,[HasMonthlyPrice]
	  ,[HasSecurityDeposit]
	  ,[HasCleaningFee]
	  ,[HasExtraPeopleFee]
	  ,[CalendarAvailableDays]
	)
	SELECT 
			[listing_id]
      ,[price]
	  	,[weekly_price]
	  	,[monthly_price]
	  	,[security_deposit]
	  	,[cleaning_fee]
	  	,[guests_included]
	  	,[extra_people_fee]
	  ,case when [has_weekly_price] = 1 then 'Y' else 'N' end
	  	,case when [has_monthly_price] = 1 then 'Y' else 'N' end
	  	,case when [has_security_deposit] = 1 then 'Y' else 'N' end
	  	,case when [has_cleaning_fee] = 1 then 'Y' else 'N' end
	  	,case when [has_extra_people_fee] = 1 then 'Y' else 'N' end
	  	,[calendar_available_days]
FROM $(stage_db).$(use_schema).[ListingPriceFees3]

SELECT TOP(10) * FROM $(dw_db).$(use_schema).[DimPriceFees];



/* Load Dimension table dev_brook.DimListingHostMap */

----------------------------------------------------------------------------------
INSERT INTO $(dw_db).$(use_schema).[DimListingHostMap]
	(
	  [ListingID]
	  ,[HostID]
	  ,[HostName]
	  ,[ListingName]
	)
	SELECT 
			lh.[listing_id]
      ,lh.[host_id]
      ,CONCAT(lh.[host_name],'<',lh.[host_id],'>') as [HostName]
      ,ld.[listing_name]
FROM $(stage_db).$(use_schema).[listing_hosts] lh
	JOIN $(stage_db).$(use_schema).[ListingDescriptionText] ld
		ON lh.listing_id = ld.listing_id

SELECT TOP(10) * FROM $(dw_db).$(use_schema).[DimListingHostMap];



/* Load Dimension table dev_brook.DimLocations 
*/
----------------------------------------------------------------------------------
INSERT INTO $(dw_db).$(use_schema).[DimLocations]
	(
	  [ListingID]
	  ,[Neighbourhood]
	  ,[Neighbourhood_Group]
	  ,[ZipCode]
	  ,[Latitude]
	  ,[Longitude]
	)
	SELECT 
			[listing_id]
      ,[neighbourhood]
      ,[neighbourhood_group]
      ,[zipcode]
      ,[latitude]
      ,[longitude]
FROM $(stage_db).$(use_schema).[ListingLocations]

SELECT TOP(10) * FROM $(dw_db).$(use_schema).[DimLocations];



/* Load Dimension table dev_brook.DimAmenities */
----------------------------------------------------------------------------------
INSERT INTO $(dw_db).$(use_schema).[DimAmenities]
	(
	  [ListingID]
	  ,[AmenityName]
	)
	SELECT 
			[listing_id]
      ,[amenity]
FROM $(stage_db).$(use_schema).[listing_property_amenities]

SELECT TOP(10) * FROM $(dw_db).$(use_schema).[DimAmenities];



/* Load Dimension table dev_brook.DimPropertyTypes */
----------------------------------------------------------------------------------
INSERT INTO $(dw_db).$(use_schema).[DimPropertyTypes]
	(
	  [ListingID]
	  ,[PropertyTypeName]
	)
	SELECT 
			[listing_id]
      ,[property_type]
FROM $(stage_db).$(use_schema).[listing_property_types]

SELECT TOP(10) * FROM $(dw_db).$(use_schema).[DimPropertyTypes]



/* Load Dimension table dev_brook.DimSubFeatures */
----------------------------------------------------------------------------------
INSERT INTO $(dw_db).$(use_schema).[DimSubFeatures]
	(
	  [ListingID]
	  ,[SubFeatureName]
	)
	SELECT 
			[listing_id]
      ,[property_feature]
FROM $(stage_db).$(use_schema).[listing_property_sub_features]

SELECT TOP(10) * FROM $(dw_db).$(use_schema).[DimSubFeatures]



/* Load Dimension table dev_brook.DimPolicies 
*/
----------------------------------------------------------------------------------
INSERT INTO $(dw_db).$(use_schema).[DimPolicies]
	(
	  [ListingID]
	  ,[Minimum_Nights]
	  ,[Maximum_Nights]
	  ,[Cancellation_Policy]
	  ,[Instant_Bookable]
	)
	SELECT 
			[listing_id]
      ,[minimum_nights]
      ,[maximum_nights]
      ,[cancellation_policy]
    ,case when [instant_bookable] = 1 then 'Y' else 'N' end
FROM $(stage_db).$(use_schema).[ListingPolicies]

SELECT TOP(10) * FROM $(dw_db).$(use_schema).[DimPolicies]



/* Load Dimension table dev_brook.DimOccupancyFeatures */
----------------------------------------------------------------------------------
INSERT INTO $(dw_db).$(use_schema).[DimOccupancyFeatures]
	(
	  [ListingID]
	  ,[Accommodates]
	  ,[Bathrooms]
	  ,[Bedrooms]
	  ,[Beds]
	  ,[Bedroom_share] --calculated value bedrooms/accommades
	  ,[Bathroom_share] --calculated value bathrooms/accommodates
	)
	SELECT 
			[listing_id]
      ,[accommodates]
      ,[bathrooms]
      ,[bedrooms]
      ,[beds]
      ,[bedroom_share]
      ,[bathroom_share]
FROM $(stage_db).$(use_schema).[occupancy_features]

SELECT TOP(10) * FROM $(dw_db).$(use_schema).[DimOccupancyFeatures]




/* Load Dimension table dev_brook.DimNeighbourhoodScores 

*/
----------------------------------------------------------------------------------
INSERT INTO $(dw_db).$(use_schema).[DimNeighbourhoodScores]
	(
	  [NeighbourhoodName]
	  ,[Rank]
	  ,[Walk_Score]
	  ,[Transit_score]
	  ,[Bike_score] 
	  ,[Population] 
	)
	SELECT 
      [Name]
      ,[Rank]
      ,[Walk_Score]
      ,[Transit_Score]
      ,[Bike_Score]
      ,[Population]
FROM $(stage_db).$(use_schema).[NeighborhoodScores]

SELECT TOP(10) * FROM $(dw_db).$(use_schema).[DimNeighbourhoodScores]




/* Load Dimension table dev_brook.FactListingHostReviewScores */
----------------------------------------------------------------------------------
INSERT INTO $(dw_db).$(use_schema).[FactHostReviewScores]
	(
	--[ListingID]
	[LocationKey]
	,[OccupancyFeatureKey]
	,[SubFeatureKey]
	,[PropertyTypeKey]
	,[ListingHostMapKey]
	,[ListingPolicyKey]
	,[AmenitiesKey]
	,[PriceFeesKey]
	
	--# Review Score Facts
--	,[FirstReviewDateKey]
--	,[LastReviewDateKey]
	,[NumberOfReviews] 
	,[ReviewsPerMonth]
	,[MinimumNights]
	,[ReviewScoresRating]
	,[ReviewScoresAccuracy]
	,[ReviewScoresCleanliness]
	,[ReviewScoresCheckin]
	,[ReviewScoresCommunication]
	,[ReviewScoresLocation]
	,[ReviewScoresValue]
	,[AvgPolarityPositiveScore]
	,[AvgPolarityNegativeScore]
	,[AvgPolarityCompoundScore]
	,[ReviewDaysRange]
	,[ReviewYearsRange]
	,[ReviewsPerYearRate]
	,[ReviewsFrequency]
	)
	SELECT 
	--rs.[listing_id]
   l.[LocationKey]
	,ocf.[OccupancyFeatureKey]
	,sf.[SubFeatureKey]
	,pt.[PropertyTypeKey]
	,lh.[ListingHostMapKey]
	,p.[ListingPolicyKey]
	,a.[AmenitiesKey]
	,pf.[PriceFeesKey]
	--,rs.[FirstReviewDateKey]
	--,rs.[LastReviewDateKey]
	,rs.[number_of_reviews] 
	,rs.[reviews_per_month]
	,rs.[minimum_nights]
	,rs.[review_scores_rating]
	,rs.[review_scores_accuracy]
	,rs.[review_scores_cleanliness]
	,rs.[review_scores_checkin]
	,rs.[review_scores_communication]
	,rs.[review_scores_location]
	,rs.[review_scores_value]
	,rs.[avg_pol_pos_score]
	,rs.[avg_pol_neg_score]
	,rs.[avg_pol_compound_score]
	,rs.[review_days_range]
	,rs.[review_years_range]
	,rs.[reviews_per_year_rate]
	,rs.[review_frequency]
FROM $(stage_db).$(use_schema).[ListingReviewScores2] rs
	JOIN $(dw_db).$(use_schema).[DimSubFeatures] sf
		ON	rs.listing_id = sf.ListingID
	JOIN $(dw_db).$(use_schema).[DimPropertyTypes] pt
		ON	rs.listing_id = pt.ListingID
	JOIN $(dw_db).$(use_schema).[DimOccupancyFeatures] ocf
		ON rs.listing_id = ocf.ListingID
	JOIN $(dw_db).$(use_schema).[DimLocations] l
		ON rs.listing_id = l.ListingID
	JOIN $(dw_db).$(use_schema).[DimListingHostMap] lh
		ON rs.listing_id = lh.ListingID
	JOIN $(dw_db).$(use_schema).[DimPolicies] p
		ON rs.listing_id = p.ListingID
	JOIN $(dw_db).$(use_schema).[DimAmenities] a
		ON rs.listing_id = a.ListingID
	JOIN $(dw_db).$(use_schema).[DimPriceFees] pf
		ON rs.listing_id = pf.ListingID

SELECT TOP(10) * FROM $(dw_db).$(use_schema).[FactHostReviewScores]



/* Load Dimension table dev_brook.FactListingEstOcc */
----------------------------------------------------------------------------------
INSERT INTO $(dw_db).$(use_schema).[FactListingEstOcc]
	(
	--[ListingID]
	[LocationKey]
	,[OccupancyFeatureKey]
	,[SubFeatureKey]
	,[PropertyTypeKey]
	,[ListingHostMapKey]
	,[ListingPolicyKey]
	,[AmenitiesKey]
	,[PriceFeesKey]
	
--# Calculated Facts --
	,[EstLifetimeOccupancyDays]
	,[EstLifetimeOccDailyRate]
	,[EstLifetimeOccYearlyRate]
	,[EstLifetimeIncome]
	,[EstLifetimeYearlyIncome]
	,[EstLifetimePotentialIncome]
	,[EstLifetimePotentialYearlyIncome]
	,[EstLifetimePercentOfPotentialIncome]
	,[LifetimeEarner]
	)
	SELECT 
	--eo.[listing_id]
   l.[LocationKey]
	,ocf.[OccupancyFeatureKey]
	,sf.[SubFeatureKey]
	,pt.[PropertyTypeKey]
	,lh.[ListingHostMapKey]
	,p.[ListingPolicyKey]
	,a.[AmenitiesKey]
	,pf.[PriceFeesKey]
	,eo.[est_lifetime_occ]
	,eo.[est_lifetime_occ_daily_rate]
	,eo.[est_lifetime_occ_yearly_rate]
	,eo.[est_lifetime_income]
	,eo.[est_lifetime_yearly_income]
	,eo.[est_lifetime_potential_income]
	,eo.[est_lifetime_potential_yearly_income]
	,eo.[est_perc_yearly_income_of_potential]
	,eo.[quantile]
FROM $(stage_db).$(use_schema).[EstimatedOccupancy] eo
	JOIN $(dw_db).$(use_schema).[DimSubFeatures] sf
		ON	eo.listing_id = sf.ListingID
	JOIN $(dw_db).$(use_schema).[DimPropertyTypes] pt
		ON	eo.listing_id = pt.ListingID
	JOIN $(dw_db).$(use_schema).[DimOccupancyFeatures] ocf
		ON eo.listing_id = ocf.ListingID
	JOIN $(dw_db).$(use_schema).[DimLocations] l
		ON eo.listing_id = l.ListingID
	JOIN $(dw_db).$(use_schema).[DimListingHostMap] lh
		ON eo.listing_id = lh.ListingID
	JOIN $(dw_db).$(use_schema).[DimPolicies] p
		ON eo.listing_id = p.ListingID
	JOIN $(dw_db).$(use_schema).[DimAmenities] a
		ON eo.listing_id = a.ListingID
	JOIN $(dw_db).$(use_schema).[DimPriceFees] pf
		ON eo.listing_id = pf.ListingID

SELECT TOP(10) * FROM $(dw_db).$(use_schema).[FactListingEstOcc]