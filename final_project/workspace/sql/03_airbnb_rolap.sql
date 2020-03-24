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

--:setvar stage_db mar653_airbnb_stage
--:setvar dw_db mar653_airbnb_dw
--print(N'$(use_database)')
--

-- SCHEMA CONFIGURATIONS -- UNCOMMENT ONLY ONE SCHEMA
--
:setvar use_schema dev_brook       -- development schema

-- IN SQL Server Manager Studio - Under 'Query' select SQLCMD Mode - to utilize command line variables for configurations
-- DATABASE CONFIGURATIONS -- UNCOMMENT ONLY ONE DATABASE
--                           
:setvar use_database mar653_airbnb_dw         -- my local development DW database

print(N'$(use_database)')
--

-- SCHEMA CONFIGURATIONS -- UNCOMMENT ONLY ONE SCHEMA
--
--:setvar use_schema dev_fudgeinc       -- development schema
--:setvar use_schema test_fudgeinc			-- testing/validation schema
--:setvar use_schema fudgeinc						-- production schema
print(N'$(use_schema)')

-- #################################################### --


-- ROLAP Start Schema Creation --
USE $(use_database);
GO

-- STEP 1: DROP VIEWS IF THEY EXIST --




-- STEP 2: DROP TABLES IF THEY EXIST --
-- DimPropertyTypes
-- DimRoomTypes
-- DimNeighbourhoods
-- DimFeatures
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

/* Drop table fudgeinc.DimRoomTypes */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'$(use_schema).DimRoomTypes') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE $(use_schema).DimRoomTypes 
;

/* Drop table fudgeinc.DimPropertyTypes */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'$(use_schema).DimPropertyTypes') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE $(use_schema).DimPropertyTypes 
;

/* Drop table fudgeinc.DimFeatures */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'$(use_schema).DimFeatures') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE $(use_schema).DimFeatures 
;

/* Drop table fudgeinc.DimAmenitites */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'$(use_schema).DimAmenitites') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE  $(use_schema).DimAmenitites 
;


/* Drop table fudgeinc.DimLocations */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'$(use_schema).DimLocations') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE $(use_schema).DimLocations 
;

/* Drop table fudgeinc.DimNeighbourhoods */
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'$(use_schema).DimNeighbourhoods') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE $(use_schema).DimNeighbourhoods 
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
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'$(use_schema).DimListingAmenties') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE $(use_schema).DimListingAmenties 
;

-- STEP 4: CREAT SCHEMAS  --
----------------------------
IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = N'$(use_schema)')
	EXEC('CREATE SCHEMA $(use_schema)');
GO


-- STEP 5: CREAT TABLES  --
---------------------------
-- DimPropertyTypes
-- DimRoomTypes
-- DimPolicies
-- DimNeighbourhoods
-- DimFeatures
-- DimAmenitites
-- DimListingAmenties
-- DimListingsHostsMap
-- FactListingHostReviewScores
-- DimLocations
-- FactListingEstOcc
---------------------------

/* Create table fudgeinc.DimPolicies */
CREATE TABLE $(use_schema).DimListingAmenties (
   [ListingAmentiesKey]  int IDENTITY  NOT NULL
,  [ListingAmentiesID]  int   NOT NULL
,  [Amenity]  varchar(50)   NOT NULL
,  [RowIsCurrent]  bit   DEFAULT 1 NOT NULL
,  [RowStartDate]  datetime  DEFAULT '12/31/1899' NOT NULL
,  [RowEndDate]  datetime  DEFAULT '12/31/9999' NOT NULL
,  [RowChangeReason]  nvarchar(200)   NULL
, CONSTRAINT [PK_$(use_schema).DimListingAmenties] PRIMARY KEY CLUSTERED 
( [ListingAmentiesKey] )
) ON [PRIMARY]
;

SET IDENTITY_INSERT $(use_schema).DimListingAmenties ON
;
INSERT INTO $(use_schema).DimListingAmenties (ListingAmentiesKey, ListingAmentiesID, Amenity, RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, -1, 'None',-1, '12/31/1899', '12/31/9999', 'N/A')
;
SET IDENTITY_INSERT $(use_schema).DimListingAmenties OFF
;

/* Create table fudgeinc.DimPolicies */
CREATE TABLE $(use_schema).DimPolicies (
   [ListingPolicyKey]  int IDENTITY  NOT NULL
,  [ListingPolicyID]  int   NOT NULL
,  [Minimum_Nights]  smallint   NOT NULL
,  [Maximum_Nights] smallint NOT NULL
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
INSERT INTO $(use_schema).DimPolicies (ListingPolicyKey, ListingPolicyID, Minimum_Nights, Maximum_Nights, Cancellation_Policy, Instant_Bookable, RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, -1, -1,-1,'None','f',-1, '12/31/1899', '12/31/9999', 'N/A')
;
SET IDENTITY_INSERT $(use_schema).DimPolicies OFF
;

/* Create table fudgeinc.DimListingHostMap */
CREATE TABLE $(use_schema).DimListingHostMap (
   [ListingHostMapKey]  int IDENTITY  NOT NULL
,  [ListingHostMapID]  int   NOT NULL
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
INSERT INTO $(use_schema).DimListingHostMap (ListingHostMapKey, ListingHostMapID, ListingName, RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, -1, 'None',-1, '12/31/1899', '12/31/9999', 'N/A')
;
SET IDENTITY_INSERT $(use_schema).DimListingHostMap OFF
;

/* Create table fudgeinc.DimPropertyTypes */
CREATE TABLE $(use_schema).DimPropertyTypes (
   [PropertyTypeKey]  int IDENTITY  NOT NULL
,  [PropertyTypeID]  int   NOT NULL
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
INSERT INTO $(use_schema).DimPropertyTypes (PropertyTypeKey, PropertyTypeID, PropertyTypeName, RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, -1, 'None', -1, '12/31/1899', '12/31/9999', 'N/A')
;
SET IDENTITY_INSERT $(use_schema).DimPropertyTypes OFF
;


/* Create table fudgeinc.DimRoomTypes */
CREATE TABLE $(use_schema).DimRoomTypes (
   [RoomTypeKey]  int IDENTITY  NOT NULL
,  [RoomTypeID]  int   NOT NULL
,  [RoomTypeName]  varchar(100)   NOT NULL
,  [RowIsCurrent]  bit   DEFAULT 1 NOT NULL
,  [RowStartDate]  datetime  DEFAULT '12/31/1899' NOT NULL
,  [RowEndDate]  datetime  DEFAULT '12/31/9999' NOT NULL
,  [RowChangeReason]  nvarchar(200)   NULL
, CONSTRAINT [PK_$(use_schema).DimRoomTypes] PRIMARY KEY CLUSTERED 
( [RoomTypeKey] )
) ON [PRIMARY]
;

SET IDENTITY_INSERT $(use_schema).DimRoomTypes ON
;
INSERT INTO $(use_schema).DimRoomTypes (RoomTypeKey, RoomTypeID, RoomTypeName, RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, -1, 'None', -1, '12/31/1899', '12/31/9999', 'N/A')
;
SET IDENTITY_INSERT $(use_schema).DimRoomTypes OFF
;

/* Create table fudgeinc.DimNeighbourhoods */
CREATE TABLE $(use_schema).DimNeighbourhoods (
   [NeighbourhoodKey]  int IDENTITY  NOT NULL
,  [NeighbourhoodID]  int   NOT NULL
,  [NeighbourhoodName]  varchar(100)   NOT NULL
,	 [Walk_Score] smallint NOT NULL
,	 [Transit_score] smallint NOT NULL
,	 [Bike_score] smallint NOT NULL
,	 [Population] int NOT NULL
,  [RowIsCurrent]  bit   DEFAULT 1 NOT NULL
,  [RowStartDate]  datetime  DEFAULT '12/31/1899' NOT NULL
,  [RowEndDate]  datetime  DEFAULT '12/31/9999' NOT NULL
,  [RowChangeReason]  nvarchar(200)   NULL
, CONSTRAINT [PK_$(use_schema).DimNeighbourhoods] PRIMARY KEY CLUSTERED 
( [NeighbourhoodKey] )
) ON [PRIMARY]
;

SET IDENTITY_INSERT $(use_schema).DimNeighbourhoods ON
;
INSERT INTO $(use_schema).DimNeighbourhoods (NeighbourhoodKey, NeighbourhoodID, NeighbourhoodName, Walk_Score, Transit_score, Bike_score, Population,RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, -1, 'None', -1,-1,-1,-1, 1, '12/31/1899', '12/31/9999', 'N/A')
;
SET IDENTITY_INSERT $(use_schema).DimNeighbourhoods OFF
;


/* Create table fudgeinc.DimFeatures */
CREATE TABLE $(use_schema).DimFeatures (
   [FeatureKey]  int IDENTITY  NOT NULL
,  [FeatureID]  int   NOT NULL
,  [FeatureName]  varchar(100)   NOT NULL
,  [RowIsCurrent]  bit   DEFAULT 1 NOT NULL
,  [RowStartDate]  datetime  DEFAULT '12/31/1899' NOT NULL
,  [RowEndDate]  datetime  DEFAULT '12/31/9999' NOT NULL
,  [RowChangeReason]  nvarchar(200)   NULL
, CONSTRAINT [PK_$(use_schema).DimFeatures] PRIMARY KEY CLUSTERED 
( [FeatureKey] )
) ON [PRIMARY]
;

SET IDENTITY_INSERT $(use_schema).DimFeatures ON
;
INSERT INTO $(use_schema).DimFeatures (FeatureKey, FeatureID, FeatureName, RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, -1, 'None', -1, '12/31/1899', '12/31/9999', 'N/A')
;
SET IDENTITY_INSERT $(use_schema).DimFeatures OFF
;


/* Create table fudgeinc.DimAmenitites */
CREATE TABLE $(use_schema).DimAmenitites (
   [AmenititesKey]  int IDENTITY  NOT NULL
,  [AmenitiesID]  int   NOT NULL
,  [AmenitiesName]  varchar(100)   NOT NULL
,  [RowIsCurrent]  bit   DEFAULT 1 NOT NULL
,  [RowStartDate]  datetime  DEFAULT '12/31/1899' NOT NULL
,  [RowEndDate]  datetime  DEFAULT '12/31/9999' NOT NULL
,  [RowChangeReason]  nvarchar(200)   NULL
, CONSTRAINT [PK_$(use_schema).DimAmenitites] PRIMARY KEY CLUSTERED 
( [AmenititesKey] )
) ON [PRIMARY]
;

SET IDENTITY_INSERT $(use_schema).DimAmenitites ON
;
INSERT INTO $(use_schema).DimAmenitites (AmenititesKey, AmenitiesID, AmenitiesName, RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, -1, 'None', -1, '12/31/1899', '12/31/9999', 'N/A')
;
SET IDENTITY_INSERT $(use_schema).DimAmenitites OFF
;


/* Create table fudgeinc.DimLocations */
CREATE TABLE $(use_schema).DimLocations (
   [LocationKey]  int IDENTITY  NOT NULL
,  [LocationID]  int   NOT NULL
,  [Neighbourhood]  varchar(100)   NOT NULL
,  [Neighbourhood_Group]  varchar(100)   NOT NULL
,	 [ZipCode] varchar(5) NOT NULL
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
INSERT INTO $(use_schema).DimLocations (LocationKey, LocationID, Neighbourhood, Neighbourhood_Group, ZipCode, Latitude, Longitude, RowIsCurrent, RowStartDate, RowEndDate, RowChangeReason)
VALUES (-1, -1, 'None', 'None','None',-1, -1,1, '12/31/1899', '12/31/9999', 'N/A')
;
SET IDENTITY_INSERT $(use_schema).DimLocations OFF
;


/* Create table fudgeinc.FactListingEstOcc */
/*
CREATE TABLE $(use_schema).FactListingEstOcc (
   [LocationKey]  int   NOT NULL
,  [AmenititesKey]  int   NOT NULL
,  [AccountKey]  int   NOT NULL
,  [AccountBilledDateKey]  int   NOT NULL
,	 [AccountBilledAmount]  decimal(25,2)   NOT NULL
,  [PlanPriceAmount]  decimal(25,2)   NOT NULL
,  [AccountTotalBilledAmount] decimal(25,2)   NOT NULL
,  [AccountTotalQuantityBilled] int NOT NULL
,  [PlanTotalBilledAmount]  decimal(25,2)   NOT NULL
,  [PlanTotalQuantityBilled]  int   NOT NULL
,  [SnapeshotDateKey]  int   NOT NULL
, CONSTRAINT [PK_$(use_schema).FactListingEstOcc] PRIMARY KEY NONCLUSTERED 
( [PlanKey], [AccountKey], [AccountBillingKey] )
) ON [PRIMARY]
;
*/



/* Create table fudgeinc.FactHostReviewScores */
/*
CREATE TABLE $(use_schema).FactHostReviewScores (
   [ListingHostReviewKey]  int IDENTITY  NOT NULL
,  [ListingHostReviewID]  int   NOT NULL
,  [HostName]  varchar(100)   NOT NULL
,

,  [RowIsCurrent]  bit   DEFAULT 1 NOT NULL
,  [RowStartDate]  datetime  DEFAULT '12/31/1899' NOT NULL
,  [RowEndDate]  datetime  DEFAULT '12/31/9999' NOT NULL
,  [RowChangeReason]  nvarchar(200)   NULL
, CONSTRAINT [PK_$(use_schema).FactHostReviewScores] PRIMARY KEY CLUSTERED 
( [FeatureKey] )
) ON [PRIMARY]
;
*/

-- STEP 6: ADD TABLE CONSTRAINTS  --
------------------------------------
---
/*
ALTER TABLE $(use_schema).DimFfAccounts ADD CONSTRAINT
   FK_$(use_schema)_DimFfAccounts_AccountOpenedDateKey FOREIGN KEY
   (
   AccountOpenedDateKey
   ) REFERENCES $(use_schema).DimDate
   ( DateKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
 
ALTER TABLE $(use_schema).DimFfAccountBilling ADD CONSTRAINT
   FK_$(use_schema)_DimFfAccountBilling_AccountBillingDateKey FOREIGN KEY
   (
   AccountBillingDateKey
   ) REFERENCES $(use_schema).DimDate
   ( DateKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;


ALTER TABLE $(use_schema).DimFfAccountTitles ADD CONSTRAINT
   FK_$(use_schema)_DimFfAccountTitles_QueuedDateKey FOREIGN KEY
   (
   QueuedDateKey
   ) REFERENCES $(use_schema).DimDate
   ( DateKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
 
ALTER TABLE $(use_schema).DimFfAccountTitles ADD CONSTRAINT
   FK_$(use_schema)_DimFfAccountTitles_ShippedDateKey FOREIGN KEY
   (
   ShippedDateKey
   ) REFERENCES $(use_schema).DimDate
   ( DateKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
ALTER TABLE $(use_schema).DimFfAccountTitles ADD CONSTRAINT
   FK_$(use_schema)_DimFfAccountTitles_ReturenedDateKey FOREIGN KEY
   (
   ReturenedDateKey
   ) REFERENCES $(use_schema).DimDate
   ( DateKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
-----

ALTER TABLE $(use_schema).FactFfPlanTypeProfits ADD CONSTRAINT
   FK_$(use_schema)_FactFfPlanTypeProfits_PlanKey FOREIGN KEY
   (
   PlanKey
   ) REFERENCES $(use_schema).DimFfPlans
   ( PlanKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
 
ALTER TABLE $(use_schema).FactFfPlanTypeProfits ADD CONSTRAINT
   FK_$(use_schema)_FactFfPlanTypeProfits_AccountKey FOREIGN KEY
   (
   AccountKey
   ) REFERENCES $(use_schema).DimFfAccounts
   ( AccountKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
 
ALTER TABLE $(use_schema).FactFfPlanTypeProfits ADD CONSTRAINT
   FK_$(use_schema)_FactFfPlanTypeProfits_AccountBilledDateKey FOREIGN KEY
   (
   AccountBilledDateKey
   ) REFERENCES $(use_schema).DimDate
   ( DateKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
 
ALTER TABLE $(use_schema).FactFfPlanTypeProfits ADD CONSTRAINT
   FK_$(use_schema)_FactFfPlanTypeProfits_SnapeshotDateKey FOREIGN KEY
   (
   SnapeshotDateKey
   ) REFERENCES $(use_schema).DimDate
   ( DateKey )
     ON UPDATE  NO ACTION
     ON DELETE  NO ACTION
;
*/

-- STEP 7: CREATE VIEWS  --