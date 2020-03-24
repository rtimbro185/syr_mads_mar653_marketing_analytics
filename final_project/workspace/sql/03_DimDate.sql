-- DECLARE CONFIG VARIABLES --

-- IN SQL Server Manager Studio - Under 'Query' select SQLCMD Mode - to utilize command line variables for configurations
-- DATABASE CONFIGURATIONS -- UNCOMMENT ONLY ONE DATABASE
--                           

:setvar stage_db mar653_airbnb_stage
:setvar dw_db mar653_airbnb_dw
--print(N'$(use_database)')
--

-- SCHEMA CONFIGURATIONS -- UNCOMMENT ONLY ONE SCHEMA
--
:setvar use_schema dev_brook       -- development schema
--:setvar use_schema test_brook			-- testing/validation schema
--:setvar use_schema brook						-- production schema
print(N'$(use_schema)')

-- #################################################### --



---------------------------------------------
-- PHASE 1: Create ddl
---------------------------------------------

-- STEP 1: DROP VIEWS IF THEY EXIST --

-- STEP 2: DROP TABLES IF THEY EXIST --
---------------------------------------

-- CONSIDER COMMENTING THIS OUT AFTER EVERTHING IS SET AND WORKING --

/* Drop brook.StgDates | Dependencies on:  */
USE $(stage_db);
GO
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'$(use_schema).StgDates') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE $(use_schema).StgDates


/* Drop table brook.DimDate */
USE $(dw_db);
GO
IF EXISTS (SELECT * FROM dbo.sysobjects WHERE id = OBJECT_ID(N'$(use_schema).DimDate') AND OBJECTPROPERTY(id, N'IsUserTable') = 1)
DROP TABLE $(use_schema).DimDate 
;



/* Create table brook.DimDate */
USE $(dw_db);
GO
CREATE TABLE $(dw_db).$(use_schema).DimDate (
   [DateKey]  int   NOT NULL
,  [Date]  datetime   NULL
,  [FullDateUSA]  nchar(11)   NOT NULL
,  [DayOfWeek]  tinyint   NOT NULL
,  [DayName]  nchar(10)   NOT NULL
,  [DayOfMonth]  tinyint   NOT NULL
,  [DayOfYear]  int   NOT NULL
,  [WeekOfYear]  tinyint   NOT NULL
,  [MonthName]  nchar(10)   NOT NULL
,  [MonthOfYear]  tinyint   NOT NULL
,  [Quarter]  tinyint   NOT NULL
,  [QuarterName]  nchar(10)   NOT NULL
,  [Year]  int   NOT NULL
,  [IsAWeekday]  nchar(1)  DEFAULT 'N' NOT NULL
, CONSTRAINT [PK_$(use_schema).DimDate] PRIMARY KEY CLUSTERED 
( [DateKey] )
) ON [PRIMARY]
;

INSERT INTO $(dw_db).$(use_schema).DimDate (DateKey, Date, FullDateUSA, DayOfWeek, DayName, DayOfMonth, DayOfYear, WeekOfYear, MonthName, MonthOfYear, Quarter, QuarterName, Year, IsAWeekday)
VALUES (-1, '', 'Unk date', 0, 'Unk date', 0, 0, 0, 'Unk month', 0, 0, 'Unk qtr', 0, 'N')
;




---------------------------------------------
-- PHASE 2: STAGE DimDate
---------------------------------------------
USE $(stage_db);
GO

--- STAGE ORDER DATES ---
SELECT *
INTO $(stage_db).$(use_schema).[StgDates]
FROM [master].[dbo].[date_dimension] -- Using ExternalSources2 rather than ExternalSources which is offline
WHERE Year between 1996 and 2015

--- Validate Staging ---
/****** Script for SelectTopNRows command from SSMS  ******/
SELECT TOP (10) [DateKey]
      ,[Date]
      ,[FullDateUK]
      ,[FullDateUSA]
      ,[DayOfMonth]
      ,[DaySuffix]
      ,[DayName]
      ,[DayOfWeekUSA]
      ,[DayOfWeekUK]
      ,[DayOfWeekInMonth]
      ,[DayOfWeekInYear]
      ,[DayOfQuarter]
      ,[DayOfYear]
      ,[WeekOfMonth]
      ,[WeekOfQuarter]
      ,[WeekOfYear]
      ,[Month]
      ,[MonthName]
      ,[MonthOfQuarter]
      ,[Quarter]
      ,[QuarterName]
      ,[Year]
      ,[YearName]
      ,[MonthYear]
      ,[MMYYYY]
      ,[FirstDayOfMonth]
      ,[LastDayOfMonth]
      ,[FirstDayOfQuarter]
      ,[LastDayOfQuarter]
      ,[FirstDayOfYear]
      ,[LastDayOfYear]
      ,[IsWeekday]
      ,[IsWeekdayYesNo]
      ,[IsHolidayUSA]
      ,[IsHolidayUSAYesNo]
      ,[HolidayNameUSA]
      ,[IsHolidayUK]
      ,[HolidayNameUK]
      ,[FiscalDayOfYear]
      ,[FiscalWeekOfYear]
      ,[FiscalMonth]
      ,[FiscalQuarter]
      ,[FiscalQuarterName]
      ,[FiscalYear]
      ,[FiscalYearName]
      ,[FiscalMonthYear]
      ,[FiscalMMYYYY]
      ,[FiscalFirstDayOfMonth]
      ,[FiscalLastDayOfMonth]
      ,[FiscalFirstDayOfQuarter]
      ,[FiscalLastDayOfQuarter]
      ,[FiscalFirstDayOfYear]
      ,[FiscalLastDayOfYear]
  FROM $(stage_db).$(use_schema).[StgDates]

--###########################################################################################################################################


---------------------------------------------
-- PHASE 3: LOAD DimDate
---------------------------------------------

/* LOAD DimDate
	Load Date Dimension
*/
--- Test Select Statement ---
SELECT TOP (10) [DateKey]
	,[Date]
      ,[FullDateUSA]
      ,[DayOfWeekUSA]
      ,[DayName]
      ,[DayOfMonth]
      ,[DayOfYear]
      ,[WeekOfYear]
      ,[MonthName]
      ,[Month]
      ,[Quarter]
      ,[QuarterName]
      ,[Year]
      ,[IsWeekday]
FROM $(stage_db).$(use_schema).[StgDates]

--- ****Load DimDate --- **Note: DimDate MonthOfYear maps to StgNorthwindDates Month**
--- First load unknown member 
--INSERT INTO $(stage_db).$(use_schema).[StgDates] (DateKey, Date, FullDateUSA, DayOfWeek, DayName, DayOfMonth, DayOfYear, WeekOfYear, MonthName, MonthOfYear, Quarter, QuarterName, Year, IsWeekday)
--VALUES (-1, '', 'Unk date', 0, 'Unk date', 0, 0, 0, 'Unk month', 0, 0, 'Unk qtr', 0, 0)
--;


INSERT INTO $(dw_db).$(use_schema).[DimDate]
	([DateKey],[Date],[FullDateUSA],[DayOfWeek],[DayName],[DayOfMonth],[DayOfYear],[WeekOfYear],[MonthName],[MonthOfYear],[Quarter],[QuarterName],[Year],[IsAWeekday])
	SELECT [DateKey]
		,[Date]
		,[FullDateUSA]
		,[DayOfWeekUSA]
		,[DayName]
		,[DayOfMonth]
		,[DayOfYear]
		,[WeekOfYear]
		,[MonthName]
		,[Month]
		,[Quarter]
		,[QuarterName]
		,[Year]
	,case when [IsWeekday] = 1 then 'Y' else 'N' end
	FROM $(stage_db).$(use_schema).StgDates

-- Validate Data Load --
SELECT TOP (10) [DateKey]
      ,[Date]
      ,[FullDateUSA]
      ,[DayOfWeek]
      ,[DayName]
      ,[DayOfMonth]
      ,[DayOfYear]
      ,[WeekOfYear]
      ,[MonthName]
      ,[MonthOfYear]
      ,[Quarter]
      ,[QuarterName]
      ,[Year]
      ,[IsAWeekday]
  FROM $(dw_db).$(use_schema).[DimDate]