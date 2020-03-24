-- STEP CREAT SCHEMAS  --
----------------------------
IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = N'dev_brook')
	EXEC('CREATE SCHEMA dev_brook');
GO

IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = N'test_brook')
	EXEC('CREATE SCHEMA test_brook');
GO

IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = N'brook')
	EXEC('CREATE SCHEMA brook');
GO

SELECT * FROM sys.schemas WHERE name = N'dev_brook';
SELECT * FROM sys.schemas WHERE name = N'test_brook';
SELECT * FROM sys.schemas WHERE name = N'brook';