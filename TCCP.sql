/* =======================================================
   TELCO CUSTOMER CHURN - COMPLETE DATA PIPELINE
   =======================================================
   Purpose:  Create and populate CustomerChurnDB for ML training
   
   This script is FULLY RE-RUNNABLE - safe to execute multiple times
   ======================================================= */

/* -------------------------------------------------------------------------- */
/* 1. DATABASE CREATION - Run only if not exists                             */
/* -------------------------------------------------------------------------- */
IF DB_ID('CustomerChurnDB') IS NULL
BEGIN
    CREATE DATABASE CustomerChurnDB;
    PRINT 'Database CustomerChurnDB created.';
END
ELSE
    PRINT 'Database CustomerChurnDB already exists.';
GO

USE CustomerChurnDB;
GO

-- Ensure consistent behavior for random number generation
SET NOCOUNT ON;
GO

/* -------------------------------------------------------------------------- */
/* 2. CLEANUP - Drop existing objects if they exist (RE-RUNNABLE)            */
/* -------------------------------------------------------------------------- */
PRINT '--- Dropping existing objects (if any) ---';
GO

-- Tables
IF OBJECT_ID('stg_telco_raw', 'U') IS NOT NULL 
    DROP TABLE stg_telco_raw;
IF OBJECT_ID('support_tickets', 'U') IS NOT NULL 
    DROP TABLE support_tickets;
IF OBJECT_ID('usage_data', 'U') IS NOT NULL 
    DROP TABLE usage_data;
IF OBJECT_ID('billing', 'U') IS NOT NULL 
    DROP TABLE billing;
IF OBJECT_ID('customers', 'U') IS NOT NULL 
    DROP TABLE customers;
IF OBJECT_ID('churn_labels', 'U') IS NOT NULL 
    DROP TABLE churn_labels;

-- Views
IF OBJECT_ID('vw_usage_agg', 'V') IS NOT NULL 
    DROP VIEW vw_usage_agg;
IF OBJECT_ID('vw_support_agg', 'V') IS NOT NULL 
    DROP VIEW vw_support_agg;
IF OBJECT_ID('vw_churn_training_features', 'V') IS NOT NULL 
    DROP VIEW vw_churn_training_features;
IF OBJECT_ID('vw_churn_deployment_features', 'V') IS NOT NULL 
    DROP VIEW vw_churn_deployment_features;
GO

/* -------------------------------------------------------------------------- */
/* 3. STAGING TABLE - Raw data import from CSV                               */
/* -------------------------------------------------------------------------- */
PRINT '--- Creating staging table ---';
GO

CREATE TABLE stg_telco_raw (
    customerID VARCHAR(50),
    gender VARCHAR(10),
    SeniorCitizen INT,
    Partner VARCHAR(5),
    Dependents VARCHAR(5),
    tenure INT,
    PhoneService VARCHAR(10),
    MultipleLines VARCHAR(25),
    InternetService VARCHAR(25),
    OnlineSecurity VARCHAR(25),
    OnlineBackup VARCHAR(25),
    DeviceProtection VARCHAR(25),
    TechSupport VARCHAR(25),
    StreamingTV VARCHAR(25),
    StreamingMovies VARCHAR(25),
    Contract VARCHAR(30),
    PaperlessBilling VARCHAR(10),
    PaymentMethod VARCHAR(50),
    MonthlyCharges DECIMAL(10,2),
    TotalCharges VARCHAR(20),
    Churn VARCHAR(5)
);
GO

PRINT '--- Bulk inserting data from CSV (RE-RUNNABLE: TRUNCATE first) ---';
GO

TRUNCATE TABLE stg_telco_raw;

BULK INSERT stg_telco_raw
FROM 'C:\Users\satya\Projects\Data Science Projects\Telco_Customer_Churn_Prediction\Telco_Customer_Churn.csv'
WITH (
    FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',
    TABLOCK
);
GO

PRINT '--- Staging table loaded. ---';
GO

/* -------------------------------------------------------------------------- */
/* 4. CORE TABLES - Star schema design                                       */
/* -------------------------------------------------------------------------- */
PRINT '--- Creating dimension and fact tables ---';
GO

-- Customers dimension
CREATE TABLE customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    gender VARCHAR(10),
    senior_citizen BIT,
    partner VARCHAR(5),
    dependents VARCHAR(5),
    age INT,
    region VARCHAR(50),
    signup_date DATE,
    contract_type VARCHAR(30)
);

-- Billing fact table
CREATE TABLE billing (
    billing_id INT IDENTITY(1,1) PRIMARY KEY,
    customer_id VARCHAR(50),
    monthly_charges DECIMAL(10,2),
    total_charges DECIMAL(12,2),
    payment_method VARCHAR(50),
    late_payments INT DEFAULT 0,
    billing_date DATE,
    paperless_billing VARCHAR(10)
);

-- Usage fact table
CREATE TABLE usage_data (
    usage_id INT IDENTITY(1,1) PRIMARY KEY,
    customer_id VARCHAR(50),
    avg_call_minutes DECIMAL(10,2),
    avg_data_usage_gb DECIMAL(10,2),
    internet_service VARCHAR(25),
    phone_service VARCHAR(10),
    multiple_lines VARCHAR(25),
    usage_month DATE
);

-- Support tickets fact table
CREATE TABLE support_tickets (
    ticket_id INT IDENTITY(1,1) PRIMARY KEY,
    customer_id VARCHAR(50),
    ticket_type VARCHAR(25),
    resolution_time_hr DECIMAL(10,2),
    satisfaction_score INT,
    ticket_date DATE
);

-- Churn labels dimension
CREATE TABLE churn_labels (
    customer_id VARCHAR(50) PRIMARY KEY,
    churn BIT,
    churn_date DATE
);
GO

/* -------------------------------------------------------------------------- */
/* 5. DATA POPULATION - Insert transformed data (RE-RUNNABLE)                */
/* -------------------------------------------------------------------------- */
PRINT '--- Truncating existing data from tables ---';
GO

TRUNCATE TABLE customers;
TRUNCATE TABLE billing;
TRUNCATE TABLE usage_data;
TRUNCATE TABLE support_tickets;
TRUNCATE TABLE churn_labels;
GO

PRINT '--- Inserting data into customers table ---';
GO

INSERT INTO customers (
    customer_id, gender, senior_citizen, partner, dependents,
    age, region, signup_date, contract_type
)
SELECT
    customerID,
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    CASE
        WHEN SeniorCitizen = 1 THEN 65 + (tenure % 20)
        ELSE 25 + (tenure % 35)
    END AS age,
    -- Distribute customers across 4 regions
    CASE ABS(CHECKSUM(NEWID())) % 4
        WHEN 0 THEN 'North'
        WHEN 1 THEN 'South'
        WHEN 2 THEN 'East'
        WHEN 3 THEN 'West'
    END AS region,
    DATEADD(month, -tenure, '2024-01-01'),
    Contract
FROM stg_telco_raw;
GO

PRINT '--- Inserting data into billing table ---';
GO

INSERT INTO billing (
    customer_id, monthly_charges, total_charges, payment_method,
    late_payments, billing_date, paperless_billing
)
SELECT
    customerID,
    MonthlyCharges,
    CAST(NULLIF(TotalCharges, '') AS DECIMAL(12,2)),
    PaymentMethod,
    -- Late payment logic: check payments and month-to-month are riskier
    CASE
        WHEN PaymentMethod LIKE '%check' THEN (tenure % 3)
        WHEN Contract = 'Month-to-month' THEN (tenure % 2)
        ELSE 0
    END AS late_payments,
    DATEADD(month, -1, GETDATE()),
    PaperlessBilling
FROM stg_telco_raw;
GO

PRINT '--- Inserting data into usage_data table ---';
GO

INSERT INTO usage_data (
    customer_id, avg_call_minutes, avg_data_usage_gb,
    internet_service, phone_service, multiple_lines, usage_month
)
SELECT
    customerID,
    -- Call minutes: more with multiple lines
    CASE
        WHEN PhoneService = 'Yes' AND MultipleLines = 'Yes'
            THEN 300 + (tenure % 200) + (ABS(CHECKSUM(NEWID())) % 100)
        WHEN PhoneService = 'Yes'
            THEN 150 + (tenure % 150) + (ABS(CHECKSUM(NEWID())) % 100)
        ELSE 0
    END AS avg_call_minutes,
    -- Data usage: fiber optic users consume more
    CASE
        WHEN InternetService = 'Fiber optic' THEN 25 + (tenure % 25) + (ABS(CHECKSUM(NEWID())) % 20)
        WHEN InternetService = 'DSL' THEN 15 + (tenure % 20) + (ABS(CHECKSUM(NEWID())) % 15)
        ELSE 0
    END AS avg_data_usage_gb,
    InternetService,
    PhoneService,
    MultipleLines,
    DATEADD(month, -1, GETDATE())
FROM stg_telco_raw;
GO

PRINT '--- Inserting data into support_tickets table with REALISTIC distribution ---';
PRINT '     Churned: 1-7 tickets (skewed right)';
PRINT '     Non-churned: 0-3 tickets (mostly 0)';
GO

INSERT INTO support_tickets (
    customer_id, ticket_type, resolution_time_hr,
    satisfaction_score, ticket_date
)
SELECT
    r.customerID,
    CASE ABS(CHECKSUM(NEWID())) % 4
        WHEN 0 THEN 'Billing Issue'
        WHEN 1 THEN 'Technical Issue'
        WHEN 2 THEN 'Service Complaint'
        WHEN 3 THEN 'Account Management'
    END AS ticket_type,
    CASE
        WHEN r.TechSupport = 'Yes' THEN 2 + (ABS(CHECKSUM(NEWID())) % 24)
        ELSE 24 + (ABS(CHECKSUM(NEWID())) % 48)
    END AS resolution_time_hr,
    CASE
        WHEN r.TechSupport = 'Yes' THEN 4 + (ABS(CHECKSUM(NEWID())) % 2)
        ELSE 1 + (ABS(CHECKSUM(NEWID())) % 3)
    END AS satisfaction_score,
    DATEADD(day, -r.tenure % 30, GETDATE())
FROM stg_telco_raw r
CROSS APPLY (
    SELECT CAST(RAND(CHECKSUM(NEWID())) AS FLOAT) AS rand_val
) rv
CROSS APPLY (
    SELECT TOP (
        CASE 
            WHEN r.Churn = 'Yes' THEN 
                CASE 
                    WHEN rv.rand_val < 0.10 THEN 1   -- 10% get 1 ticket
                    WHEN rv.rand_val < 0.25 THEN 2   -- 15% get 2 tickets
                    WHEN rv.rand_val < 0.45 THEN 3   -- 20% get 3 tickets
                    WHEN rv.rand_val < 0.65 THEN 4   -- 20% get 4 tickets
                    WHEN rv.rand_val < 0.80 THEN 5   -- 15% get 5 tickets
                    WHEN rv.rand_val < 0.90 THEN 6   -- 10% get 6 tickets
                    ELSE 7                           -- 10% get 7 tickets
                END
            ELSE  -- Non-churn customers
                CASE 
                    WHEN rv.rand_val < 0.70 THEN 0   -- 70% get 0 tickets
                    WHEN rv.rand_val < 0.85 THEN 1   -- 15% get 1 ticket
                    WHEN rv.rand_val < 0.95 THEN 2   -- 10% get 2 tickets
                    ELSE 3                           -- 5% get 3 tickets
                END
        END
    ) 1 AS dummy
    FROM sys.objects
) x;
GO

PRINT '--- Inserting data into churn_labels table ---';
GO

INSERT INTO churn_labels (customer_id, churn, churn_date)
SELECT
    customerID,
    CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END,
    CASE WHEN Churn = 'Yes'
        THEN DATEADD(day, -(ABS(CHECKSUM(NEWID())) % 30), GETDATE())
        ELSE NULL
    END
FROM stg_telco_raw;
GO

/* -------------------------------------------------------------------------- */
/* 6. AGGREGATION VIEWS - Pre-aggregated data for ML                         */
/* -------------------------------------------------------------------------- */
PRINT '--- Creating aggregation views ---';
GO

-- Support tickets aggregated by customer
CREATE VIEW vw_support_agg AS
SELECT
    customer_id,
    COUNT(*) AS support_ticket_count,
    AVG(resolution_time_hr) AS avg_resolution_time,
    AVG(satisfaction_score) AS avg_satisfaction_score,
    MAX(ticket_date) AS last_ticket_date,
    DATEDIFF(day, MAX(ticket_date), GETDATE()) AS days_since_last_ticket
FROM support_tickets
GROUP BY customer_id;
GO

-- Usage data aggregated by customer
CREATE VIEW vw_usage_agg AS
SELECT
    customer_id,
    AVG(avg_call_minutes) AS avg_call_minutes,
    AVG(avg_data_usage_gb) AS avg_data_usage_gb,
    MAX(internet_service) AS internet_service,
    MAX(phone_service) AS phone_service,
    MAX(multiple_lines) AS multiple_lines
FROM usage_data
GROUP BY customer_id;
GO

/* -------------------------------------------------------------------------- */
/* 7. TRAINING VIEW - Complete feature set for model training                */
/* -------------------------------------------------------------------------- */

CREATE VIEW vw_churn_training_features AS
SELECT
    -- Primary key
    c.customer_id,

    /* =========================
       LIFECYCLE FEATURES
       ========================= */
    DATEDIFF(month, c.signup_date, GETDATE()) AS tenure_months,

    /* =========================
       CONTRACT & COMMITMENT
       ========================= */
    c.contract_type,

    /* =========================
       BILLING & PAYMENT
       ========================= */
    b.monthly_charges,
    b.total_charges,
    b.late_payments,
    b.payment_method,

    /* =========================
       USAGE & ENGAGEMENT
       ========================= */
    ISNULL(u.avg_call_minutes, 0) AS avg_call_minutes,
    ISNULL(u.avg_data_usage_gb, 0) AS avg_data_usage_gb,

    /* =========================
       CUSTOMER EXPERIENCE
       ========================= */
    ISNULL(s.support_ticket_count, 0) AS support_ticket_count,
    ISNULL(s.avg_resolution_time, 0) AS avg_resolution_time,
    ISNULL(s.avg_satisfaction_score, 0) AS avg_satisfaction_score,

    /* =========================
       PRODUCT STICKINESS
       ========================= */
    CASE WHEN r.OnlineSecurity = 'Yes' THEN 1 ELSE 0 END AS has_online_security,
    CASE WHEN r.TechSupport = 'Yes' THEN 1 ELSE 0 END AS has_tech_support,
    (
        CASE WHEN r.StreamingTV = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN r.StreamingMovies = 'Yes' THEN 1 ELSE 0 END
    ) AS streaming_services_count,

    /* =========================
       TARGET VARIABLE
       ========================= */
    ch.churn

FROM customers c

JOIN billing b
    ON c.customer_id = b.customer_id

LEFT JOIN vw_usage_agg u
    ON c.customer_id = u.customer_id

LEFT JOIN vw_support_agg s
    ON c.customer_id = s.customer_id

JOIN churn_labels ch
    ON c.customer_id = ch.customer_id

JOIN stg_telco_raw r
    ON c.customer_id = r.customerID;
GO

/* -------------------------------------------------------------------------- */
/* 8. DEPLOYMENT VIEW - Only top 7 features for production                   */
/* -------------------------------------------------------------------------- */
PRINT '--- Creating deployment view with 7 features ---';
GO

CREATE VIEW vw_churn_deployment_features AS
SELECT
    customer_id,
    tenure_months,
    contract_type,
    monthly_charges,
    payment_method,
    support_ticket_count,
    avg_call_minutes,
    avg_data_usage_gb,
    churn
FROM vw_churn_training_features;
GO

/* -------------------------------------------------------------------------- */
/* 9. VERIFICATION QUERIES - Validate data quality                           */
/* -------------------------------------------------------------------------- */
PRINT '--- VERIFICATION: Support ticket distribution for CHURNED customers ---';
PRINT '     Expected: 1-7 tickets with 10-20% each';
GO

-- Check distribution for churned customers
SELECT 
    support_ticket_count,
    COUNT(*) as customer_count,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS DECIMAL(5,2)) AS percentage
FROM vw_churn_training_features
WHERE churn = 1
GROUP BY support_ticket_count
ORDER BY support_ticket_count;
GO

PRINT '--- VERIFICATION: Support ticket distribution for NON-CHURNED customers ---';
PRINT '     Expected: 0 tickets (70%), 1 ticket (15%), 2 tickets (10%), 3 tickets (5%)';
GO

-- Check distribution for non-churned customers
SELECT 
    support_ticket_count,
    COUNT(*) as customer_count,
    CAST(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS DECIMAL(5,2)) AS percentage
FROM vw_churn_training_features
WHERE churn = 0
GROUP BY support_ticket_count
ORDER BY support_ticket_count;
GO

PRINT '--- VERIFICATION: Sample of 7 deployment features for churned customers ---';
GO

-- Check deployment view
SELECT TOP 20 *
FROM vw_churn_deployment_features
WHERE churn = 1
ORDER BY support_ticket_count DESC;
GO

PRINT '================================================';
PRINT 'CustomerChurnDB pipeline execution COMPLETE';

DECLARE @CustomerCount INT;
DECLARE @ChurnRate DECIMAL(5,2);

SELECT @CustomerCount = COUNT(*) FROM customers;
SELECT @ChurnRate = AVG(CAST(churn AS FLOAT)) * 100 FROM churn_labels;

PRINT '   Total customers: ' + CAST(@CustomerCount AS VARCHAR);
PRINT '   Churn rate: ' + CAST(@ChurnRate AS VARCHAR) + '%';
PRINT '================================================';
GO