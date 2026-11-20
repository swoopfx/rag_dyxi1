<#
.SYNOPSIS
    Enterprise Agentic RAG: PowerShell Database & Table Setup Script
.DESCRIPTION
    Creates the necessary MySQL database (rag_metadata) and all lookup tables
    (rag_document_metadata, rag_extracted_tables, rag_extracted_images, rag_extracted_formulas)
    defined in init_schema.sql.
.PARAMETER Host
    MySQL Server host (default: from .env or localhost)
.PARAMETER Port
    MySQL Server port (default: from .env or 3306)
.PARAMETER User
    MySQL Username (default: from .env or root)
.PARAMETER Password
    MySQL Password (default: from .env or empty)
.PARAMETER Database
    MySQL Database name (default: from .env or rag_metadata)
.PARAMETER File
    Path to SQL schema file (default: scripts\init_schema.sql)
.EXAMPLE
    .\scripts\setup_database.ps1
.EXAMPLE
    .\scripts\setup_database.ps1 -User root -Password mypass -Host localhost
#>

[CmdletBinding()]
param (
    [string]$HostName = "",
    [int]$Port = 0,
    [string]$User = "",
    [string]$Password = "",
    [string]$Database = "",
    [string]$File = ""
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$EnvFile = Join-Path $ProjectRoot ".env"
$DefaultSqlFile = Join-Path $ScriptDir "init_schema.sql"

# Read variable from .env
function Get-EnvValue([string]$key, [string]$defaultVal) {
    if (Test-Path $EnvFile) {
        $lines = Get-Content $EnvFile
        foreach ($line in $lines) {
            if ($line -match "^\s*$key\s*=\s*(.*)$") {
                $val = $matches[1].Trim()
                $val = $val.Trim('"', "'")
                return $val
            }
        }
    }
    return $defaultVal
}

if (-not $HostName) { $HostName = Get-EnvValue "MYSQL_HOST" "localhost" }
if ($Port -eq 0) { $Port = [int](Get-EnvValue "MYSQL_PORT" "3306") }
if (-not $User) { $User = Get-EnvValue "MYSQL_USER" "root" }
if (-not $Password) { $Password = Get-EnvValue "MYSQL_PASSWORD" "" }
if (-not $Database) { $Database = Get-EnvValue "MYSQL_DATABASE" "rag_metadata" }
if (-not $File) { $File = $DefaultSqlFile }

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host " Enterprise Agentic RAG: Relational Database & Table Setup" -ForegroundColor White
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  Host:         $HostName`:$Port"
Write-Host "  User:         $User"
Write-Host "  Database:     $Database"
Write-Host "  SQL Schema:   $File"
Write-Host "------------------------------------------------------------------------" -ForegroundColor Cyan

if (-not (Test-Path $File)) {
    Write-Error "[ERROR] Schema file not found at: $File"
    exit 1
}

# Find Python binary in .venv or system
$PythonBin = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonBin)) {
    $PythonBin = "python"
}

$InitDbScript = Join-Path $ScriptDir "init_db.py"

& $PythonBin $InitDbScript -H $HostName -P $Port -u $User -p $Password -d $Database -f $File
if ($LASTEXITCODE -ne 0) {
    Write-Error "[ERROR] Database setup failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
