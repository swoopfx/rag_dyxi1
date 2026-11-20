#!/usr/bin/env bash
# ==============================================================================
# Enterprise Agentic RAG: Database & Table Setup Script
# Usage:
#   ./scripts/setup_database.sh [OPTIONS]
# Options:
#   -h, --host      MySQL host (default from .env or localhost)
#   -P, --port      MySQL port (default from .env or 3306)
#   -u, --user      MySQL user (default from .env or root)
#   -p, --password  MySQL password (default from .env, or prompted if empty)
#   -d, --database  MySQL database name (default from .env or rag_metadata)
#   -f, --file      Path to SQL schema file (default: scripts/init_schema.sql)
#   --help          Display this help message
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
SQL_FILE="${SCRIPT_DIR}/init_schema.sql"

# ANSI Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Helper to read .env variable without export side effects
read_env_var() {
    local var_name="$1"
    local default_val="$2"
    if [ -f "${ENV_FILE}" ]; then
        local line
        line=$(grep -E "^[[:space:]]*${var_name}[[:space:]]*=" "${ENV_FILE}" | tail -n 1 || true)
        if [ -n "${line}" ]; then
            local val
            val=$(echo "${line}" | sed -E "s/^[[:space:]]*${var_name}[[:space:]]*=[[:space:]]*//" | sed -E 's/^["'"'"']//;s/["'"'"']$//')
            echo "${val}"
            return
        fi
    fi
    echo "${default_val}"
}

# Load Defaults from .env or fallback
DB_HOST=$(read_env_var "MYSQL_HOST" "localhost")
DB_PORT=$(read_env_var "MYSQL_PORT" "3306")
DB_USER=$(read_env_var "MYSQL_USER" "root")
DB_PASS=$(read_env_var "MYSQL_PASSWORD" "")
DB_NAME=$(read_env_var "MYSQL_DATABASE" "rag_metadata")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--host)
            DB_HOST="$2"
            shift 2
            ;;
        -P|--port)
            DB_PORT="$2"
            shift 2
            ;;
        -u|--user)
            DB_USER="$2"
            shift 2
            ;;
        -p|--password)
            if [[ $# -gt 1 && "$2" != -* ]]; then
                DB_PASS="$2"
                shift 2
            else
                read -rsp "Enter MySQL Password for ${DB_USER}: " DB_PASS
                echo ""
                shift 1
            fi
            ;;
        -d|--database)
            DB_NAME="$2"
            shift 2
            ;;
        -f|--file)
            SQL_FILE="$2"
            shift 2
            ;;
        --help)
            echo -e "${BOLD}Enterprise Agentic RAG: Database Initialization Script${NC}"
            echo ""
            echo "Usage: ./scripts/setup_database.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -h, --host <host>         MySQL server host (default: ${DB_HOST})"
            echo "  -P, --port <port>         MySQL server port (default: ${DB_PORT})"
            echo "  -u, --user <user>         MySQL database username (default: ${DB_USER})"
            echo "  -p, --password [password] MySQL database password (reads .env or prompts)"
            echo "  -d, --database <name>     Target database name (default: ${DB_NAME})"
            echo "  -f, --file <path>         Path to SQL DDL schema file"
            echo "  --help                    Show this help message and exit"
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR] Unknown option: $1${NC}"
            echo "Use --help for usage details."
            exit 1
            ;;
    esac
done

echo -e "${CYAN}========================================================================${NC}"
echo -e "${BOLD} Enterprise Agentic RAG: Relational Database & Table Setup${NC}"
echo -e "${CYAN}========================================================================${NC}"
echo -e "  Host:         ${BOLD}${DB_HOST}:${DB_PORT}${NC}"
echo -e "  User:         ${BOLD}${DB_USER}${NC}"
echo -e "  Database:     ${BOLD}${DB_NAME}${NC}"
echo -e "  SQL Schema:   ${BOLD}${SQL_FILE}${NC}"
echo -e "${CYAN}------------------------------------------------------------------------${NC}"

# Check if SQL file exists
if [ ! -f "${SQL_FILE}" ]; then
    echo -e "${RED}[ERROR] Schema file not found at: ${SQL_FILE}${NC}"
    exit 1
fi

# Determine execution strategy (native mysql CLI vs Python PyMySQL)
if command -v mysql >/dev/null 2>&1; then
    echo -e "${BLUE}[*] Found native 'mysql' CLI client. Executing schema...${NC}"
    
    MYSQL_CMD="mysql -h${DB_HOST} -P${DB_PORT} -u${DB_USER}"
    if [ -n "${DB_PASS}" ]; then
        MYSQL_CMD="${MYSQL_CMD} -p${DB_PASS}"
    fi

    # Execute DDL
    if eval "${MYSQL_CMD} < \"${SQL_FILE}\""; then
        echo -e "${GREEN}[+] SQL statements executed successfully via mysql CLI.${NC}"
    else
        echo -e "${RED}[ERROR] Execution failed with mysql CLI.${NC}"
        exit 1
    fi

    # Verify tables
    echo -e "${BLUE}[*] Verifying created tables in '${DB_NAME}'...${NC}"
    TABLES_QUERY="SELECT table_name, table_rows FROM information_schema.tables WHERE table_schema='${DB_NAME}';"
    eval "${MYSQL_CMD} -e \"${TABLES_QUERY}\""

    echo -e "${GREEN}========================================================================${NC}"
    echo -e "${BOLD}${GREEN}[SUCCESS] Database '${DB_NAME}' and tables initialized successfully!${NC}"
    echo -e "${GREEN}========================================================================${NC}"
    exit 0

else
    echo -e "${YELLOW}[!] 'mysql' CLI client not found in PATH.${NC}"
    echo -e "${BLUE}[*] Falling back to Python PyMySQL runner...${NC}"

    PYTHON_BIN=""
    if [ -f "${PROJECT_ROOT}/.venv/bin/python" ]; then
        PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
    elif [ -f "${PROJECT_ROOT}/.venv/Scripts/python.exe" ]; then
        PYTHON_BIN="${PROJECT_ROOT}/.venv/Scripts/python.exe"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo -e "${RED}[ERROR] Neither 'mysql' CLI nor a Python interpreter was found.${NC}"
        echo "Please install MySQL client or run in the Python virtual environment (.venv)."
        exit 1
    fi

    "${PYTHON_BIN}" "${SCRIPT_DIR}/init_db.py" \
        -H "${DB_HOST}" \
        -P "${DB_PORT}" \
        -u "${DB_USER}" \
        -p "${DB_PASS}" \
        -d "${DB_NAME}" \
        -f "${SQL_FILE}"

    exit $?
fi
