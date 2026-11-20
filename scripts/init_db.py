"""
Universal Database Initialization Helper for Enterprise Agentic RAG.
Reads environment variables from .env or CLI parameters and executes
the init_schema.sql DDL script against MySQL via PyMySQL.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Try importing pymysql
try:
    import pymysql
    import pymysql.constants.CLIENT
except ImportError:
    print("[ERROR] PyMySQL is not installed. Run: pip install pymysql")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Initialize MySQL Database and Tables for Multimodal RAG."
    )
    parser.add_argument("-H", "--host", default=None, help="MySQL Host (default from .env or localhost)")
    parser.add_argument("-P", "--port", type=int, default=None, help="MySQL Port (default from .env or 3306)")
    parser.add_argument("-u", "--user", default=None, help="MySQL User (default from .env or root)")
    parser.add_argument("-p", "--password", default=None, help="MySQL Password (default from .env)")
    parser.add_argument("-d", "--database", default=None, help="MySQL Database name (default from .env or rag_metadata)")
    parser.add_argument("-f", "--file", default=None, help="Path to SQL file (default scripts/init_schema.sql)")
    return parser.parse_args()


def load_config(args):
    # Search for .env in current and parent directory
    base_dir = Path(__file__).resolve().parent.parent
    env_path = base_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    host = args.host or os.getenv("MYSQL_HOST", "localhost")
    port = args.port or int(os.getenv("MYSQL_PORT", "3306"))
    user = args.user or os.getenv("MYSQL_USER", "root")
    password = args.password if args.password is not None else os.getenv("MYSQL_PASSWORD", "")
    database = args.database or os.getenv("MYSQL_DATABASE", "rag_metadata")
    
    sql_file = Path(args.file) if args.file else Path(__file__).resolve().parent / "init_schema.sql"

    return host, port, user, password, database, sql_file


def init_database():
    args = parse_args()
    host, port, user, password, database, sql_file = load_config(args)

    if not sql_file.exists():
        print(f"[ERROR] SQL schema file not found at: {sql_file}")
        sys.exit(1)

    print("=" * 70)
    print(" Enterprise Agentic RAG: Relational Database Initialization")
    print("=" * 70)
    print(f" Target Host:     {host}:{port}")
    print(f" Target User:     {user}")
    print(f" Target Database: {database}")
    print(f" SQL File:        {sql_file}")
    print("-" * 70)

    try:
        # Step 1: Connect to MySQL server (without specifying DB so we can CREATE DATABASE)
        print(f"[*] Connecting to MySQL server at {host}:{port}...")
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            charset="utf8mb4",
            client_flag=pymysql.constants.CLIENT.MULTI_STATEMENTS,
            autocommit=True
        )
        print("[+] Server connection established successfully.")

        # Step 2: Read and execute SQL statements
        print(f"[*] Reading schema script: {sql_file.name}...")
        with open(sql_file, "r", encoding="utf-8") as f:
            sql_script = f.read()

        print("[*] Executing database and table creation DDL...")
        with conn.cursor() as cursor:
            cursor.execute(sql_script)
        print("[+] Schema initialization statements executed successfully.")

        # Step 3: Verify created database and tables
        with conn.cursor() as cursor:
            cursor.execute(f"USE `{database}`;")
            cursor.execute("SHOW TABLES;")
            tables = [row[0] for row in cursor.fetchall()]

        print("-" * 70)
        print(f"[SUCCESS] Database `{database}` initialized successfully.")
        print(f"[+] Verified Tables in `{database}` ({len(tables)} tables):")
        for table in tables:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM `{table}`;")
                count = cursor.fetchone()[0]
            print(f"    - {table:<30} [Records: {count}]")
        print("=" * 70)
        conn.close()

    except pymysql.MySQLError as err:
        print(f"\n[ERROR] MySQL Error encountered: {err}")
        print("\nPlease check:")
        print(" 1. Is your MySQL daemon running on the specified host & port?")
        print(" 2. Are the credentials (user/password) valid?")
        print(" 3. Does the user have permission to CREATE DATABASE / CREATE TABLE?")
        sys.exit(1)
    except Exception as ex:
        print(f"\n[ERROR] Unexpected error: {ex}")
        sys.exit(1)


if __name__ == "__main__":
    init_database()
