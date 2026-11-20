"""
Validation test for database initialization scripts and schema.
"""

from pathlib import Path
import re
import pytest

def test_init_schema_sql():
    sql_file = Path("scripts/init_schema.sql")
    assert sql_file.exists(), "scripts/init_schema.sql does not exist"
    
    content = sql_file.read_text(encoding="utf-8")
    
    # Check Database creation
    assert "CREATE DATABASE IF NOT EXISTS `rag_metadata`" in content
    assert "USE `rag_metadata`;" in content
    
    # Check all four tables
    expected_tables = [
        "rag_document_metadata",
        "rag_extracted_tables",
        "rag_extracted_images",
        "rag_extracted_formulas"
    ]
    for table in expected_tables:
        pattern = rf"CREATE TABLE IF NOT EXISTS `{table}`"
        assert re.search(pattern, content), f"Table {table} not found in schema"
        
    # Check critical columns
    assert "`description` LONGTEXT NULL" in content, "Missing description column in rag_extracted_images"
    assert "`formula_id` VARCHAR(128) NOT NULL" in content
    assert "`table_id` VARCHAR(128) NOT NULL" in content
    assert "`image_id` VARCHAR(128) NOT NULL" in content
    assert "`document_id` VARCHAR(128) NOT NULL" in content

def test_script_files_exist():
    assert Path("scripts/setup_database.sh").exists()
    assert Path("scripts/setup_database.ps1").exists()
    assert Path("scripts/init_db.py").exists()

if __name__ == "__main__":
    test_init_schema_sql()
    test_script_files_exist()
    print("All schema and script tests PASSED!")
