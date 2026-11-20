-- ==============================================================================
-- Enterprise Agentic RAG: Relational Database Initialization DDL
-- Target Engine: MySQL 8.0+ / MariaDB 10.5+
-- Character Set: utf8mb4 (Unicode 5.2.0+)
-- Collation: utf8mb4_unicode_ci
-- ==============================================================================

-- 1. Create Database if it does not exist
CREATE DATABASE IF NOT EXISTS `rag_metadata`
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE `rag_metadata`;

-- ------------------------------------------------------------------------------
-- Table 1: rag_document_metadata
-- Root registry for ingested multimodal documents and global element counts
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `rag_document_metadata` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `document_id` VARCHAR(128) NOT NULL,
    `file_path` VARCHAR(1024) NOT NULL,
    `file_name` VARCHAR(255) NOT NULL,
    `file_extension` VARCHAR(32) NOT NULL,
    `file_size_bytes` BIGINT NULL,
    `mime_type` VARCHAR(128) NULL,
    `num_pages` INT DEFAULT 1,
    `tables_count` INT DEFAULT 0,
    `images_count` INT DEFAULT 0,
    `formulas_count` INT DEFAULT 0,
    `extra_metadata_json` LONGTEXT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY `uk_doc_id` (`document_id`),
    INDEX `idx_doc_file_name` (`file_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------------------------
-- Table 2: rag_extracted_tables
-- Lookup store for complete, intact structured tables indexed by table_id
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `rag_extracted_tables` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `table_id` VARCHAR(128) NOT NULL,
    `document_id` VARCHAR(128) NOT NULL,
    `table_index` INT NOT NULL,
    `page_no` INT NULL,
    `caption` TEXT NULL,
    `summary` TEXT NULL,
    `markdown` LONGTEXT NOT NULL,
    `html` LONGTEXT NULL,
    `data_rows_json` LONGTEXT NULL,
    `num_rows` INT DEFAULT 0,
    `num_cols` INT DEFAULT 0,
    `bbox_json` TEXT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY `uk_table_id` (`table_id`),
    INDEX `idx_table_doc_id` (`document_id`),
    INDEX `idx_table_page_no` (`document_id`, `page_no`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------------------------
-- Table 3: rag_extracted_images
-- Lookup store for visual elements, diagrams, base64 data, and AI vision summaries
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `rag_extracted_images` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `image_id` VARCHAR(128) NOT NULL,
    `document_id` VARCHAR(128) NOT NULL,
    `image_index` INT NOT NULL,
    `page_no` INT NULL,
    `caption` TEXT NULL,
    `description` LONGTEXT NULL,
    `summary` TEXT NULL,
    `label` VARCHAR(64) NOT NULL DEFAULT 'picture',
    `width` INT NULL,
    `height` INT NULL,
    `format` VARCHAR(32) NOT NULL DEFAULT 'PNG',
    `saved_path` VARCHAR(1024) NULL,
    `base64_uri` LONGTEXT NULL,
    `bbox_json` TEXT NULL,
    `annotations_json` TEXT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY `uk_image_id` (`image_id`),
    INDEX `idx_img_doc_id` (`document_id`),
    INDEX `idx_img_page_no` (`document_id`, `page_no`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------------------------
-- Table 4: rag_extracted_formulas
-- Lookup store for raw intact LaTeX-formatted mathematical equations
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `rag_extracted_formulas` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `formula_id` VARCHAR(128) NOT NULL,
    `document_id` VARCHAR(128) NOT NULL,
    `formula_index` INT NOT NULL,
    `page_no` INT NULL,
    `latex` LONGTEXT NOT NULL,
    `context` TEXT NULL,
    `summary` TEXT NULL,
    `bbox_json` TEXT NULL,
    `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY `uk_formula_id` (`formula_id`),
    INDEX `idx_formula_doc_id` (`document_id`),
    INDEX `idx_formula_page_no` (`document_id`, `page_no`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
