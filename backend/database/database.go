package database

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
)

import _ "github.com/lib/pq"

func Connect() (*sql.DB, error) {
	host := os.Getenv("POSTGRES_HOST")
	port := os.Getenv("POSTGRES_PORT")
	user := os.Getenv("POSTGRES_USER")
	password := os.Getenv("POSTGRES_PASSWORD")
	dbname := os.Getenv("POSTGRES_DB")

	psqlInfo := fmt.Sprintf(
		"host=%s port=%s user=%s password=%s dbname=%s sslmode=disable",
		host,
		port,
		user,
		password,
		dbname,
	)

	db, err := sql.Open("postgres", psqlInfo)
	if err != nil {
		return nil, err
	}

	if err := db.Ping(); err != nil {
		db.Close()
		return nil, err
	}

	if err := initializeSchema(db); err != nil {
		db.Close()
		return nil, err
	}

	return db, nil
}

func initializeSchema(db *sql.DB) error {
	schemaPath := filepath.Join(
		"database",
		"schema.sql",
	)

	schema, err := os.ReadFile(schemaPath)
	if err != nil {
		return fmt.Errorf(
			"read database schema: %w",
			err,
		)
	}

	if _, err := db.Exec(string(schema)); err != nil {
		return fmt.Errorf(
			"initialize database schema: %w",
			err,
		)
	}

	return nil
}
