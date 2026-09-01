package database

import (
	"database/sql"
	"fmt"
	"os"

	_ "github.com/lib/pq"
)

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

    return db, nil
}

