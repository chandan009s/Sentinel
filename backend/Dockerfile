FROM golang:1.26

WORKDIR /app

COPY backend/go.mod backend/go.sum ./

RUN go mod download

COPY backend/ ./

COPY database/schema.sql /app/database/schema.sql

RUN mkdir -p /app/ml/models

COPY ml/models/isolation_forest_v1_meta.json \
    /app/ml/models/isolation_forest_v1_meta.json

RUN go build -o server .

EXPOSE 8080

CMD ["./server"]