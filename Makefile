.PHONY: up build down logs

up:
	docker compose up --build

build:
	docker compose build

down:
	docker compose down

logs:
	docker compose logs -f
