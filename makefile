.PHONY: run init-run clear
rerun: clear
	docker-compose up -d --force-recreate --remove-orphans 
	docker-compose logs -f

init-run:
	docker-compose build
	docker-compose up -d --force-recreate --remove-orphans 
	docker-compose logs -f

clear:
	docker container stop screwdriver | true
	docker container rm screwdriver | true
	docker-compose stop | true 
	docker-compose rm | true
