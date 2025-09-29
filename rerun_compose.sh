#!/bin/bash
docker container stop screwdriver 
docker-compose stop && docker-compose rm
docker-compose up -d --force-recreate --remove-orphans 
docker-compose logs -f