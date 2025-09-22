## Requirements

- ubuntu 22/24
- docker
- docker-compose-plugin(recommended)
- Nvidia GPU

## Deploy 'screwdriver' as docker-compose app
### Build image (can be skipped)
```shell
docker-compose build
```
### Run app
- [optional] Clean last deploy
```sh
docker-compose stop && docker-compose rm
```
Deploy web app
```shell
docker-compose up -d --force-recreate
#Optional step: check logs
docker compose logs
#Optional step: attach console 
docker-compose exec screwdriver bash
```