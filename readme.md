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
# or 
ocker container rm screwdriver
```
Deploy web app
```shell
docker-compose up -d --force-recreate --remove-orphans 
#Optional step: check logs
docker-compose logs -f
#Optional step: attach console 
docker-compose exec -it screwdriver bash
docker exec -it b87440e6d810 bash
```

```sh
npx -y @diplodoc/cli@next -i ./docs -o ./.workdir/docs --config .yfw  --output-format html 
```


# Просмотр базы данных
Установка  sqlitebrowser для просмотрп db
```sh
sudo apt-get install sqlitebrowser
```

Просмотр данных
```sh
sqlitebrowser .workdir/sql_app.db

````