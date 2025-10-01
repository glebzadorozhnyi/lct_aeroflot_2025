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
docker container stop screwdriver 
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

```




## Deploy 'screwdriver' as bazel label
Установите bazel, например как бинрный файл ( по https://bazel.build/install/ubuntu#binary-installer)
```sh
pushd /tmp;
wget https://github.com/bazelbuild/bazel/releases/download/8.4.2rc2/bazel_8.4.2rc2-linux-x86_64.deb
sudo apt install ./bazel_8.4.2rc2-linux-x86_64.deb 
popd;
```
Очистка
```sh 
rm -r "$(bazel info repository_cache)"
 bazel clean --async --expunge
```