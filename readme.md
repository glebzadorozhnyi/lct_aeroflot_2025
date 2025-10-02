# Requirements

## Software
- ubuntu 22/24
- docker
- docker-compose-plugin(recommended)
- Nvidia GPU [optional]
## Hardware
- 2 GB RAM
- 2 CPU
- 100 GB SSH/HDD
- CUDA GPU Compute Capability [optional]
#  Первый запуск
## Получения исходников 🪛'ки
```sh
git clone https://github.com/glebzadorozhnyi/lct_aeroflot_2025.git
```
Перейти в каталог проекта:
```sh
cd ./lct_aeroflot_2025
```
Дальнейшие действия продолжать в директории `lct_aeroflot_2025`.
## Сборка образа 🪛'ки
```shell
docker-compose build
```
##  Запуск web-приложения
```sh
docker-compose up -d --force-recreate --remove-orphans 
docker-compose logs -f
```
Далее зайти в UI: http://0.0.0.0:8000 через браузер.
## Просмотр базы данных
Установка  sqlitebrowser для просмотрп db
```sh
sudo apt-get install sqlitebrowser
```
Просмотр данных
```sh
sqlitebrowser .workdir/sql_app.db

```
## Просмотр базы данных
Установка  sqlitebrowser для просмотрп db
```sh
sudo apt-get install sqlitebrowser
```
Просмотр данных
```sh
sqlitebrowser .workdir/sql_app.db
```
## Описание REST API интерфейса
http://0.0.0.0:8000/docs

## Перезапуск
```sh 
make rerun
```
# Demo 🪛
http://main.screwdriver-and-co.ru