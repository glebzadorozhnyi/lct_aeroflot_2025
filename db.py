from sqlalchemy.orm import DeclarativeBase, sessionmaker, Mapped
from sqlalchemy import Boolean, Column, Integer, String, create_engine
from constants import TOOL_CLASSES

SQLALCHEMY_DATABASE_URL = "sqlite:///.workdir/sql_app.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})


class Base(DeclarativeBase):
    pass


# class AeroTool(Base):
#     __tablename__ = "aerotool_set"

#     id = Column(Integer, primary_key=True, index=True)
#     name = Column(String)
#     delivery_state = Column(String)  # "in_stock", "on_hands"
#     type = Column(
#         String,
#     )
#     delivery_id = Column(Integer,)  # Наборов для выдачи может быть много и каждый инструмент принадлежит к одной из выдач
#     detect_state = Column(
#         Boolean,
#     )

# JSONList = ["test1", "test2"]

class AeroToolDelivery(Base):
    __tablename__ = "aerotool_set_delivery"

    id = Column(Integer, primary_key=True, index=True)
    image_file_id = Column(
        String,
    )
    # my_list_of_strings:Mapped[list[str]] = mapped_column(JSONList)
    founded_screw_flat = Column(Integer,)
    founded_screw_plus = Column(Integer,)
    founded_offset_plus_screw = Column(Integer,)
    founded_kolovorot = Column(Integer,)
    founded_safety_pliers = Column(Integer,)
    founded_pliers = Column(Integer,)
    founded_shernitsa = Column(Integer,)
    founded_adjustable_wrench = Column(Integer,)
    founded_can_opener = Column(Integer,)
    founded_open_end_wrench = Column(Integer,)
    founded_side_cutters = Column(Integer,)






    datatime = Column(
        String,
    )

    # image_lable = Column(String)


    # delivery_state = Column(String)  # "in_stock", "on_hands"
    # type = Column(
    #     String,
    # )
    # delivery_id = Column(
    #     Integer,
    # )  # Наборов для выдачи может быть много и каждый инструмент принадлежит к одной из выдач
    # detect_state = Column(
    #     Boolean,
    # )



# создаем таблицы
Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(autoflush=False, bind=engine)
db = SessionLocal()


def check_exist_tool_in_db_by_name(name) -> bool:
    first = db.query(AeroToolDelivery).filter(AeroToolDelivery.image_file_id == name).first()

    if first:
        # print(
        #     f"Find existing item with name {first.name}: id {first.id} - ({first.type}) - {first.delivery_id}"
        # )
        return True
    else:
        return False


# def fill_test_data():
#     deliveries = [1, 2, 3]

#     for i_delevery in deliveries:
#         for i_tool_type in TOOL_CLASSES:
#             new_tool_unique_name = f"{i_tool_type}_{i_delevery}"
#             if not check_exist_tool_in_db_by_name(new_tool_unique_name):
#                 screw_plus = AeroTool(
#                     name=new_tool_unique_name,
#                     type=i_tool_type,
#                     delivery_id=i_delevery,
#                     delivery_state="on_hands",
#                 )
#                 db.add(screw_plus)  # добавляем в бд
#     db.commit()  # сохраняем изменения

# def add_delivery_set(hash_id, image_file_id, datatime):
#     delivery_set = AeroToolDelivery(
#                     id=hash_id,
#                     image_file_id=image_file_id,
#                     # type=i_tool_type,
#                     # delivery_id=i_delevery,
#                     datatime=datatime,
#                 )
#     db.add(delivery_set)  # добавляем в бд
#     db.commit()


def print_all_from_db():
    # получение всех объектов
    aero_tools = db.query(AeroToolDelivery).all()
    print("| id|           type        |         name         |")
    for i_aero_tool in aero_tools:
        print(f"|{i_aero_tool.id:3}| {i_aero_tool.datatime:22}| {i_aero_tool.image_file_id:20} |")


# def get_all_uploaded_files():
#     # получение всех объектов
#     aero_tools = db.query(AeroTool).all()
#     print("| id|           type        |         name         |")
#     for i_aero_tool in aero_tools:
#         print(f"|{i_aero_tool.id:3}| {i_aero_tool.name:22}| {i_aero_tool.type:20} |")




# fill_test_data()
# print_all_from_db()
