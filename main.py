# main.py
from fastapi import FastAPI
from sqlalchemy import create_engine
SQLALCHEMY_DATABASE_URL = "sqlite:///./.workdir/sql_app.db"
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import  Column, Integer, String
from sqlalchemy.orm import sessionmaker


# создание движка
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


class Base(DeclarativeBase): pass
class AeroTool(Base):
    __tablename__ = "aerotool_set"
 
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    delivery_state = Column(String) # "in_stock", "on_hands"
    type = Column(Integer,)
    delivery_id =Column(Integer,)  # Наюоров для выдачи может быть много и каждый инструмент принадлежит к одной из выдач

# создаем таблицы
Base.metadata.create_all(bind=engine)

SessionLocal = sessionmaker(autoflush=False, bind=engine)
db = SessionLocal()

def check_exist_tool_in_db_by_name(name) -> bool:
    first = db.query(AeroTool).filter(AeroTool.name==name).first() 

    if first :
        print(f"Find existing item with name {first.name}: id {first.id} - ({first.type}) - {first.delivery_id}")
        return True    
    else:
        return False

def fill_test_data():
    deliveries = [1, 2, 3]
    tool_types = [
                    "screw_flat", # 1. Плоская отвертка (-)
                    "screw_plus", # 2. Крестовая отвертка (+)
                    "offset plus_screw", # 3. отвертка на смещенный крест
                    "screw_plus", # 4. Коловорот
                    "safety_pliers", # 5. Пассатижи контровочные
                    "pliers", # 6. Пассатижи
                    "shernitsa", # 7. Шерница
                    "adjustable_wrench", # 8. Разводной ключ
                    "can_opener", # 9. Открывалка для банок с маслом
                    "open_end_wrench_3_4", # 10. Ключ рожковый накидной 3/4
                    "side_cutters", # 11. Бокорезы
                  ]
    for i_delevery in deliveries:
        for i_tool_type in tool_types:
            new_tool_unique_name = f"{i_tool_type}_{i_delevery}"
            if not check_exist_tool_in_db_by_name(new_tool_unique_name):
                screw_plus = AeroTool(name=new_tool_unique_name, type=i_tool_type, delivery_id=i_delevery, delivery_state = "on_hands")
                db.add(screw_plus)     # добавляем в бд
    db.commit()     # сохраняем изменения

def print_all_from_db():
    # получение всех объектов
    aero_tools = db.query(AeroTool).all()
    for i_aero_tool in aero_tools:
        print(f"{i_aero_tool.id}.{i_aero_tool.type} ({i_aero_tool.name})")

fill_test_data()
print_all_from_db()
