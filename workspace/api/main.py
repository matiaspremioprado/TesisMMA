from fastapi import FastAPI
from pydantic import BaseModel

# Inicializa la aplicación FastAPI
app = FastAPI()

# Modelo de datos de entrada (request body)
class ItemIn(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

# Modelo de datos de salida (respuesta)
class ItemOut(BaseModel):
    name: str
    price: float
    total_price: float

# Ruta de ejemplo para procesar un objeto de tipo ItemIn
@app.post("/items/", response_model=ItemOut)
async def create_item(item: ItemIn):
    total_price = item.price + (item.tax or 0)
    return ItemOut(name=item.name, price=item.price, total_price=total_price)

# Ruta para obtener el estado de la API
@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI!"}
