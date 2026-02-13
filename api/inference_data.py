from pydantic import BaseModel
class ClientData(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    DAYS_EMPLOYED: float
    DAYS_BIRTH: float
   
    class Config:
        json_schema_extra = {
            "example": {
                "AMT_INCOME_TOTAL": 120000,
                "AMT_CREDIT": 350000,
                "AMT_ANNUITY": 18000,
                "DAYS_EMPLOYED": -2000,
                "DAYS_BIRTH": -15000
            }
        }