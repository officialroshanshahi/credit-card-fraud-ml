from typing import List
from pydantic import BaseModel, Field

# PLEASE NOTE: the v1 to v28 could be other features than named v... but since we do not have the raw dataset
# we only have access to PCA dataset so using v1 to v28 for this
class CreditCardTransaction(BaseModel):
    Time: float = Field(..., alias="Time", example=61290.0)
    V1: float = Field(..., alias="V1", example=1.2288211502)
    V2: float = Field(..., alias="V2", example=-0.0634077165)
    V3: float = Field(..., alias="V3", example=0.2741451422)
    V4: float = Field(..., alias="V4", example=0.6474650218)
    V5: float = Field(..., alias="V5", example=-0.0481345612)
    V6: float = Field(..., alias="V6", example=0.3720730286)
    V7: float = Field(..., alias="V7", example=-0.2242305874)
    V8: float = Field(..., alias="V8", example=0.0799390492)
    V9: float = Field(..., alias="V9", example=0.6407588171)
    V10: float = Field(..., alias="V10", example=-0.2730537022)
    V11: float = Field(..., alias="V11", example=-1.2527279388)
    V12: float = Field(..., alias="V12", example=0.4650787707)
    V13: float = Field(..., alias="V13", example=0.4005021153)
    V14: float = Field(..., alias="V14", example=-0.2928418606)
    V15: float = Field(..., alias="V15", example=-0.101774016)
    V16: float = Field(..., alias="V16", example=-0.3998358978)
    V17: float = Field(..., alias="V17", example=0.0343356568)
    V18: float = Field(..., alias="V18", example=-0.7835502549)
    V19: float = Field(..., alias="V19", example=0.1413449004)
    V20: float = Field(..., alias="V20", example=-0.0965659024)
    V21: float = Field(..., alias="V21", example=-0.1295544481)
    V22: float = Field(..., alias="V22", example=-0.0837793282)
    V23: float = Field(..., alias="V23", example=-0.1516614739)
    V24: float = Field(..., alias="V24", example=-0.7003715973)
    V25: float = Field(..., alias="V25", example=0.5985501645)
    V26: float = Field(..., alias="V26", example=0.4914090706)
    V27: float = Field(..., alias="V27", example=0.0029892597)
    V28: float = Field(..., alias="V28", example=0.0017822861)
    Amount: float = Field(..., alias="Amount", example=11.5)

    class Config:
        populate_by_name = True

class UserAuth(BaseModel):
    username: str
    password: str

class ReqSchema(BaseModel):
    data: CreditCardTransaction

class Batch(BaseModel):
    data: List[CreditCardTransaction]

class Label(BaseModel):
    request_id: str
    class_label: int = Field(..., alias="class", example=0)

    class Config:
        populate_by_name = True