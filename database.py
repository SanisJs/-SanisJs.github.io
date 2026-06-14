from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./etemar.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Simulado(Base):
    __tablename__ = "simulados"
    id = Column(Integer, primary_key=True, index=True)
    titulo = Column(String)
    is_reforco = Column(Boolean, default=False)
    data_criacao = Column(DateTime, default=datetime.utcnow)

class Questao(Base):
    __tablename__ = "questoes"
    id = Column(Integer, primary_key=True, index=True)
    simulado_id = Column(Integer)
    pergunta = Column(Text)
    opcoes = Column(Text)
    correta = Column(Integer)
    explicacao = Column(Text)
    dica_tutor = Column(Text)
    tema_especifico = Column(String) # A IA vai classificar o tema ex: "Termodinâmica"

class HistoricoDesempenho(Base):
    __tablename__ = "historico_desempenho"
    id = Column(Integer, primary_key=True, index=True)
    tema = Column(String, index=True)
    acertou = Column(Boolean)
    data = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)