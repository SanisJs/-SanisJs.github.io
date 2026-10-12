import os
import json
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict
from groq import AsyncGroq
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

app = FastAPI(title="ETEMAR AGRO - Logic & RAG Engine")
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

client = AsyncGroq(api_key=GROQ_API_KEY)

banco_vetorial = None
try:
    if os.path.exists("./banco_conhecimento"):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        banco_vetorial = Chroma(persist_directory="./banco_conhecimento", embedding_function=embeddings)
        print(">> Banco de Conhecimento (PDFs) carregado com sucesso!")
except Exception as e:
    print(f">> Erro ao carregar Banco de Conhecimento: {e}")

@app.get("/api/status")
async def rota_status():
    return {"status": "Online", "sistema": "E.T.E.M.A.R Agro Engine", "versao": "3.0 + Banco Vetorial"}

@app.get('/favicon.ico')
async def favicon():
    return Response(status_code=204)

@app.get('/')
async def root():
    return FileResponse(Path(__file__).resolve().parent / 'index.html')

@app.get('/index.html')
async def index_html():
    return FileResponse(Path(__file__).resolve().parent / 'index.html')

app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent), name="static")

class ConsultaAgro(BaseModel):
    pergunta: str
    historico: List[Dict[str, str]] = []

@app.post("/api/agro/consultar")
async def consultar_ia_agro(req: ConsultaAgro):
    if len(req.pergunta) < 2:
        raise HTTPException(status_code=400, detail="Entrada vazia. A pergunta deve conter pelo menos 2 caracteres.")

    instrucao_base = ""
    if banco_vetorial:
        try:
            resultados = banco_vetorial.similarity_search(req.pergunta, k=4)
            contexto_extraido = "\n\n".join([doc.page_content for doc in resultados])
            if contexto_extraido:
                instrucao_base = f"\nATENÇÃO! UTILIZE ESTES DADOS DE PESQUISAS CIENTÍFICAS COMO BASE TÉCNICA PARA A SUA RESPOSTA:\n{contexto_extraido}\n----------------------\n"
        except Exception as e:
            print(f"Erro na busca do banco vetorial: {e}")

    prompt_sistema = f"""Você é o Pesquisador Chefe de Agronomia do sistema E.T.E.M.A.R. 
    
    {instrucao_base}

    Se o produtor der dados vagos, exija Área, Cultura, Local e Época retornando: {{"precisa_info": true, "pergunta_ia": "..."}}.
    
    REGRA DE OURO: 
    É ESTRITAMENTE PROIBIDO dar respostas rasas de 1 linha. Use negrito (**) para palavras-chave.
    Seja transparente e lógico na matemática. Sempre cruze o relato do usuário com os dados das pesquisas acima, caso existam.

    RETORNE OBRIGATORIAMENTE ESTE JSON ESTRITO:
    {{
        "precisa_info": false,
        "rastreabilidade": {{
            "fonte_clima": "🔵 INMET / Pesquisas ETEMAR",
            "fonte_mercado": "🔵 CEPEA / CONAB"
        }},
        "diagnostico": {{
            "texto": "Diagnóstico aprofundado com pelo menos 2 parágrafos. Explique cientificamente a causa.",
            "matriz_confianca": [
                {{"causa": "Estresse Hídrico", "probabilidade": 90}},
                {{"causa": "Deficiência Nutricional", "probabilidade": 45}}
            ]
        }},
        "plano_acao": {{
            "imediato": "Ação tática de campo imediata.",
            "dias_30": "Ação para os próximos 30 dias.",
            "proxima_safra": "Estratégia de investimento estrutural."
        }},
        "textos_extras": {{
            "clima": "Análise climática baseada no local.",
            "mercado": "Lei de oferta e demanda aplicada ao caso."
        }},
        "financeiro": {{
            "premissas": {{
                "area": "80 ha",
                "produtividade_media": "30 t/ha",
                "cotacao_atual": "R$ 2,50/kg"
            }},
            "impacto": {{
                "taxa_perda": "50%",
                "volume_perdido": "1.200 t"
            }},
            "resultado": "R$ 3.000.000,00",
            "formula_aplicada": "(80 ha * 30 t/ha) * 50% de perda = 1.200 t. (1.200 t * 1000 kg) * R$ 2,50 = R$ 3.000.000,00"
        }},
        "graficos": [
            {{"titulo": "Histórico de Preço (R$/kg)", "tipo": "line", "labels": ["Safra Ant", "Mês Passado", "Atual", "Proj"], "dados": [4.1, 4.3, 5.0, 5.2]}},
            {{"titulo": "Anomalia Climática (mm)", "tipo": "bar", "labels": ["Mês -2", "Mês -1", "Atual"], "dados": [80, 40, 10]}}
        ]
    }}
    """

    mensagens = [{"role": "system", "content": prompt_sistema}]
    for msg in req.historico: mensagens.append(msg)
    mensagens.append({"role": "user", "content": req.pergunta})

    try:
        res = await client.chat.completions.create(
            messages=mensagens,
            model="llama-3.3-70b-versatile",
            max_tokens=4500, 
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        conteudo = res.choices[0].message.content.strip()
        if conteudo.startswith("```json"):
            conteudo = conteudo[7:]
        if conteudo.endswith("```"):
            conteudo = conteudo[:-3]
            
        return json.loads(conteudo)
        
    except Exception as e:
        print(f"🚨 ERRO AGRO: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_agro:app", host="0.0.0.0", port=3001, reload=True)