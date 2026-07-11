import os
import json
import logging
from pathlib import Path
try:
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.responses import FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
except Exception:
    # Fallback stubs so linters/IDEs won't error when fastapi isn't installed.
    class FastAPI:
        def __init__(self, *args, **kwargs):
            raise ImportError("fastapi is required to run this application. Install it with 'pip install fastapi'.")

    class HTTPException(Exception):
        def __init__(self, status_code: int = 500, detail: str | None = None):
            super().__init__(detail)

    class Response:
        def __init__(self, *args, **kwargs):
            raise ImportError("fastapi is required to run this application. Install it with 'pip install fastapi'.")

    class FileResponse:
        def __init__(self, *args, **kwargs):
            raise ImportError("fastapi is required to run this application. Install it with 'pip install fastapi'.")

    class CORSMiddleware:
        def __init__(self, *args, **kwargs):
            raise ImportError("fastapi is required to run this application. Install it with 'pip install fastapi'.")

    class StaticFiles:
        def __init__(self, *args, **kwargs):
            raise ImportError("fastapi is required to run this application. Install it with 'pip install fastapi'.")
import pydantic
from typing import List, Dict
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

app = FastAPI(title="ETEMAR AGRO - Logic Engine")
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
@app.get("/")
async def rota_principal():
    return {"status": "Online", "sistema": "E.T.E.M.A.R Agro Engine", "versao": "2.0"}

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

client = AsyncGroq(api_key=GROQ_API_KEY)

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

class ConsultaAgro(pydantic.BaseModel):
    pergunta: str
    historico: List[Dict[str, str]] = []

@app.post("/api/agro/consultar")
async def consultar_ia_agro(req: ConsultaAgro):
    print(f"[API] /api/agro/consultar called with pergunta={req.pergunta!r}, historico_items={len(req.historico)}")
    if len(req.pergunta) < 2:
        raise HTTPException(status_code=400, detail="Entrada vazia. A pergunta deve conter pelo menos 2 caracteres.")

    # PROMPT DE ALTO NÍVEL: O JSON do financeiro agora é estruturado
    prompt_sistema = """Você é o Pesquisador Chefe de Agronomia do sistema E.T.E.M.A.R. 
    Se o produtor der dados vagos, exija Área, Cultura, Local e Época retornando: {"precisa_info": true, "pergunta_ia": "..."}.
    
    REGRA DE OURO: 
    É ESTRITAMENTE PROIBIDO dar respostas rasas de 1 linha. Use negrito (**) para palavras-chave.
    Seja transparente e lógico na matemática.

    RETORNE OBRIGATORIAMENTE ESTE JSON ESTRITO:
    {
        "precisa_info": false,
        "rastreabilidade": {
            "fonte_clima": "🔵 INMET / MapBiomas",
            "fonte_mercado": "🔵 CEPEA / CONAB"
        },
        "diagnostico": {
            "texto": "Diagnóstico aprofundado com pelo menos 2 parágrafos. Explique cientificamente a causa.",
            "matriz_confianca": [
                {"causa": "Estresse Hídrico", "probabilidade": 90},
                {"causa": "Deficiência Nutricional", "probabilidade": 45}
            ]
        },
        "plano_acao": {
            "imediato": "Ação tática de campo imediata.",
            "dias_30": "Ação para os próximos 30 dias.",
            "proxima_safra": "Estratégia de investimento estrutural."
        },
        "textos_extras": {
            "clima": "Análise climática baseada no local.",
            "mercado": "Lei de oferta e demanda aplicada ao caso."
        },
        "financeiro": {
            "premissas": {
                "area": "80 ha",
                "produtividade_media": "30 t/ha",
                "cotacao_atual": "R$ 2,50/kg"
            },
            "impacto": {
                "taxa_perda": "50%",
                "volume_perdido": "1.200 t"
            },
            "resultado": "R$ 3.000.000,00",
            "formula_aplicada": "(80 ha * 30 t/ha) * 50% de perda = 1.200 t. (1.200 t * 1000 kg) * R$ 2,50 = R$ 3.000.000,00"
        },
        "graficos": [
            {"titulo": "Histórico de Preço (R$/kg)", "tipo": "line", "labels": ["Safra Ant", "Mês Passado", "Atual", "Proj"], "dados": [4.1, 4.3, 5.0, 5.2]},
            {"titulo": "Anomalia Climática (mm)", "tipo": "bar", "labels": ["Mês -2", "Mês -1", "Atual"], "dados": [80, 40, 10]}
        ]
    }
    """

    mensagens = [{"role": "system", "content": prompt_sistema}]
    for msg in req.historico: mensagens.append(msg)
    mensagens.append({"role": "user", "content": req.pergunta})

    try:
        res = await client.chat.completions.create(
            messages=mensagens,
            model="llama-3.3-70b-versatile",
            max_tokens=4500, 
            temperature=0.3,
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