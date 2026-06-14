import os
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

app = FastAPI(title="ETEMAR AGRO - Logic Engine")
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

client = AsyncGroq(api_key=GROQ_API_KEY)

class ConsultaAgro(BaseModel):
    pergunta: str
    historico: List[Dict[str, str]] = []

@app.post("/api/agro/consultar")
async def consultar_ia_agro(req: ConsultaAgro):
    if len(req.pergunta) < 2:
        raise HTTPException(status_code=400, detail="Entrada vazia.")

    # PROMPT DE ALTO NÍVEL: Focado em matemática agronômica real e rastreabilidade.
    prompt_sistema = """Você é o Motor de Inteligência Agronômica E.T.E.M.A.R. 
    Se o produtor der dados vagos, exija Área, Cultura, Local e Época retornando: {"precisa_info": true, "pergunta_ia": "..."}.
    
    Se houver dados suficientes, gere um DOSSIÊ TÉCNICO usando COERÊNCIA MATEMÁTICA AGRONÔMICA:
    - Manga Tommy produz ~30 a 40 t/ha. Soja ~50 a 60 sc/ha. 
    - Se a seca foi severa, a perda deve refletir isso (ex: 20% a 40%). Não invente perdas de 1% para climas extremos.
    - MOSTRE A MATEMÁTICA: (Área * Produtividade Média * % Perda * Preço/kg).
    
    RETORNE OBRIGATORIAMENTE ESTE JSON ESTRITO:
    {
        "precisa_info": false,
        "rastreabilidade": {
            "fonte_clima": "🔵 INMET / API Simulado",
            "fonte_mercado": "🔵 CEPEA / CONAB Simulado"
        },
        "diagnostico": {
            "texto": "Diagnóstico agronômico direto e técnico.",
            "matriz_confianca": [
                {"causa": "Estresse Hídrico Severo", "probabilidade": 90},
                {"causa": "Deficiência Nutricional (Ca/K)", "probabilidade": 45},
                {"causa": "Ataque Fitossanitário", "probabilidade": 15}
            ]
        },
        "plano_acao": {
            "imediato": "O que fazer nas próximas 48h (ex: irrigação socorro).",
            "dias_30": "O que fazer no próximo mês (ex: manejo foliar).",
            "proxima_safra": "Investimentos estruturais (ex: automação, poda)."
        },
        "textos_extras": {
            "clima": "Análise climática com dados.",
            "mercado": "Comparativo de safras anteriores e mercado atual."
        },
        "financeiro": {
            "memoria_calculo": "80 ha x 30 t/ha = 2.400 t. Perda de 25% = 600 t. Cotação: R$ 2,50/kg.",
            "perda_toneladas": "600 t",
            "prejuizo_total": "R$ 1.500.000,00"
        },
        "graficos": [
            {"titulo": "Histórico e Projeção de Preço (R$/kg)", "tipo": "line", "labels": ["Safra 23", "Safra 24", "Atual", "Proj 26"], "dados": [2.10, 2.30, 2.50, 2.20]},
            {"titulo": "Déficit Hídrico (mm)", "tipo": "bar", "labels": ["Mês -2", "Mês -1", "Atual"], "dados": [120, 40, 10]}
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
            max_tokens=4000, 
            temperature=0.1, # Quase zero: Máxima precisão matemática e zero alucinação
            response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        print(f"🚨 ERRO AGRO: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_agro:app", host="0.0.0.0", port=3001, reload=True)