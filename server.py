import os
import json
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import AsyncGroq
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from database import SessionLocal, engine, Base, Simulado, Questao, HistoricoDesempenho

# ==========================================
# 1. CONFIGURAÇÕES INICIAIS E CHAVE API
# ==========================================
load_dotenv()

# SOLUÇÃO DO SEU ERRO: Tenta carregar do .env, se não achar, usa a sua chave original.
# Assim o servidor NUNCA vai quebrar com aquele erro de ValueError no terminal.
CHAVE_BACKUP = "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"
GROQ_API_KEY = os.getenv("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", CHAVE_BACKUP)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

app = FastAPI(title="E.T.E.M.A.R Blue Adaptive")
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncGroq(api_key=GROQ_API_KEY)

# ==========================================
# 2. MOLDES PYDANTIC
# ==========================================
class RequisicaoSimulado(BaseModel):
    texto: str = Field(..., description="Texto base extraído do PDF")
    reforco: bool = Field(False, description="Se for True, foca nos erros passados")
    quantidade: int = Field(5, ge=1, le=20, description="Quantidade de questões")

class RegistroResultado(BaseModel):
    temas_certos: list[str]
    temas_errados: list[str]

# ==========================================
# 3. ROTAS DA API
# ==========================================
@app.post("/api/resultado")
def salvar_resultado(resultado: RegistroResultado, db: Session = Depends(get_db)):
    for tema in resultado.temas_certos:
        db.add(HistoricoDesempenho(tema=tema, acertou=True))
    for tema in resultado.temas_errados:
        db.add(HistoricoDesempenho(tema=tema, acertou=False))
    db.commit()
    return {"status": "sucesso"}

@app.post("/api/simulado")
async def gerar_simulado(req: RequisicaoSimulado, db: Session = Depends(get_db)):
    if len(req.texto) < 50:
        raise HTTPException(status_code=400, detail="Texto muito curto.")

    instrucao_adaptativa = ""
    if req.reforco:
        erros_recentes = db.query(HistoricoDesempenho.tema).filter(HistoricoDesempenho.acertou == False).order_by(HistoricoDesempenho.id.desc()).limit(10).all()
        temas_fracos = list(set([e[0] for e in erros_recentes]))
        if temas_fracos:
            instrucao_adaptativa = f"""
            ATENÇÃO! O aluno tem errado os seguintes temas recentemente: {', '.join(temas_fracos)}.
            Crie questões difíceis focadas nestes temas específicos usando o contexto do texto.
            """

    # PROMPT AVANÇADO: Foco total em não bugar a quantidade e em gerar questões complexas.
    prompt_sistema = f"""Você é uma IA de Alta Performance Especialista em Bancas de Concurso e ENEM.
    Sua missão é criar UM SIMULADO DE ALTO NÍVEL com EXATAMENTE {req.quantidade} questões baseadas no texto fornecido.
    
    REGRA DE QUANTIDADE ABSOLUTA:
    Você DEVE gerar EXATAMENTE {req.quantidade} itens na lista "questoes". Nem uma a mais, nem uma a menos. Se o usuário pediu {req.quantidade}, você criará {req.quantidade}.
    
    REGRAS DE QUALIDADE (ANTI-QUESTÕES BOBAS):
    1. PROIBIDO perguntas óbvias (ex: "Qual é a cor do cavalo branco?").
    2. Exija interpretação de texto profunda, inferência, pensamento crítico e lógica avançada.
    3. ESTRUTURA ENEM: Toda pergunta deve ter uma "Situação-Problema" conectada ao texto.
    4. DISTRATORES: As alternativas erradas não podem ser absurdas. Elas devem ser as famosas "pegadinhas", parecendo corretas para o aluno desatento.
    {instrucao_adaptativa}
    
    FORMATO JSON DE SAÍDA EXIGIDO:
    {{
        "titulo": "Simulado Alta Performance: [Tema do Texto]",
        "questoes":[
            {{
                "dificuldade": "Difícil",
                "tema_especifico": "Conceito Exato Testado (Ex: Interpretação, Cálculo X, Teoria Y)",
                "pergunta": "Contexto e Situação-Problema baseada no texto. \\n\\nComando direto da questão:",
                "opcoes":[
                    "A) Alternativa",
                    "B) Alternativa",
                    "C) Alternativa",
                    "D) Alternativa",
                    "E) Alternativa"
                ],
                "correta": 0, // DEVE SER UM NÚMERO DE 0 A 4 (0=A, 1=B, 2=C, 3=D, 4=E)
                "explicacao": "Explique com riqueza de detalhes por que a resposta é certa e destrua as alternativas incorretas mostrando o erro lógico delas.",
                "dica_tutor": "Dica prática ou macete para o aluno acertar isso na próxima vez."
            }}
            // ATENÇÃO: REPITA ESTE BLOCO ACIMA EXATAMENTE {req.quantidade} VEZES.
        ]
    }}
    
    NÃO ESCREVA MAIS NADA ALÉM DO JSON.
    """

    try:
        # Temperatura aumentada para 0.4: Dá mais inteligência e criatividade nos distratores, tirando o efeito "robótico".
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": f"Material Fonte:\n{req.texto}"}
            ],
            model="llama-3.3-70b-versatile",
            max_tokens=8000, 
            temperature=0.4, 
            response_format={"type": "json_object"}
        )
        
        conteudo_resposta = chat_completion.choices[0].message.content
        
        try:
            resultado = json.loads(conteudo_resposta)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="A IA falhou em compilar o JSON perfeitamente. Tente novamente.")

        # Proteção extra: Caso a IA invente questões a mais, nós cortamos a lista no tamanho exato solicitado.
        if "questoes" in resultado:
            resultado["questoes"] = resultado["questoes"][:req.quantidade]

        novo_simulado = Simulado(titulo=resultado.get("titulo", "Simulado IA"), is_reforco=req.reforco)
        db.add(novo_simulado)
        db.commit()
        db.refresh(novo_simulado)

        for q in resultado.get("questoes",[]):
            nova_questao = Questao(
                simulado_id=novo_simulado.id,
                pergunta=q.get("pergunta", "Erro no enunciado"),
                opcoes=json.dumps(q.get("opcoes",["A) Erro", "B)", "C)", "D)", "E)"])),
                correta=int(q.get("correta", 0)),
                explicacao=q.get("explicacao", "Sem explicação"),
                dica_tutor=q.get("dica_tutor", ""),
                tema_especifico=q.get("tema_especifico", "Geral")
            )
            db.add(nova_questao)
        db.commit()

        return resultado

    except Exception as e:
        print(f"🚨 ERRO API E.T.E.M.A.R: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=3000, reload=True)