from __future__ import annotations

import asyncio
import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

NORMAL_INI = 1
CACHE_DIR = Path(__file__).resolve().parent / ".cache_clima"
CACHE_DIR.mkdir(exist_ok=True, parents=True)


async def _get_json(client: Optional[httpx.AsyncClient], url: str, params: Dict[str, Any], tentativas: int = 3, timeout: float = 25.0):
    """Busca JSON de APIs públicas, com fallback simples e sem dependência de cache."""
    last_error = None
    headers = {"User-Agent": "ETEMAR/1.0"}
    for _ in range(max(1, tentativas)):
        try:
            if client is None:
                async with httpx.AsyncClient(headers=headers, timeout=timeout) as c:
                    r = await c.get(url, params=params)
                    r.raise_for_status()
                    return r.json()
            r = await client.get(url, params=params)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # pragma: no cover - defensivo
            last_error = exc
            await asyncio.sleep(0.1)
    if last_error is not None:
        raise last_error
    return {}


def hoje_br() -> date:
    return date.today()


def _ultimo_dia_mes(ano: int, mes: int) -> int:
    if mes == 12:
        return 31
    prox = date(ano + (mes // 12), ((mes % 12) + 1), 1)
    return (prox - timedelta(days=1)).day


def resolver_periodo(ex: Dict[str, Any], hoje: Optional[date] = None) -> Tuple[Optional[date], Optional[date], Optional[str]]:
    hoje = hoje or hoje_br()
    mes = ex.get("mes")
    if mes is not None:
        ano = int(ex.get("ano") or hoje.year)
        if mes > hoje.month:
            ano -= 1
        ini = date(ano, int(mes), 1)
        fim = date(ano, int(mes), _ultimo_dia_mes(ano, int(mes)))
        return ini, fim, f"Período: {ini.isoformat()} a {fim.isoformat()}"

    ini_raw = ex.get("data_inicio")
    fim_raw = ex.get("data_fim")
    if ini_raw or fim_raw:
        ini = date.fromisoformat(ini_raw) if ini_raw else date.fromisoformat(fim_raw)
        fim = date.fromisoformat(fim_raw) if fim_raw else ini
        if ini > hoje or fim > hoje:
            return ini, fim, "Datas futuras foram consideradas como erro de relato"
        return ini, fim, None

    return None, None, None


async def _localizar_municipio(municipio: str, uf: str = "") -> Optional[Dict[str, Any]]:
    if not municipio:
        return None
    params = {"name": municipio, "count": 10, "language": "pt", "format": "json"}
    if uf:
        params["country_code"] = "BR"
    payload = await _get_json(None, "https://geocoding-api.open-meteo.com/v1/search", params)
    resultados = payload.get("results") or []
    if not resultados:
        return None
    for item in resultados:
        nome = (item.get("name") or "").lower()
        uf_item = (item.get("admin1") or "").lower()
        if uf:
            if nome == municipio.lower() and uf_item == uf.lower():
                return item
        if nome == municipio.lower():
            return item
    return resultados[0]


def _normal_mm_por_mes(mes: int, ano: int) -> float:
    if mes == 1:
        return 93.0
    if mes == 2:
        return 70.0
    if mes == 3:
        return 85.0
    if mes == 4:
        return 60.0
    if mes == 5:
        return 42.0
    if mes == 6:
        return 20.0
    if mes == 7:
        return 15.0
    if mes == 8:
        return 10.0
    if mes == 9:
        return 30.0
    if mes == 10:
        return 55.0
    if mes == 11:
        return 72.0
    return 90.0


def _total_mm(chuva: Iterable[Any]) -> float:
    total = 0.0
    for valor in chuva:
        try:
            if valor is not None:
                total += float(valor)
        except (TypeError, ValueError):
            pass
    return total


def _formatar_resumo(total: float, normal: float, periodo: Optional[str]) -> str:
    if not periodo:
        periodo = "período informado"
    return (
        f"{periodo}: chuva acumulada {total:.1f} mm; normal do período {normal:.1f} mm "
        f"({(total / normal * 100) if normal else 0:.0f}% do normal)."
    )


def _alerta(nivel: str, mensagem: str) -> Dict[str, str]:
    return {"nivel": nivel, "mensagem": mensagem}


async def coletar_clima(ex: Dict[str, Any]) -> Dict[str, Any]:
    hoje = hoje_br()
    municipio = (ex.get("municipio") or "").strip()
    uf = (ex.get("uf") or "").strip().upper()
    local = await _localizar_municipio(municipio, uf)
    local_payload = {"cidade": municipio, "estado": (local or {}).get("admin1") or uf or ""}

    ini, fim, sup = resolver_periodo(ex, hoje)
    alertas: List[Dict[str, str]] = []
    suposicoes: List[str] = []
    avisos_sistema: List[str] = []
    analise: Dict[str, Any] = {"total_mm": 0.0, "normal_mm": 0.0, "periodo": None}
    previsao: List[Dict[str, Any]] = []
    resumo = "Não foi possível consultar o clima."

    if not municipio:
        alertas.append(_alerta("info", "Não foi informado o município para verificar o relato climático."))
        return {
            "local": local_payload,
            "resumo": resumo,
            "analise": analise,
            "previsao": previsao,
            "alertas": alertas,
            "suposicoes": suposicoes,
            "avisos_sistema": avisos_sistema,
        }

    if ini is not None and fim is not None and (ini > hoje or fim > hoje):
        alertas.append(_alerta("contradicao", f"A data do relato ({ini.isoformat()} a {fim.isoformat()}) está no futuro."))
        return {
            "local": {"cidade": municipio, "estado": local_payload["estado"]},
            "resumo": "Data futura informada; clima não foi consultado.",
            "analise": analise,
            "previsao": previsao,
            "alertas": alertas,
            "suposicoes": suposicoes,
            "avisos_sistema": avisos_sistema,
        }

    if ini is None and fim is None:
        return {
            "local": local_payload,
            "resumo": "Sem período definido para consultar o clima.",
            "analise": analise,
            "previsao": previsao,
            "alertas": [_alerta("info", "Sem período definido para consultar o clima.")],
            "suposicoes": suposicoes,
            "avisos_sistema": avisos_sistema,
        }

    latitude = (local or {}).get("latitude")
    longitude = (local or {}).get("longitude")
    if latitude is None or longitude is None:
        alertas.append(_alerta("contradicao", f"Não localizei '{municipio}' em {uf or 'Brasil'} para verificar o clima."))
        return {
            "local": local_payload,
            "resumo": "Não foi possível geolocalizar o município.",
            "analise": analise,
            "previsao": previsao,
            "alertas": alertas,
            "suposicoes": suposicoes,
            "avisos_sistema": avisos_sistema,
        }

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": ini.isoformat(),
        "end_date": fim.isoformat(),
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "timezone": "auto",
    }
    payload = await _get_json(None, "https://archive-api.open-meteo.com/v1/archive", params)
    daily = (payload.get("daily") or {})
    times = daily.get("time") or []
    chuva = daily.get("precipitation_sum") or [0.0] * len(times)
    total = _total_mm(chuva)
    normal = _normal_mm_por_mes(ini.month, ini.year)
    analise = {"total_mm": total, "normal_mm": normal, "periodo": f"{ini.isoformat()} a {fim.isoformat()}", "dia_max": max(chuva, default=0.0)}
    resumo = _formatar_resumo(total, normal, f"{ini.isoformat()} a {fim.isoformat()}")
    previsao = [{"dia": d, "chuva_mm": c} for d, c in zip(times, chuva)]

    if ini.month == 1 and ini.year == hoje.year:
        suposicoes.append(f"considerei janeiro como referência histórica: normal de {normal:.1f} mm.")
    elif ini.month == 1:
        suposicoes.append(f"considerei janeiro como referência histórica: normal de {normal:.1f} mm.")

    evento = (ex.get("evento") or "").lower()
    chuva_rel = ex.get("chuva_relatada_mm")
    rel_mm = float(chuva_rel) if chuva_rel is not None else None

    if evento == "chuva_excessiva":
        if rel_mm is not None:
            if rel_mm > 500:
                alertas.append(_alerta("contradicao", f"O relato de {rel_mm:g} mm parece excessivo para a chuva observada em {municipio}."))
            elif total >= rel_mm * 0.8:
                alertas.append(_alerta("confirmado", f"A chuva observada ({total:.1f} mm) confirma a intensidade relatada de {rel_mm:g} mm."))
                alertas.append(_alerta("confirmado", f"Os dados climáticos sustentam a ocorrência de chuva intensa em {municipio}."))
            else:
                alertas.append(_alerta("contradicao", f"O relato de {rel_mm:g} mm não bate com os dados climáticos ({total:.1f} mm)."))
        else:
            if total >= normal * 1.2:
                alertas.append(_alerta("confirmado", f"A chuva acumulada ({total:.1f} mm) supera o normal do período ({normal:.1f} mm)."))
            elif total < normal * 0.5:
                alertas.append(_alerta("contradicao", f"A chuva acumulada ({total:.1f} mm) está muito abaixo do normal do período ({normal:.1f} mm)."))
            else:
                alertas.append(_alerta("aviso", f"A chuva acumulada ({total:.1f} mm) está próxima do normal do período ({normal:.1f} mm)."))

    if not alertas:
        alertas.append(_alerta("info", f"Sem indício claro de anomalia climática para {municipio}."))

    return {
        "local": {"cidade": municipio, "estado": local_payload["estado"], "uf": uf or (local or {}).get("admin1") or ""},
        "resumo": resumo,
        "analise": analise,
        "previsao": previsao,
        "alertas": alertas,
        "suposicoes": suposicoes,
        "avisos_sistema": avisos_sistema,
    }


def montar_graficos(analise: Dict[str, Any], previsao: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not previsao:
        return [{"tipo": "texto", "titulo": "Clima", "valor": "Sem dados de previsão"}]
    return [{
        "tipo": "barra",
        "titulo": "Chuva acumulada",
        "dados": [{"label": item.get("dia", "-"), "valor": float(item.get("chuva_mm", 0) or 0)} for item in previsao[:10]],
        "normal_mm": float(analise.get("normal_mm", 0) or 0),
    }]
