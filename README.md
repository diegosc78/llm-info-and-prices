# LLM Info & Prices

Microservicio (FastAPI + FastMCP) que agrega modelos, descripciones, capacidades y
precios de LLMs desde varias fuentes y los sirve de tres maneras:

1. **Cost map compatible con LiteLLM** (`model_prices_and_context_window.json`) listo
   para cargar en tu proxy o usar en `/v1/model_group/info`.
2. **API REST** con búsqueda y consulta enriquecida de todo lo recopilado.
3. **Servidor MCP (Streamable HTTP)** para que agentes consulten nombres,
   descripciones y URLs de los modelos.

## Fuentes de datos

| Fuente | Endpoint | Contenido |
|--------|----------|-----------|
| LiteLLM upstream | GitHub `model_prices_and_context_window.json` | 3.5k modelos con precios oficiales |
| Portkey | `pricing/{provider}.json` + `general/{provider}.json` | Fichas de productos y proveedores |
| CloudPrice | `ai.cloudprice.net/api/v1` (flat map + catálogo) | Precios y capacidades, aliases |
| OpenRouter | `GET /models` | Slug canónico, precios, context window, descripción |
| **Tu LiteLLM** | `/v1/model/info` (fallback `/v1/models`) | Modelos de tus instancias |
| **Tu OpenWebUI** | `/api/models/list` + `/api/models/model?id=` | Modelos y agentes de tu instancia |

Los modelos de tus instancias (LiteLLM + OpenWebUI) se fusionan con los de las
fuentes públicas: si el mismo modelo existe en abierto, hereda la información/price
pública y queda marcado como `instances: ["litellm_user", "openwebui"]`.

### Prioridad de precios

OpenRouter > CloudPrice > Portkey > LiteLLM upstream.
Cuando hay varias fuentes, se elige por campo la de mayor prioridad y se guarda el
origen en `pricing_source`.

## Configuración

Variables de entorno (ver `.env.example`):

```env
# Tus instancias (opcional pero recomendado)
LITELLM_API_URL=https://llmproxy.altia.es
LITELLM_API_KEY=sk-...
OPENWEBUI_API_URL=https://myassistant.altia.es
OPENWEBUI_API_KEY=sk-...

# OpenRouter (opcional)
OPENROUTER_API_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=sk-or-...

# Cache / refresh
REFRESH_INTERVAL_SECONDS=3600
CACHE_TTL_SECONDS=3600
HTTP_TIMEOUT_SECONDS=15
HTTP_MAX_RETRIES=3
STARTUP_REFRESH=true

# Validación del cost map
MODEL_COST_MAP_MIN_MODEL_COUNT=50
MODEL_COST_MAP_MAX_SHRINK_RATIO=0.5
```

## Arranque

```bash
uv sync            # instala dependencias en .venv
cp .env.example .env   # rellena tus claves
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t llm-info-and-prices .
docker run -p 8000:8000 --env-file .env llm-info-and-prices
```

Servicio listo en `http://localhost:8000` (la primera carga tarda ~20-40s en
recopilar todas las fuentes).

## API REST

| Endpoint | Descripción |
|----------|-------------|
| `GET /health` | Estado de cada fuente, total de modelos, staleness |
| `GET /sources` | Detalle de las 6 fuentes y sus counters |
| `GET /models?q=&provider=&mode=&user_instances=&page=&per_page=` | Lista/búsqueda |
| `GET /models/{slug}` | Detalle completo de un modelo (slash separado: `/models/openai/gpt-4o`) |
| `GET /models/{slug}/litellm` | Entrada del modelo en formato cost-map LiteLLM |
| `GET /model?slug=openai/gpt-4o` | Lookup por query string (alternativa a la ruta con slash) |
| `GET /litellm-cost-map?user_only=` | Cost map completo (cada variante con su precio) |
| `GET /litellm-proxy` | Config de LiteLLM proxy para usar el cost map |
| `POST /refresh` | Refresca todas las fuentes y re-ejecuta el merge |

### Ejemplos

```bash
# Búsqueda de gpt-4o
curl "http://localhost:8000/models?q=gpt-4o&per_page=20"

# Detalle de un modelo (slashes en la URL)
curl "http://localhost:8000/models/openai/gpt-4o-2024-08-06"

# Cost map solo de modelos disponibles en tus instancias
curl "http://localhost:8000/litellm-cost-map?user_only=true"
```

El cost map se puede apuntar desde tu LiteLLM proxy:

```yaml
model_list:
  - model_name: "*"
    litellm_params:
      model: openai/*
    model_info:
      cost_map_url: http://localhost:8000/litellm-cost-map?user_only=true
```

## MCP

Servidor MCP **Streamable HTTP** en `http://localhost:8000/mcp` con tres tools:
`list_models`, `search_models`, `get_model_details`.

```bash
# Cliente Python de prueba
pip install fastmcp
```

```python
from fastmcp import Client

with Client("http://localhost:8000/mcp") as client:
    tools = client.list_tools()                        # -> names/descriptions
    client.call_tool("search_models", {"query": "claude"})
    client.call_tool("get_model_details", {"slug": "anthropic/claude-4.7-opus-20260416"})
```

Configuración en un cliente MCP genérico (Claude Code, OpenWebUI, Cursor...):

```json
{
  "mcpServers": {
    "llm-info-and-prices": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

## Estructura

```
app/
  main.py               # FastAPI + lifespan, monta el MCP
  config.py             # Settings desde .env
  state.py              # AppState: fetch concurrente, merge, refresh periódico
  cache.py              # TTL cache en memoria
  models.py             # ModelData, SourcePayload, SourceStatus
  merge.py              # Dedup (union-find), fusión de precios/capacidades, variantes
  fetchers/
    base.py             # Fetcher base con retries (sin retry en 4xx)
    litellm_upstream.py # GitHub json oficial de LiteLLM
    portkey.py          # pricing/general por provider (paralelo, Sin retry 4xx)
    cloudprice.py       # ai.cloudprice.net: flat map + catálogo con aliases
    openrouter.py       # GET /models con paginación
    litellm_user.py     # tu proxy: /v1/model/info, /v1/models, /v1/model_group/info
    openwebui.py        # /api/models/list + detalle por modelo (rate-limited)
  api/
    routes.py           # REST endpoints
    litellm_format.py   # Conversión a model_prices_and_context_window.json
  mcp/server.py         # FastMCP con tools list/search/get_details
```

## Decisiones de diseño

- **Dedup "all merged"**: todas las variantes se conservan como aliases (union-find);
  el slug canónico se elige por el que se considera más representativo. No hay una
  fuente "maestra" de identidades.
- **Variantes con precio propio**: las variantes provider-específicas con precio
  distinto (p. ej. `azure/eu/gpt-4o-2024-08-06`, `bedrock/...`) no se fusionan en el
  canónico sino que se emiten como claves independientes en el cost map con su
  precio real, exactamente igual que hacen los mapas oficiales de LiteLLM.
- **Tolerancia a fallos**: cada fuente fetchea de forma independiente con retries
  exponenciales (0.5s→8s, hasta 3 intentos) solo en errores de red/5xx; un 4xx no se
  reintenta. Una fuente caída no impide que el resto se cargue.
- **Modelos de usuario**: se fusionan con el catálogo público; los agentes
  personalizados de OpenWebUI (sin precio público) se sirven igual, con
  `litellm_provider` informativo.
```