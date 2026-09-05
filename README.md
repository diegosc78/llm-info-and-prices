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
| LiveBench | `livebench.ai/table_YYYY_MM_DD.csv` (la más reciente, vía GitHub) | Puntuaciones reales (54 modelos): overall, coding, agentic coding, reasoning, math... |
| **Tu LiteLLM** | `/v1/model/info` (fallback `/v1/models`) | Modelos de tus instancias + modelo origen (`litellm_params.model`) |
| **Tu OpenWebUI** | `/api/models/list` + detalle en `info` | Modelos/agentes de tu instancia + `base_model_id` y descripción |

Los modelos de tus instancias (LiteLLM + OpenWebUI) se fusionan con los de las
fuentes públicas: si el mismo modelo existe en abierto, hereda la información/price
pública y queda marcado como `instances: ["litellm_user", "openwebui"]`.

### Resolución de modelo base (precios subyacentes)

Los modelos personalizados/agentes y los aliases de tu proxy no tienen precio propio,
pero sí un **modelo base/origen**:

- OpenWebUI: `info.base_model_id` (el modelo que el agente usa por debajo).
- LiteLLM: `litellm_params.model` (el modelo origen del proxy).

Se sigue la cadena recursivamente hasta un modelo con datos reales y se heredan:
- **Precios** del eslabón más profundo que tenga alguno.
- **Ventana de contexto** (`context_length`, `max_input_tokens`, `max_output_tokens`)
  y `litellm_provider` del eslabón más profundo que los tenga.

Solo se rellena lo que falte; nunca se sobreescribe un valor propio.
Ejemplo real:

```
traductor-tecnlogo (OpenWebUI) -> base_model_id: openrouter/google/gemini-3-flash-preview
  => pricing: input 5e-7, output 3e-6  (pricing_source: "derived from openrouter/google/gemini-3-flash-preview")
  => window: max_input_tokens 1048576, max_output_tokens 65536
     (resolved_context_from: openrouter/google/gemini-3-flash-preview)
```

Los campos `base_model_id`, `resolved_price_from` y `resolved_context_from` se
exponen en la API REST y MCP. En el cost map LiteLLM estos modelos ya aparecen
con su precio y ventana derivados.

### Puntuaciones de benchmarks (LiveBench)

El fetcher de LiveBench descarga la tabla mensual más reciente (detección por
GitHub con fallback) y calcula 8 ejes: `overall`, `coding`, `agentic_coding`,
`reasoning`, `math`, `data_analysis`, `language` e `instruction_following`.
Cada fila se engancha al modelo correspondiente con un **matching por firma**
(letras como conjunto, dígitos de versión como tupla ordenada): tolera
`claude-opus-4-5` vs `claude-4-5-opus`, pero nunca confunde `gpt-4.5` con
`gpt-5.4` ni etiqueta variantes (codex/nano/chat) con la puntuación de la
familia. El resultado queda en `model.benchmarks.livebench`.

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

# LiveBench (opcional; por defecto detecta la tabla mensual más reciente)
# LIVEBENCH_TABLE_URL=https://livebench.ai/table_2026_06_25.csv

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
| `GET /models/top` | Top 'n' por score de LiveBench con filtros (ver abajo) |
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

# Top 10 para codificación agéntica (por score LiveBench)
curl "http://localhost:8000/models/top?top=10&axis=agentic_coding&dedupe=model"

# Top por valor: agentic coding bueno y barato, en rango de precio, ventana >= 200k
# y con tooling nativo (function calling + structured outputs + razonamiento)
curl "http://localhost:8000/models/top?axis=agentic_coding&sort_by=value&dedupe=model&max_output_per_1m=3&min_context_tokens=200000&capabilities=function_calling&capabilities=structured_outputs&capabilities=reasoning"
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

### `/models/top` — selección por capacidades

| Parámetro | Uso |
|-----------|-----|
| `top` | Cuántos devolver (1-200) |
| `axis` | Eje de LiveBench: `overall`, `coding`, `agentic_coding`, `reasoning`, `math`, `data_analysis`, `language`, `instruction_following` (def. `agentic_coding`) |
| `min_score` | Score mínimo en ese eje |
| `min_context_tokens` | Ventana mínima (usa `max_input_tokens` o `context_length`) |
| `max_input_per_1m` / `max_output_per_1m` | Rango de precio, USD por millón de tokens |
| `capabilities` | Capacidades requeridas (repetible; todas deben ser true): `function_calling`, `structured_outputs`, `reasoning`, `code_execution`... |
| `sort_by` | `score` (eje desc), `price` (output asc), `value` (score / $/1M output) |
| `dedupe` | `model` colapsa variantes regionales/gateway/instancia que comparten `livebench_id` |
| `q`, `provider`, `mode`, `user_instances` | Mismos filtros del listado |

Cada item devuelve `score`, `value_score`, `price_per_1m` y el modelo serializado
(con `benchmarks`, capacidades y precios) para decidir en detalle.

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
  models.py             # ModelData (con benchmarks), SourcePayload, SourceStatus
  merge.py              # Dedup, fusión, variantes y enganche de benchmarks (LiveBench)
  fetchers/
    base.py             # Fetcher base con retries (sin retry en 4xx)
    litellm_upstream.py # GitHub json oficial de LiteLLM
    portkey.py          # pricing/general por provider (paralelo, Sin retry 4xx)
    cloudprice.py       # ai.cloudprice.net: flat map + catálogo con aliases
    openrouter.py       # GET /models con paginación
    livebench.py        # Scores LiveBench (tabla mensual más reciente)
    litellm_user.py     # tu proxy: /v1/model/info, /v1/models, /v1/model_group/info
    openwebui.py        # /api/models/list + detalle por modelo (rate-limited)
  api/
    routes.py           # REST endpoints (incl. /models/top)
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