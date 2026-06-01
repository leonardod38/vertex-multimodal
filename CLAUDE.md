# vertex-multimodal

Projeto Python para integração com Vertex AI e modelos multimodais do Google.

---

## Ambiente

- Python 3.11.4
- Ambiente virtual: `.venv/` (ativar com `.\.venv\Scripts\Activate.ps1`)
- Launcher: usar `py` em vez de `python` (alias não está no PATH)

## Estrutura

```
vertex-multimodal/
├── .venv/          # ambiente virtual (não commitado)
├── .env            # variáveis de ambiente (não commitado)
├── .gitignore
├── main.py         # ponto de entrada principal
└── CLAUDE.md
```

## Padrões do projeto

- Logging com `logging` — nunca `print()` para debug
- Configurações sensíveis (API keys, URLs) via `.env` com `os.environ.get()`
- Testes com `pytest`
- Versão no formato `MAJOR.MINOR.PATCH` documentada no topo do módulo principal

## Git

- Remote: https://github.com/leonardod38/vertex-multimodal
- Branch principal: `master`
- `.venv/` e `.env` estão no `.gitignore`
