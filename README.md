# Agent Factory

**105 production-ready AI agent prompts + generators that create more. Not just prompts — a self-expanding library.**

> Most "awesome prompt" repos give you static text files. Agent Factory gives you a complete system: curated agent prompts, industry playbooks, a growing glossary, department guides, and **generators that create fresh content using free LLM APIs.**

---

## What's Inside

```
agent-factory/
├── agents/                    # The library
│   ├── agents_data.json       # 84 agents across 12 departments
│   ├── playbooks_data.json    # 20 industry playbooks
│   ├── glossary_data.json     # 49 AI/ML terms (plain English)
│   ├── guides_data.json       # 12 "How to Use AI in [Dept]" guides
│   └── bonus_prompts.json     # Auto-generated bonus prompts
│
├── generators/                # The factory
│   ├── llm_providers.py       # Multi-provider LLM fallback (6 providers)
│   ├── generate_bonus_prompts.py   # Create new agent prompts
│   ├── generate_dept_guides.py     # Create department guides
│   ├── generate_glossary_terms.py  # Expand the glossary
│   └── .env.example           # API key template
│
├── quality-gates/             # Bonus: production quality tools
│   ├── html_invariants.py     # HTML checker (SEO, meta, links)
│   ├── banned_patterns.py     # Secret/debug pattern scanner
│   ├── smoke_test.py          # Post-deploy site health check
│   ├── validate_jsonld.py     # JSON-LD structured data validator
│   ├── check_imports.py       # Python import resolver
│   └── pre-commit-hook        # Git hook for banned patterns
│
└── README.md
```

---

## The Agents (84 across 12 departments)

| Department | Agents | Examples |
|---|---|---|
| Engineering & Dev | 7 | Frontend Developer, Backend Architect, DevOps Engineer |
| Marketing | 7 | Growth Strategist, Content Writer, SEO Specialist |
| Sales | 7 | Sales Coach, Cold Outreach Writer, Deal Analyst |
| HR & People | 7 | Recruiter, Culture Advisor, Interview Designer |
| Finance | 7 | Financial Analyst, Budget Planner, Revenue Modeler |
| Legal & Compliance | 7 | Contract Reviewer, Compliance Officer, IP Advisor |
| Operations | 7 | Process Engineer, Supply Chain Analyst, Quality Manager |
| Customer Support | 7 | Ticket Triager, Knowledge Base Writer, Escalation Handler |
| Product | 7 | Product Manager, Feature Prioritizer, User Researcher |
| Design | 7 | UI Designer, UX Researcher, Design System Architect |
| Data & Analytics | 7 | Data Analyst, ML Engineer, Dashboard Builder |
| Executive | 7 | Strategy Advisor, Board Deck Writer, M&A Analyst |

Each agent includes:
- **Full system prompt** (150-300 words, ready to paste)
- **One-liner** description
- **Deliverables** list
- **Tags** for search/filtering
- **Combine-with** suggestions (which agents work well together)
- **Works-with** compatibility (Claude, GPT-4, Gemini, etc.)

---

## The Playbooks (20 industries)

Real-world AI adoption guides for: Healthcare, Financial Services, Retail, Manufacturing, Education, Legal, Real Estate, Agriculture, Energy, Logistics, Media, Hospitality, Insurance, Telecom, Government, Nonprofit, Construction, Automotive, Pharma, and Cybersecurity.

Each playbook includes:
- Industry summary
- 5 concrete use cases with descriptions
- 5 key takeaways (practical, not theoretical)
- Recommended agents from the library

---

## The Generators (the real differentiator)

Static prompt libraries go stale. Agent Factory **generates fresh content** using free LLM APIs.

### Setup

```bash
cd generators
cp .env.example .env
# Add at least one API key (all providers are free tier)
pip install python-dotenv requests google-genai groq cerebras-cloud-sdk
```

### Generate new agent prompts

```bash
python generate_bonus_prompts.py                    # all departments
python generate_bonus_prompts.py --department=sales  # one department
python generate_bonus_prompts.py --dry-run           # preview first
```

### Generate department guides

```bash
python generate_dept_guides.py              # refresh oldest guide
python generate_dept_guides.py --all        # generate all 12
python generate_dept_guides.py --dry-run    # preview
```

### Expand the glossary

```bash
python generate_glossary_terms.py            # add 5 new terms
python generate_glossary_terms.py --count=10 # add 10 terms
python generate_glossary_terms.py --dry-run  # preview
```

### LLM Provider Fallback

The generators use a multi-provider fallback chain — if one provider is rate-limited, it automatically tries the next:

```
Gemini → Cerebras → CF Workers AI → Groq → SambaNova → OpenRouter
```

All providers offer free tiers. You only need **one** API key to get started.

---

## Quality Gates (bonus)

Production-tested quality tools extracted from a multi-site deployment pipeline. Each script is standalone with zero external dependencies (stdlib only).

| Script | What it does |
|---|---|
| `html_invariants.py` | Checks title, viewport, meta desc, canonical, OG tags, JSON-LD, broken links, broken images, placeholder text |
| `banned_patterns.py` | Scans for hardcoded secrets, debug statements, placeholder text |
| `smoke_test.py` | Concurrent HTTP health checks with content verification |
| `validate_jsonld.py` | Deep JSON-LD validation (required fields per schema type) |
| `check_imports.py` | AST-based Python import checker (no code execution) |
| `pre-commit-hook` | Git hook that catches secrets and debug code before commit |

Install the pre-commit hook:
```bash
ln -sf $(pwd)/quality-gates/pre-commit-hook .git/hooks/pre-commit
```

---

## Quick Start

### Option 1: Just use the prompts

Open `agents/agents_data.json`, find an agent, copy the `prompt` field into your AI tool.

### Option 2: Use as Claude Code agents

```bash
# Copy agents to Claude Code's agent directory
mkdir -p ~/.claude/agents
# Then use agents_data.json to create individual .md files per agent
```

### Option 3: Run the generators

```bash
cd generators
cp .env.example .env
# Add your free API key(s)
pip install python-dotenv requests
python generate_bonus_prompts.py --dry-run
```

---

## How This Was Built

This library was extracted from a production AI pipeline that runs daily across 7+ websites. Every prompt has been tested in real workflows. The generators have been running weekly for months, creating fresh content without human intervention.

Built by [Peter Saddington](https://github.com/agilepeter) — 4+ years of building AI systems, from single prompts to autonomous multi-agent pipelines.

---

## Contributing

1. Fork the repo
2. Add or improve an agent in `agents_data.json`
3. Run the generators to verify your changes don't break anything
4. Open a PR with a clear description

Agent prompts should be:
- **Practical** — someone can copy-paste and get immediate value
- **Specific** — not "you are a helpful assistant"
- **150-300 words** — enough detail to be useful, short enough to be fast
- **Include deliverables** — what does this agent actually produce?

---

## License

MIT — use it however you want.

---

## Stats

- **105** agent prompts (84 curated + 21 generated)
- **12** departments covered
- **20** industry playbooks
- **58** glossary terms (and growing)
- **12** department guides (freshly regenerated)
- **6** quality gate scripts
- **6** LLM providers supported
- **0** external dependencies for quality gates
