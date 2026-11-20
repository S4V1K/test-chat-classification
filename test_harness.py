#!/usr/bin/env python3

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any, Mapping

DEFAULT_SEVERITY: dict[str, int] = {"—": 0, "P3": 1, "P2": 2, "P1": 3}

CLAIM_TYPES = frozenset(
    {
        "delay_asserted",
        "eta_corrected",
        "delay_canceled",
        "docs_asserted",
        "manager_confirm_asserted",
        "receipt_asserted",
        "dispute",
        "other",
    }
)

# Claims, которые описывают время прибытия. Среди них остаётся только последний по ts.
ETA_CLAIM_TYPES = frozenset(
    {"delay_asserted", "eta_corrected", "delay_canceled"}
)

# Грубый фильтр с ВЫСОКИМ recall: перестраховываемся, LLM потом выкинет лишнее.
COARSE_KEYWORDS = (
    "задерж",
    "приеду",
    "приедет",
    " eta",
    "срок",
    "до 10:00",
    "документ",
    "фото",
    "отправил",
    "передал",
    "руководител",
    "обсудил",
    "в курсе",
    "согласовал",
    "подтверд",
    "получил",
    "проверил",
    "принял",
    "карт",
    "спор",
    "не хват",
    "расхожд",
    "одобр",
    "разрешил",
    "застр",
    "ошибся",
    "передумал",
    "отменяю",
    "на самом деле",
    "вовремя",
    "уже не",
)

RAW_LIMIT = 300
MIN_CONFIDENCE = 0.7
QUOTE_THRESHOLD = 0.8

# Размытые временны́е выражения. Ловим детерминированно, без участия LLM.
VAGUE_ETA = re.compile(
    r"(?:ближе\s+)?к\s+часу(?:\s+дня)?"
    r"|около\s+часа"
    r"|примерно\s+в\s+час"
    r"|в\s+районе\s+часа"
    r"|после\s+обеда"
    r"|после\s+полудня"
    r"|скоро"
    r"|чуть\s+позже"
    r"|через\s+\w+\s+мин(?:ут)?",
    re.IGNORECASE,
)
COARSE_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")

RESCUE_WINDOW_SEC = 15 * 60
RESCUE_IGNORED_ROLES = frozenset({"bot", "other"})


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    cases: Path
    instruction: Path
    out: Path
    dry_run: bool
    fixture: bool
    no_coarse_filter: bool
    show_prompts: bool
    base_url: str
    api_key: str
    model: str
    prompt_version: str
    timeout: int = 120
    rules: Path = Path("rules.json")
    explain: str | None = None

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> Config:
        parser = argparse.ArgumentParser(
            description="Тест AI-автоматизации: CRM + чат -> правила -> сводка"
        )
        parser.add_argument("--cases", default="cases.json", help="файл с кейсами")
        parser.add_argument(
            "--instruction",
            default="AGENTS.md",
            help="промпт агента: секции ## SYSTEM и ## USER",
        )
        parser.add_argument("--out", default="runs/runs.jsonl", help="файл лога JSONL")
        parser.add_argument(
            "--rules",
            default="rules.json",
            help="правила в виде данных (JSON, DSL условий)",
        )
        parser.add_argument(
            "--explain",
            metavar="CASE_ID",
            help="показать полный разбор одного кейса: промпты, ответы, проверки, правила",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="не вызывать LLM, проверить только правила",
        )
        parser.add_argument(
            "--fixture",
            action="store_true",
            help="подставить ожидаемые claims вместо LLM: проверка пайплайна и метрик без модели",
        )
        parser.add_argument(
            "--no-coarse-filter",
            action="store_true",
            help="отправить в модель все сообщения, включая шум",
        )
        parser.add_argument(
            "--show-prompts",
            action="store_true",
            help="печатать системный и пользовательский промпт",
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default=int(os.getenv("LLM_TIMEOUT", "120")),
            help="таймаут одного запроса к LLM в секундах (полезно при холодном старте Ollama)",
        )
        args = parser.parse_args(argv)

        return cls(
            cases=Path(args.cases),
            instruction=Path(args.instruction),
            out=Path(args.out),
            dry_run=args.dry_run,
            fixture=args.fixture,
            no_coarse_filter=args.no_coarse_filter,
            show_prompts=args.show_prompts,
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("LLM_API_KEY", "ollama"),
            model=os.getenv("LLM_MODEL", "llama3.2"),
            prompt_version=os.getenv("PROMPT_VERSION", "extract_v3"),
            timeout=args.timeout,
            rules=Path(args.rules),
            explain=args.explain,
        )


# ---------------------------------------------------------------------------
# Инструкция для модели
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instruction:
    system: str
    user: Template

    @classmethod
    def load(cls, path: Path) -> Instruction:
        text = path.read_text(encoding="utf-8")
        try:
            system = text.split("## SYSTEM", 1)[1].split("## USER", 1)[0].strip()
            user = text.split("## USER", 1)[1].strip()
        except IndexError as exc:
            raise SystemExit(f"В {path} нет секций ## SYSTEM и ## USER") from exc
        return cls(system=system, user=Template(user))

    @staticmethod
    def _history(case: dict[str, Any], message: Mapping[str, Any]) -> str:
        earlier = [
            m
            for m in case["chat"]
            if m["id"] != message["id"]
            and m["author_role"] == message["author_role"]
            and m["ts"] <= message["ts"]
        ]
        earlier.sort(key=lambda m: m["ts"])
        if not earlier:
            return "нет"
        return "; ".join(f"id={m['id']} «{m['text']}»" for m in earlier[-3:])

    def render(self, case: dict[str, Any], message: Mapping[str, Any]) -> str:
        return self.user.substitute(
            case_id=case["id"],
            case_type=case["type"],
            crm=json.dumps(case["crm"], ensure_ascii=False),
            history=self._history(case, message),
            msg_id=message["id"],
            author=message["author"],
            author_role=message["author_role"],
            ts=message["ts"],
            text=message["text"],
        )


# ---------------------------------------------------------------------------
# Извлечение фактов из чата
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    message_id: str
    author: str
    ts: str
    claim_type: str
    author_role: str
    eta: str | None
    eta_precision: str
    eta_hint: str | None
    evidence: str
    confidence: float
    note: str
    supersedes: str | None
    raw: str


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value).replace(tzinfo=None)


def coarse_filter(text: str) -> bool:
    """Дешёвый фильтр с высоким recall:keywords + часы + размытые времена."""
    low = text.lower()
    return (
        COARSE_TIME.search(text) is not None
        or VAGUE_ETA.search(text) is not None
        or any(keyword in low for keyword in COARSE_KEYWORDS)
    )


def contextual_rescue(
    message: Mapping[str, Any],
    anchors_by_author: Mapping[str, list[datetime]],
) -> bool:
    """Пустить сообщение, если его автор недавно уже дал значимую реплику."""
    ts = parse_dt(message["ts"])
    if ts is None:
        return False
    return any(
        abs((ts - anchor).total_seconds()) <= RESCUE_WINDOW_SEC
        for anchor in anchors_by_author.get(message["author"], ())
    )


def evidence_ok(evidence: str, text: str) -> bool:
    """Дословная подстрока или почти дословная (пунктуация/регистр/пробелы)."""
    if evidence in text:
        return True

    def norm(value: str) -> str:
        return re.sub(r"[\s.,!?;:«»\"'()\-]+", "", value.lower())

    needle = norm(evidence)
    if not needle:
        return False
    haystack = norm(text)
    match = difflib.SequenceMatcher(None, needle, haystack).find_longest_match(
        0, len(needle), 0, len(haystack)
    )
    return match.size / len(needle) >= QUOTE_THRESHOLD


def classify_eta(
    value: Any, text: str
) -> tuple[str | None, str, str | None, bool]:
    """Возвращает (eta, precision, hint, dropped_exact).

    precision: exact | vague | none.
    dropped_exact=True, если модель назвала часы, которых нет в тексте.
    """
    raw = str(value).strip() if value is not None else ""

    if re.fullmatch(r"\d{1,2}:\d{2}", raw) and raw in text:
        return raw, "exact", None, False

    vague = VAGUE_ETA.search(text)
    hint = vague.group(0) if vague else None

    if raw:
        # Модель дала время (или что-то на него похожее), но оно не подтверждено текстом.
        return None, ("vague" if hint else "none"), hint, True

    if hint:
        return None, "vague", hint, False

    return None, "none", None, False


def normalize_supersedes(value: Any) -> str | None:
    """Приводит поле supersedes к строке id или None."""
    if value is None:
        return None
    return str(value).strip() or None


def parse_json_loose(text: str) -> dict[str, Any]:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end > start:
        s = s[start : end + 1]
    out = json.loads(s)
    if not isinstance(out, dict):
        raise ValueError("ожидался JSON-объект")
    return out


class LLMClient:
    """Stateless обёртка над OpenAI-совместимым /chat/completions."""

    def __init__(self, config: Config) -> None:
        base = config.base_url.rstrip("/")
        self.endpoint = base if base.endswith("/v1") else base + "/v1"
        self.endpoint += "/chat/completions"
        self.api_key = config.api_key
        self.model = config.model
        self.timeout = config.timeout

    def complete(self, system: str, user: str) -> str:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(
                {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0,
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]


@dataclass
class Extraction:
    claims: list[Claim] = field(default_factory=list)
    review: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    messages_seen: int = 0
    messages_filtered: int = 0
    messages_rescued: int = 0
    llm_signals: int = 0
    valid_quotes: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)


def effective_claims(claims: list[Claim]) -> list[Claim]:
    """Оставляет последнее ETA-утверждение каждого автора.

    `supersedes` сохраняется в логе как улика, но не влияет на выбор:
    побеждает максимальный ts; при равенстве — последний в порядке чата.
    """
    eta_claims: dict[str, list[Claim]] = {}
    for claim in claims:
        if claim.claim_type in ETA_CLAIM_TYPES or claim.eta is not None:
            eta_claims.setdefault(claim.author, []).append(claim)

    stale: set[str] = set()
    for author_claims in eta_claims.values():
        if len(author_claims) < 2:
            continue
        _, winner = max(
            enumerate(author_claims),
            key=lambda pair: (pair[1].ts, pair[0]),
        )
        stale.update(
            c.message_id for c in author_claims if c.message_id != winner.message_id
        )

    return [c for c in claims if c.message_id not in stale]


class ClaimExtractor:
    """Фильтрует сообщения, вызывает модель и проверяет её ответ."""

    def __init__(self, config: Config, llm: LLMClient | None) -> None:
        self.config = config
        self.llm = llm

    def extract(self, case: dict[str, Any], instruction: Instruction) -> Extraction:
        result = Extraction()
        expected_by_msg = {
            claim["message_id"]: claim for claim in case["expected"]["claims"]
        }

        anchors_by_author: dict[str, list[datetime]] = {}
        for item in case["chat"]:
            if item["author_role"] in RESCUE_IGNORED_ROLES:
                continue
            if not coarse_filter(item["text"]):
                continue
            ts = parse_dt(item["ts"])
            if ts:
                anchors_by_author.setdefault(item["author"], []).append(ts)

        for message in case["chat"]:
            result.messages_seen += 1
            passed = coarse_filter(message["text"])
            rescued = False
            if not self.config.no_coarse_filter:
                rescued = contextual_rescue(message, anchors_by_author)
                if not passed and not rescued:
                    result.messages_filtered += 1
                    trace = self._trace(result, case, message)
                    if trace is not None:
                        trace["decision"] = "отсеяно грубым фильтром"
                    continue
                if rescued and not passed:
                    result.messages_rescued += 1

            trace = self._trace(result, case, message)
            if trace is not None:
                trace["filter"] = {"coarse": passed, "rescued": rescued and not passed}

            if self.config.dry_run:
                if trace is not None:
                    trace["decision"] = "dry-run: модель не вызывалась"
                continue

            prompt = instruction.render(case, message)
            if trace is not None:
                trace["prompt"] = {"system": instruction.system, "user": prompt}

            try:
                raw = self._raw_response(
                    message["id"], expected_by_msg, prompt, instruction
                )
                output = parse_json_loose(raw)
            except (
                urllib.error.URLError,
                ValueError,
                KeyError,
                json.JSONDecodeError,
            ) as exc:
                result.errors.append({"message_id": message["id"], "error": str(exc)})
                if trace is not None:
                    trace["decision"] = f"ошибка: {exc}"
                continue

            if trace is not None:
                trace["raw"] = raw
                trace["parsed"] = output

            if not output.get("signal"):
                if trace is not None:
                    trace["decision"] = "шум (signal=false)"
                continue
            result.llm_signals += 1

            evidence = str(output.get("evidence") or "")
            confidence = self._confidence(output.get("confidence"))
            if not evidence_ok(evidence, message["text"]):
                result.review.append(
                    {
                        "message_id": message["id"],
                        "reason": "цитата не найдена в тексте",
                        "evidence": evidence,
                        "raw": raw[:RAW_LIMIT],
                    }
                )
                if trace is not None:
                    trace["checks"] = {"quote_ok": False, "confidence": confidence}
                    trace["decision"] = "review: цитата не найдена в тексте"
                continue
            if confidence < MIN_CONFIDENCE:
                result.review.append(
                    {
                        "message_id": message["id"],
                        "reason": f"низкая уверенность {confidence}",
                        "raw": raw[:RAW_LIMIT],
                    }
                )
                if trace is not None:
                    trace["checks"] = {"quote_ok": True, "confidence": confidence}
                    trace["decision"] = f"review: низкая уверенность {confidence}"
                continue

            result.valid_quotes += 1
            claim_type = str(output.get("claim_type"))
            eta, eta_precision, eta_hint, eta_dropped = classify_eta(
                output.get("eta"), message["text"]
            )
            note = str(output.get("note") or "")
            if eta_dropped:
                note = (note + "; ETA не найден в тексте, сброшен").strip("; ")
            if eta_precision == "vague":
                note = (note + f"; размытое время «{eta_hint}» — уточнить").strip("; ")
            claim = Claim(
                message_id=message["id"],
                author=message["author"],
                ts=message["ts"],
                claim_type=claim_type if claim_type in CLAIM_TYPES else "other",
                author_role=message["author_role"],
                eta=eta,
                eta_precision=eta_precision,
                eta_hint=eta_hint,
                evidence=evidence,
                confidence=confidence,
                note=note,
                supersedes=normalize_supersedes(output.get("supersedes")),
                raw=raw[:RAW_LIMIT],
            )
            result.claims.append(claim)
            if trace is not None:
                trace["checks"] = {
                    "quote_ok": True,
                    "confidence": confidence,
                    "eta_raw": output.get("eta"),
                    "eta": eta,
                    "eta_precision": eta_precision,
                    "eta_hint": eta_hint,
                    "eta_dropped": eta_dropped,
                }
                trace["claim"] = asdict(claim)
                trace["decision"] = "claim принят"
        return result

    def _trace(
        self,
        result: Extraction,
        case: dict[str, Any],
        message: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Собирает диагностику только для кейса, запрошенного через --explain."""
        if self.config.explain != case["id"]:
            return None
        entry: dict[str, Any] = {
            "message_id": message["id"],
            "author": message["author"],
            "author_role": message["author_role"],
            "ts": message["ts"],
            "text": message["text"],
        }
        result.trace.append(entry)
        return entry

    def _raw_response(
        self,
        message_id: str,
        expected_by_msg: dict[str, dict[str, Any]],
        prompt: str,
        instruction: Instruction,
    ) -> str:
        if self.config.fixture:
            want = expected_by_msg.get(message_id)
            if not want:
                return json.dumps({"signal": False})
            return json.dumps(
                {
                    "signal": True,
                    "claim_type": want["claim_type"],
                    "eta": want.get("eta"),
                    "evidence": want["evidence"],
                    "confidence": 0.95,
                    "supersedes": want.get("supersedes"),
                    "note": "fixture",
                },
                ensure_ascii=False,
            )
        if self.llm is None:
            raise RuntimeError("LLM-клиент не создан")
        return self.llm.complete(instruction.system, prompt)

    @staticmethod
    def _confidence(value: Any) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0


# ---------------------------------------------------------------------------
# Правила и вердикт
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Facts:
    id: str
    type: str
    crm: dict[str, Any]
    now: str
    claims: tuple[Claim, ...]


# ---------------------------------------------------------------------------
# Правила как данные: небольшой DSL условий
# ---------------------------------------------------------------------------


def resolve_field(obj: Any, path: str) -> Any:
    """Читает 'crm.deadline' из Facts и 'claim_type' из Claim одинаково."""
    current = obj
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current


def matches(node: Mapping[str, Any], obj: Any, root: Any | None = None) -> bool:
    """Вычисляет условие из rules.json. root — исходный Facts для сравнения полей."""
    if root is None:
        root = obj

    if "all" in node:
        return all(matches(child, obj, root) for child in node["all"])
    if "any" in node:
        return any(matches(child, obj, root) for child in node["any"])
    if "not" in node:
        return not matches(node["not"], obj, root)

    op = node.get("op")
    value = resolve_field(obj, node["field"]) if "field" in node else None

    if op == "eq":
        return value == node["value"]
    if op == "ne":
        return value != node["value"]
    if op == "in":
        return value in node["value"]
    if op == "truthy":
        return bool(value)
    if op == "falsy":
        return not bool(value)
    if op == "some":
        return any(matches(node["where"], item, root) for item in (value or ()))
    if op == "lte_field":
        left = parse_dt(value)
        right = parse_dt(resolve_field(root, node["value"]))
        return bool(left and right and left <= right)
    raise ValueError(f"неизвестный оператор DSL: {op!r}")


@dataclass(frozen=True)
class Rule:
    id: str
    severity: str
    action: str
    when: Mapping[str, Any]


@dataclass(frozen=True)
class RuleHit:
    rule: Rule
    matched: bool
    clauses: tuple[tuple[str, bool], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.rule.id,
            "severity": self.rule.severity,
            "action": self.rule.action,
            "matched": self.matched,
            "clauses": [{"condition": text, "ok": ok} for text, ok in self.clauses],
        }


@dataclass(frozen=True)
class Verdict:
    priority: str
    rule_ids: list[str]
    actions: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuleEngine:
    def __init__(
        self,
        rules: tuple[Rule, ...],
        severity: Mapping[str, int] | None = None,
    ) -> None:
        self.rules = rules
        self.severity = dict(severity or DEFAULT_SEVERITY)

    @classmethod
    def load(cls, path: Path) -> RuleEngine:
        data = json.loads(path.read_text(encoding="utf-8"))
        rules = tuple(
            Rule(item["id"], item["severity"], item["action"], item["when"])
            for item in data["rules"]
        )
        return cls(rules, data.get("severity_order") or DEFAULT_SEVERITY)

    def evaluate(self, facts: Facts) -> list[RuleHit]:
        hits: list[RuleHit] = []
        for rule in self.rules:
            node = rule.when
            key = "all" if "all" in node else "any" if "any" in node else None
            children = node[key] if key else [node]
            matched = matches(node, facts)
            clauses = tuple(
                (json.dumps(child, ensure_ascii=False), matches(child, facts))
                for child in children
            )
            hits.append(RuleHit(rule, matched, clauses))
        return hits

    def decide(self, facts: Facts) -> Verdict:
        hits = [hit.rule for hit in self.evaluate(facts) if hit.matched]
        if not hits:
            return Verdict("P2", ["FALLBACK"], ["Ручная сверка: неизвестный сценарий"])

        top = max(hits, key=lambda rule: self.severity.get(rule.severity, 0))
        rule_ids = [rule.id for rule in hits]
        if top.severity == "—":
            return Verdict("—", rule_ids, [top.action])

        actions: list[str] = []
        for rule in sorted(hits, key=lambda r: -self.severity.get(r.severity, 0)):
            if rule.action not in actions:
                actions.append(rule.action)
        return Verdict(top.severity, rule_ids, actions)


# ---------------------------------------------------------------------------
# Результат кейса
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    prompt_version: str
    model: str
    crm_summary: str
    claims: tuple[Claim, ...]
    review: tuple[dict[str, Any], ...]
    errors: tuple[dict[str, str], ...]
    messages_seen: int
    messages_filtered: int
    messages_rescued: int
    llm_signals: int
    valid_quotes: int
    verdict: Verdict
    expected: dict[str, Any]

    def as_log(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "case_id": self.case_id,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "crm_summary": self.crm_summary,
            "claims": [asdict(claim) for claim in self.claims],
            "review": list(self.review),
            "errors": list(self.errors),
            "messages_seen": self.messages_seen,
            "messages_filtered": self.messages_filtered,
            "messages_rescued": self.messages_rescued,
            "llm_signals": self.llm_signals,
            "valid_quotes": self.valid_quotes,
        }
        row.update(self.verdict.as_dict())
        row.update(
            {
                "expected_priority": self.expected["priority"],
                "expected_need_action": self.expected["need_action"],
                "expected_rule_ids": self.expected["rule_ids"],
                "expected_claims": self.expected["claims"],
            }
        )
        return row


def crm_summary(case: dict[str, Any]) -> str:
    crm, kind = case["crm"], case["type"]
    if kind == "onboarding":
        docs = (
            "ожидаются" if crm.get("docs") == "expected" else (crm.get("docs") or "—")
        )
        return f"Первый выход {crm.get('first_day', '—')}. Документы: {docs}"
    if kind == "offboarding":
        conf = crm.get("manager_confirmation") or "отсутствует"
        return f"Последний день {crm.get('last_day', '—')}. Подтверждение руководителя: {conf}"
    if crm.get("status") == "Принято":
        accepted = str(crm.get("accepted_at", "—")).split("T")[-1][:5]
        return f"Приёмка {accepted}. Статус: Принято"
    deadline = str(crm.get("deadline", "—")).split("T")[-1][:5]
    return f"Доставка до {deadline}. Статус: {crm.get('status', '—')}"


class Harness:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.instruction = Instruction.load(config.instruction)
        self.data = json.loads(config.cases.read_text(encoding="utf-8"))
        llm = None if config.dry_run or config.fixture else LLMClient(config)
        self.extractor = ClaimExtractor(config, llm)
        self.engine = RuleEngine.load(config.rules)
        self.traces: dict[str, list[dict[str, Any]]] = {}

    def run(self) -> list[CaseResult]:
        return [self._run_case(case) for case in self.data["cases"]]

    def _run_case(self, case: dict[str, Any]) -> CaseResult:
        extraction = self.extractor.extract(case, self.instruction)
        if extraction.trace:
            self.traces[case["id"]] = extraction.trace
        claims = effective_claims(extraction.claims)
        facts = Facts(
            id=case["id"],
            type=case["type"],
            crm=case["crm"],
            now=self.data["now"],
            claims=tuple(claims),
        )
        return CaseResult(
            case_id=case["id"],
            prompt_version=self.config.prompt_version,
            model=self.config.model,
            crm_summary=crm_summary(case),
            claims=tuple(claims),
            review=tuple(extraction.review),
            errors=tuple(extraction.errors),
            messages_seen=extraction.messages_seen,
            messages_filtered=extraction.messages_filtered,
            messages_rescued=extraction.messages_rescued,
            llm_signals=extraction.llm_signals,
            valid_quotes=extraction.valid_quotes,
            verdict=self.engine.decide(facts),
            expected=case["expected"],
        )

    def show_prompts(self) -> None:
        demo = next(
            (case for case in self.data["cases"] if case["id"] == "П-08"),
            self.data["cases"][2],
        )
        message = demo["chat"][0]
        print("=== SYSTEM ===\n" + self.instruction.system)
        print("\n=== USER (П-08, курьер) ===")
        print(self.instruction.render(demo, message))


# ---------------------------------------------------------------------------
# Метрики и отчёт
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    p1_recall: float
    false_close: list[str]
    unknown_rate: float
    rule_match: str
    claim_precision: float
    claim_recall: float
    quote_valid: str
    vague_eta: int
    context_rescued: int
    noise_rejected: str
    llm_errors: int
    review_queue: int

    @classmethod
    def from_results(cls, results: list[CaseResult]) -> Metrics:
        expected_p1 = [r for r in results if r.expected["priority"] == "P1"]
        p1_recall = (
            sum(1 for r in expected_p1 if r.verdict.priority == "P1") / len(expected_p1)
            if expected_p1
            else 1.0
        )
        false_close = [
            r.case_id
            for r in results
            if r.verdict.priority == "—" and r.expected["need_action"]
        ]
        fallback = [r for r in results if r.verdict.rule_ids == ["FALLBACK"]]
        rule_match = sum(
            1
            for r in results
            if sorted(r.verdict.rule_ids) == sorted(r.expected["rule_ids"])
        )

        expected_claims: Counter = Counter()
        predicted_claims: Counter = Counter()
        for result in results:
            for claim in result.expected["claims"]:
                expected_claims[_claim_key(result.case_id, claim)] += 1
            for claim in result.claims:
                predicted_claims[_claim_key(result.case_id, asdict(claim))] += 1

        true_positive = sum((expected_claims & predicted_claims).values())
        precision = (
            true_positive / sum(predicted_claims.values()) if predicted_claims else 0.0
        )
        recall = (
            true_positive / sum(expected_claims.values()) if expected_claims else 0.0
        )

        signals = sum(r.llm_signals for r in results)
        valid_quotes = sum(r.valid_quotes for r in results)
        seen = sum(r.messages_seen for r in results)
        filtered = sum(r.messages_filtered for r in results)

        return cls(
            p1_recall=p1_recall,
            false_close=false_close,
            unknown_rate=len(fallback) / len(results) if results else 0.0,
            rule_match=f"{rule_match}/{len(results)}",
            claim_precision=precision,
            claim_recall=recall,
            quote_valid=f"{valid_quotes}/{signals}" if signals else "—",
            vague_eta=sum(
                1 for r in results for c in r.claims if c.eta_precision == "vague"
            ),
            context_rescued=sum(r.messages_rescued for r in results),
            noise_rejected=f"{filtered}/{seen}",
            llm_errors=sum(len(r.errors) for r in results),
            review_queue=sum(len(r.review) for r in results),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _claim_key(case_id: str, claim: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        case_id,
        claim["message_id"],
        claim["claim_type"],
        claim["author_role"],
        claim.get("eta"),
        claim.get("supersedes"),
    )


def render_digest(results: list[CaseResult]) -> str:
    lines = ["# Сводка для support-отдела", ""]
    lines.append("| Кейс | CRM | Чат | Проверить / действие | Приоритет |")
    lines.append("|---|---|---|---|---|")
    for result in results:
        chat_parts: list[str] = []
        for claim in result.claims:
            part = f"[{claim.author_role}] «{claim.evidence}»"
            if claim.eta_precision == "vague":
                part += f" (размыто: {claim.eta_hint})"
            chat_parts.append(part)
        chat = "; ".join(chat_parts) or "значимых сообщений нет"
        lines.append(
            f"| {result.case_id} | {result.crm_summary} | {chat} "
            f"| {'; '.join(result.verdict.actions)} | {result.verdict.priority} |"
        )

    p1_rows = [r for r in results if r.verdict.priority == "P1"]
    first = (
        min(p1_rows, key=lambda r: 0 if "SLA_BREACH" in r.verdict.rule_ids else 1)
        if p1_rows
        else None
    )
    lines.append("")
    if first:
        reason = " — срок уже нарушен" if "SLA_BREACH" in first.verdict.rule_ids else ""
        lines.append(f"**Начать с:** {first.case_id}{reason}")
    else:
        lines.append("**Начать с:** P1-кейсов нет")
    return "\n".join(lines)


def write_log(path: Path, results: list[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result.as_log(), ensure_ascii=False) + "\n")


def _indent(text: str, spaces: int = 4) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def render_explain(
    config: Config,
    case_id: str,
    harness: Harness,
    results: list[CaseResult],
) -> str:
    case = next((c for c in harness.data["cases"] if c["id"] == case_id), None)
    if case is None:
        known = ", ".join(c["id"] for c in harness.data["cases"])
        return f"Кейс {case_id!r} не найден. Доступные: {known}"

    result = next(r for r in results if r.case_id == case_id)
    facts = Facts(
        id=case["id"],
        type=case["type"],
        crm=case["crm"],
        now=harness.data["now"],
        claims=tuple(result.claims),
    )

    lines = [
        f"=== Разбор кейса {case_id} ===",
        f"Модель: {config.model} | промпт: {config.prompt_version} "
        f"| правила: {config.rules} | now: {harness.data['now']}",
        f"Тип: {case['type']}",
        "CRM: " + json.dumps(case["crm"], ensure_ascii=False),
        "",
        "=== Сообщения ===",
    ]

    for entry in harness.traces.get(case_id, ()):
        lines.append(
            f"\n--- {entry['message_id']} | {entry['author']} "
            f"({entry['author_role']}) | {entry['ts']} ---"
        )
        lines.append(f"  текст: «{entry['text']}»")
        if "filter" in entry:
            lines.append(
                f"  фильтр: coarse={entry['filter']['coarse']}, "
                f"rescue={entry['filter']['rescued']}"
            )
        if "prompt" in entry:
            lines.append("  SYSTEM:\n" + _indent(entry["prompt"]["system"]))
            lines.append("  USER:\n" + _indent(entry["prompt"]["user"]))
        if "raw" in entry:
            lines.append(
                "  ответ модели: "
                + json.dumps(entry.get("parsed", {}), ensure_ascii=False)
            )
        if "checks" in entry:
            lines.append("  проверки: " + json.dumps(entry["checks"], ensure_ascii=False))
        lines.append("  решение: " + entry.get("decision", "—"))
        if "claim" in entry:
            lines.append("  claim: " + json.dumps(entry["claim"], ensure_ascii=False))

    accepted_ids = {
        entry["claim"]["message_id"]
        for entry in harness.traces.get(case_id, ())
        if "claim" in entry
    }
    effective_ids = {claim.message_id for claim in result.claims}
    stale = sorted(accepted_ids - effective_ids)
    lines.extend(
        [
            "",
            "=== Resolver ===",
            f"Принято claim'ов: {len(accepted_ids)}",
            f"Осталось после resolver: {len(result.claims)}",
            "Отброшены как устаревшие: " + (", ".join(stale) or "—"),
            "",
            "=== Правила ===",
        ]
    )
    for hit in harness.engine.evaluate(facts):
        mark = "✓" if hit.matched else "·"
        lines.append(
            f"{mark} {hit.rule.id} [{hit.rule.severity}] — {hit.rule.action}"
        )
        for text, ok in hit.clauses:
            lines.append(f"    {'OK ' if ok else 'NO '} {text}")

    lines.extend(
        [
            "",
            "=== Вердикт ===",
            f"Приоритет: {result.verdict.priority}",
            "Правила: " + ", ".join(result.verdict.rule_ids),
            "Действия: " + "; ".join(result.verdict.actions),
        ]
    )
    if result.review:
        lines.append("Review: " + json.dumps(result.review, ensure_ascii=False))
    if result.errors:
        lines.append("Ошибки: " + json.dumps(result.errors, ensure_ascii=False))
    lines.append(f"Лог: {config.out}")
    return "\n".join(lines)


def print_report(
    config: Config, results: list[CaseResult], metrics: Metrics, log_path: Path
) -> None:
    print(render_digest(results))
    print("\n=== Метрики ===")
    for key, value in metrics.as_dict().items():
        print(f"{key:>18}: {value}")
    print(f"\nЛог: {log_path}")
    if config.dry_run:
        print(
            "ВНИМАНИЕ: --dry-run — модель не вызывалась, claims пусты. Проверена только защита CRM-only."
        )
        print(
            "rule_match будет 3/4: правило DELAY_CONFIRMED требует claim из чата — это ожидаемо."
        )
    elif config.fixture:
        print(
            "ВНИМАНИЕ: --fixture — вместо модели подставлены ожидаемые claims. Проверяется весь пайплайн и метрики."
        )


def main(argv: list[str] | None = None) -> int:
    config = Config.from_args(argv)
    harness = Harness(config)
    if config.show_prompts:
        harness.show_prompts()
        return 0

    results = harness.run()
    write_log(config.out, results)
    if config.explain:
        print(render_explain(config, config.explain, harness, results))
        return 0
    print_report(config, results, Metrics.from_results(results), config.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
