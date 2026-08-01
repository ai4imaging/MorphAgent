"""Batched LLM review for ambiguous semantic redundancy."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from config import make_chat_llm


class LLMRedundancyReviewer:
    """Review ambiguous redundancy groups in batched LLM calls."""

    def __init__(self) -> None:
        self.llm = make_chat_llm(temperature=0)
        self.max_chars_per_chunk = 24000

    def review_groups(self, groups: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Review connected groups and return parsed resolutions."""

        if not groups:
            return {"success": True, "groups": [], "chunks": []}

        chunks = self._chunk_groups(groups)
        all_resolutions: List[Dict[str, Any]] = []
        chunk_records: List[Dict[str, Any]] = []

        for chunk_index, chunk in enumerate(chunks, start=1):
            payload = json.dumps(chunk, indent=2, ensure_ascii=False)
            chunk_record: Dict[str, Any] = {
                "chunk_index": chunk_index,
                "group_count": len(chunk),
                "input_group_ids": [group["group_id"] for group in chunk],
                "request_payload": chunk,
            }
            try:
                response = self.llm.invoke(
                    [
                        SystemMessage(content=self._system_prompt()),
                        HumanMessage(content=self._user_prompt(payload)),
                    ]
                )
                content = response.content if hasattr(response, "content") else str(response)
                parsed = self._parse_response(content)
                chunk_record["raw_response"] = content
                chunk_record["parsed_response"] = parsed
                all_resolutions.extend(parsed)
            except Exception as exc:
                chunk_record["error"] = repr(exc)
                return {
                    "success": False,
                    "error": repr(exc),
                    "groups": [],
                    "chunks": chunk_records + [chunk_record],
                }
            chunk_records.append(chunk_record)

        return {"success": True, "groups": all_resolutions, "chunks": chunk_records}

    def _chunk_groups(self, groups: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        chunks: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_size = 0
        for group in groups:
            serialized = json.dumps(group, ensure_ascii=False)
            if current and current_size + len(serialized) > self.max_chars_per_chunk:
                chunks.append(current)
                current = []
                current_size = 0
            current.append(group)
            current_size += len(serialized)
        if current:
            chunks.append(current)
        return chunks

    def _system_prompt(self) -> str:
        return (
            "You are reviewing feature redundancy decisions for microscopy feature engineering.\n"
            "Be conservative: prefer keeping the older or numerically stronger feature unless the new feature is clearly better or explicitly complementary.\n"
            "Return ONLY valid JSON.\n"
            "For each group return an object with keys: group_id, decision, keep_feature_ids, drop_feature_ids, rationale.\n"
            "Allowed decision values: keep_old, keep_new, keep_both.\n"
            "Use keep_both only when the features are genuinely complementary and should coexist."
        )

    def _user_prompt(self, payload: str) -> str:
        return (
            "Review the following ambiguous redundancy groups.\n"
            "Each group already passed a deterministic pre-screen and needs a conservative semantic decision.\n"
            "Groups:\n"
            f"{payload}\n"
        )

    def _parse_response(self, content: str) -> List[Dict[str, Any]]:
        text = content.strip()
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("[")
            end = text.rfind("]")
            if start == -1 or end == -1:
                raise
            parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            raise ValueError("LLM redundancy review must return a JSON list.")
        return parsed
